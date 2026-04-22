"""
Unit tests for regulatory knowledge base (regulatory_kb.py).

Tests cover:
- Delta-checking (ETag, Last-Modified, SHA-256)
- Chunking and embedding ingestion
- Notification fan-out
- In-memory retriever cache invalidation
"""

import hashlib
import json
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch, Mock

import numpy as np
import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from models import Base, RegulationSource, Notification, User, Organization, utc_now
from regulatory_kb import (
    fetch_if_changed,
    ingest_source,
    _notify_all_users,
    load_sources,
    build_retriever,
    get_retriever,
    _retrievers,
)


@pytest.fixture
def db_session():
    """In-memory SQLite session for testing."""
    engine = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(engine)
    Session = sessionmaker(bind=engine)
    session = Session()

    # Create default org and user for notifications
    org = Organization(name="Test Org")
    session.add(org)
    session.commit()

    user1 = User(
        email="user1@example.com",
        display_name="User One",
        hashed_password="fake",
        api_key="key1",
        organization_id=org.id,
    )
    user2 = User(
        email="user2@example.com",
        display_name="User Two",
        hashed_password="fake",
        api_key="key2",
        organization_id=org.id,
    )
    session.add_all([user1, user2])
    session.commit()

    yield session
    session.close()


@pytest.fixture
def temp_kb_dir():
    """Temporary directory for KB storage."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


class TestDeltaCheck:
    """Tests for fetch_if_changed() HTTP delta-checking."""

    def test_304_not_modified(self):
        """HEAD returns 304 Not Modified → no change, zero cost."""
        source = RegulationSource(
            module="eu_sfdr_csrd",
            name="SFDR Test",
            url="https://example.com/sfdr.txt",
            etag='"abc123"',
        )

        with patch("regulatory_kb.httpx.head") as mock_head:
            mock_head.return_value = Mock(status_code=304)

            changed, text = fetch_if_changed(source)

            assert changed is False
            assert text is None
            mock_head.assert_called_once()

    def test_200_content_unchanged_by_hash(self):
        """200 OK but SHA-256 matches → no change."""
        content = b"Regulation text here"
        content_hash = hashlib.sha256(content).hexdigest()

        source = RegulationSource(
            module="eu_sfdr_csrd",
            name="SFDR Test",
            url="https://example.com/sfdr.txt",
            content_sha256=content_hash,
        )

        with patch("regulatory_kb.httpx.head") as mock_head, \
             patch("regulatory_kb.httpx.get") as mock_get:
            mock_head.return_value = Mock(status_code=200)
            mock_get.return_value = Mock(text=content.decode(), status_code=200)

            changed, text = fetch_if_changed(source)

            assert changed is False
            assert text is None

    def test_200_content_changed(self):
        """200 OK with different SHA-256 → changed, return text."""
        old_content = b"Old regulation text"
        new_content = b"New regulation text"

        source = RegulationSource(
            module="eu_sfdr_csrd",
            name="SFDR Test",
            url="https://example.com/sfdr.txt",
            content_sha256=hashlib.sha256(old_content).hexdigest(),
        )

        with patch("regulatory_kb.httpx.head") as mock_head, \
             patch("regulatory_kb.httpx.get") as mock_get:
            mock_head.return_value = Mock(status_code=200)
            mock_get.return_value = Mock(text=new_content.decode(), status_code=200)

            changed, text = fetch_if_changed(source)

            assert changed is True
            assert text == new_content.decode()

    def test_head_request_includes_cache_headers(self):
        """fetch_if_changed sends If-None-Match and If-Modified-Since."""
        source = RegulationSource(
            module="eu_sfdr_csrd",
            name="SFDR Test",
            url="https://example.com/sfdr.txt",
            etag='"abc123"',
            last_modified="Wed, 21 Apr 2026 10:00:00 GMT",
        )

        with patch("regulatory_kb.httpx.head") as mock_head:
            mock_head.return_value = Mock(status_code=304)

            fetch_if_changed(source)

            call_args = mock_head.call_args
            headers = call_args[1]["headers"]
            assert headers["If-None-Match"] == '"abc123"'
            assert headers["If-Modified-Since"] == "Wed, 21 Apr 2026 10:00:00 GMT"


class TestIngestion:
    """Tests for ingest_source() chunking and embedding."""

    def test_ingest_writes_chunks_to_disk(self, db_session, temp_kb_dir):
        """ingest_source writes chunks.jsonl to disk."""
        source = RegulationSource(
            module="eu_sfdr_csrd",
            name="SFDR Level 1",
            url="https://example.com/sfdr",
            chunk_count=0,
        )
        db_session.add(source)
        db_session.commit()

        raw_text = "Regulation text. " * 100  # Long enough to create multiple chunks

        with patch("regulatory_kb.KB_BASE", temp_kb_dir), \
             patch("regulatory_kb.chunk_text") as mock_chunk, \
             patch("regulatory_kb.DenseRetriever") as mock_retriever_class:
            mock_chunk.return_value = ["Chunk 1", "Chunk 2", "Chunk 3"]
            mock_retriever = MagicMock()
            mock_retriever.embeddings = np.zeros((3, 384))
            mock_retriever_class.return_value = mock_retriever

            chunk_count = ingest_source(source, raw_text, db_session)

            assert chunk_count == 3

            chunks_file = temp_kb_dir / "eu_sfdr_csrd" / "chunks.jsonl"
            assert chunks_file.exists()

            with open(chunks_file) as f:
                chunks = [json.loads(line)["text"] for line in f]
            assert chunks == ["Chunk 1", "Chunk 2", "Chunk 3"]

    def test_ingest_updates_source_metadata(self, db_session, temp_kb_dir):
        """ingest_source updates chunk_count, last_fetched, last_changed in DB."""
        source = RegulationSource(
            module="eu_sfdr_csrd",
            name="SFDR Level 1",
            url="https://example.com/sfdr",
            chunk_count=0,
            last_fetched=None,
            last_changed=None,
        )
        db_session.add(source)
        db_session.commit()

        raw_text = "Regulation text. " * 100

        with patch("regulatory_kb.KB_BASE", temp_kb_dir), \
             patch("regulatory_kb.chunk_text") as mock_chunk, \
             patch("regulatory_kb.DenseRetriever") as mock_retriever_class:
            mock_chunk.return_value = ["Chunk 1", "Chunk 2"]
            mock_retriever = MagicMock()
            mock_retriever.embeddings = np.zeros((2, 384))
            mock_retriever_class.return_value = mock_retriever

            ingest_source(source, raw_text, db_session)

            # Reload from DB
            updated = db_session.query(RegulationSource).filter_by(id=source.id).first()
            assert updated.chunk_count == 2
            assert updated.last_fetched is not None
            assert updated.last_changed is not None
            assert updated.content_sha256 is not None


class TestNotificationFanOut:
    """Tests for _notify_all_users() notification creation."""

    def test_notify_all_users_creates_one_per_user(self, db_session):
        """_notify_all_users creates one notification for each user."""
        source = RegulationSource(
            module="eu_sfdr_csrd",
            name="SFDR Level 1",
            url="https://example.com/sfdr",
            version_label="ELI:32021R2088",
        )
        db_session.add(source)
        db_session.commit()

        _notify_all_users(db_session, "eu_sfdr_csrd", source, "ELI:32021R2087")

        notifications = db_session.query(Notification).all()
        assert len(notifications) == 2  # One for each test user

    def test_notification_body_includes_source_name_and_date(self, db_session):
        """Notification body includes source name and update date."""
        source = RegulationSource(
            module="eu_sfdr_csrd",
            name="SFDR Level 1 (EU 2019/2088)",
            url="https://example.com/sfdr",
            version_label="v2024-01",
        )
        db_session.add(source)
        db_session.commit()

        _notify_all_users(db_session, "eu_sfdr_csrd", source, "v2024-01")

        notification = db_session.query(Notification).first()
        assert "SFDR Level 1" in notification.body
        assert "re-review" in notification.body.lower()

    def test_notification_not_created_when_no_change(self, db_session):
        """No notification if _notify_all_users is not called."""
        initial_count = db_session.query(Notification).count()
        assert initial_count == 0


class TestRetrieverCache:
    """Tests for get_retriever() in-memory caching."""

    def test_retriever_cache_returns_none_if_kb_not_found(self):
        """get_retriever returns None if chunks/embeddings files not on disk."""
        # Clear cache
        _retrievers["eu_sfdr_csrd"] = None

        retriever = get_retriever("eu_sfdr_csrd")
        assert retriever is None

    def test_retriever_cache_is_invalidated_on_ingest(self, db_session, temp_kb_dir):
        """After ingest_source, _retrievers[module] is set to None (cache invalid)."""
        import regulatory_kb as kb_module

        source = RegulationSource(
            module="eu_sfdr_csrd",
            name="SFDR Level 1",
            url="https://example.com/sfdr",
        )
        db_session.add(source)
        db_session.commit()

        # Pre-populate cache with a fake retriever
        kb_module._retrievers["eu_sfdr_csrd"] = MagicMock()

        raw_text = "Regulation text. " * 100

        with patch("regulatory_kb.KB_BASE", temp_kb_dir), \
             patch("regulatory_kb.chunk_text") as mock_chunk, \
             patch("regulatory_kb.DenseRetriever") as mock_retriever_class:
            mock_chunk.return_value = ["Chunk 1", "Chunk 2"]
            mock_retriever = MagicMock()
            mock_retriever.embeddings = np.zeros((2, 384))
            mock_retriever_class.return_value = mock_retriever

            ingest_source(source, raw_text, db_session)

            # Cache should be invalidated (None)
            assert kb_module._retrievers["eu_sfdr_csrd"] is None


class TestLoadSources:
    """Tests for load_sources() DB query."""

    def test_load_sources_returns_all_from_db(self, db_session):
        """load_sources returns all RegulationSource rows from DB."""
        source1 = RegulationSource(
            module="eu_sfdr_csrd",
            name="SFDR Level 1",
            url="https://example.com/sfdr",
        )
        source2 = RegulationSource(
            module="ca_sb54",
            name="SB 54 Rules",
            url="https://example.com/sb54",
        )
        db_session.add_all([source1, source2])
        db_session.commit()

        sources = load_sources(db_session)

        assert len(sources) == 2
        assert sources[0].name in ["SFDR Level 1", "SB 54 Rules"]
        assert sources[1].name in ["SFDR Level 1", "SB 54 Rules"]
