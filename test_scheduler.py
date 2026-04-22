"""
Tests for background scheduler (scheduler.py).

Tests cover:
- Scheduler startup/shutdown
- Daily regulatory update job triggering
- Error handling and resilience
"""

from unittest.mock import MagicMock, patch, Mock
import logging

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from models import Base, RegulationSource, Notification, User, Organization
from scheduler import start_scheduler, stop_scheduler, run_regulatory_update


logger = logging.getLogger(__name__)


@pytest.fixture
def db_session():
    """In-memory SQLite session for testing."""
    engine = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(engine)
    Session = sessionmaker(bind=engine)
    session = Session()

    # Create organization and users
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


class TestSchedulerStartup:
    """Tests for start_scheduler() and stop_scheduler()."""

    def test_start_scheduler_returns_scheduler(self):
        """start_scheduler returns a running BackgroundScheduler."""
        scheduler = start_scheduler()
        assert scheduler is not None
        assert scheduler.running is True
        scheduler.shutdown()

    def test_scheduler_includes_regulatory_update_job(self):
        """Scheduler has regulatory_update job with daily 07:00 UTC trigger."""
        scheduler = start_scheduler()
        try:
            jobs = scheduler.get_jobs()
            assert len(jobs) >= 1
            job_ids = [job.id for job in jobs]
            assert "regulatory_update" in job_ids
        finally:
            scheduler.shutdown()

    def test_stop_scheduler_shuts_down(self):
        """stop_scheduler() stops the running scheduler."""
        # Import the scheduler module to access the global _scheduler
        import scheduler as scheduler_module

        # Start fresh
        scheduler_module._scheduler = None
        scheduler = start_scheduler()
        assert scheduler.running is True

        stop_scheduler()

        # After stop, _scheduler should be None
        assert scheduler_module._scheduler is None

    def test_start_scheduler_idempotent(self):
        """Calling start_scheduler() twice doesn't create duplicate jobs."""
        scheduler1 = start_scheduler()
        job_count_1 = len(scheduler1.get_jobs())

        scheduler2 = start_scheduler()
        job_count_2 = len(scheduler2.get_jobs())

        # Should be same scheduler with same job count (replace_existing=True)
        assert scheduler1 is scheduler2
        assert job_count_2 == job_count_1

        scheduler2.shutdown()


class TestRegulatoryUpdateJob:
    """Tests for run_regulatory_update() job logic."""

    def test_run_regulatory_update_no_sources(self, db_session):
        """Job runs gracefully when no sources are configured."""
        with patch("scheduler.get_db") as mock_get_db, \
             patch("scheduler.load_sources") as mock_load:
            mock_get_db.return_value = iter([db_session])
            mock_load.return_value = []

            # Should not raise
            run_regulatory_update()

    def test_run_regulatory_update_checks_all_sources(self, db_session):
        """Job fetches all sources and checks each for changes."""
        source1 = RegulationSource(
            module="eu_sfdr_csrd",
            name="SFDR Level 1",
            url="https://example.com/sfdr",
        )
        source2 = RegulationSource(
            module="ca_sb54",
            name="SB 54",
            url="https://example.com/sb54",
        )
        db_session.add_all([source1, source2])
        db_session.commit()

        with patch("scheduler.get_db") as mock_get_db, \
             patch("scheduler.load_sources") as mock_load, \
             patch("scheduler.fetch_if_changed") as mock_fetch:
            mock_get_db.return_value = iter([db_session])
            mock_load.return_value = [source1, source2]
            mock_fetch.return_value = (False, None)  # No changes

            run_regulatory_update()

            # fetch_if_changed should be called for each source
            assert mock_fetch.call_count == 2

    def test_run_regulatory_update_ingests_on_change(self, db_session):
        """Job calls ingest_source and _notify_all_users when content changed."""
        source = RegulationSource(
            module="eu_sfdr_csrc",
            name="SFDR Level 1",
            url="https://example.com/sfdr",
            version_label="v1",
        )
        db_session.add(source)
        db_session.commit()

        raw_text = "New regulation text"

        with patch("scheduler.get_db") as mock_get_db, \
             patch("scheduler.load_sources") as mock_load, \
             patch("scheduler.fetch_if_changed") as mock_fetch, \
             patch("scheduler.ingest_source") as mock_ingest, \
             patch("scheduler._notify_all_users") as mock_notify:
            mock_get_db.return_value = iter([db_session])
            mock_load.return_value = [source]
            mock_fetch.return_value = (True, raw_text)  # Changed!
            mock_ingest.return_value = 5  # 5 chunks ingested

            run_regulatory_update()

            # ingest_source and _notify_all_users should be called
            mock_ingest.assert_called_once()
            mock_notify.assert_called_once()

    def test_run_regulatory_update_skips_unchanged_sources(self, db_session):
        """Job skips ingest for sources with no changes."""
        source = RegulationSource(
            module="eu_sfdr_csrd",
            name="SFDR Level 1",
            url="https://example.com/sfdr",
        )
        db_session.add(source)
        db_session.commit()

        with patch("scheduler.get_db") as mock_get_db, \
             patch("scheduler.load_sources") as mock_load, \
             patch("scheduler.fetch_if_changed") as mock_fetch, \
             patch("scheduler.ingest_source") as mock_ingest:
            mock_get_db.return_value = iter([db_session])
            mock_load.return_value = [source]
            mock_fetch.return_value = (False, None)  # No change

            run_regulatory_update()

            # ingest_source should NOT be called
            mock_ingest.assert_not_called()

    def test_run_regulatory_update_continues_on_source_error(self, db_session):
        """Job continues to next source if one raises an error."""
        source1 = RegulationSource(
            module="eu_sfdr_csrd",
            name="SFDR Level 1",
            url="https://example.com/sfdr",
        )
        source2 = RegulationSource(
            module="ca_sb54",
            name="SB 54",
            url="https://example.com/sb54",
        )
        db_session.add_all([source1, source2])
        db_session.commit()

        with patch("scheduler.get_db") as mock_get_db, \
             patch("scheduler.load_sources") as mock_load, \
             patch("scheduler.fetch_if_changed") as mock_fetch:
            mock_get_db.return_value = iter([db_session])
            mock_load.return_value = [source1, source2]

            # First source raises error, second returns False
            mock_fetch.side_effect = [Exception("Network error"), (False, None)]

            # Should not raise despite first error
            run_regulatory_update()

            # Both sources should have been attempted
            assert mock_fetch.call_count == 2

    def test_run_regulatory_update_updates_last_fetched(self, db_session):
        """Job updates last_fetched timestamp even when no change."""
        source = RegulationSource(
            module="eu_sfdr_csrd",
            name="SFDR Level 1",
            url="https://example.com/sfdr",
            last_fetched=None,
        )
        db_session.add(source)
        db_session.commit()
        source_id = source.id

        with patch("scheduler.get_db") as mock_get_db, \
             patch("scheduler.load_sources") as mock_load, \
             patch("scheduler.fetch_if_changed") as mock_fetch:
            mock_get_db.return_value = iter([db_session])
            mock_load.return_value = [source]
            mock_fetch.return_value = (False, None)

            run_regulatory_update()

            # Reload from DB
            updated = db_session.query(RegulationSource).filter_by(id=source_id).first()
            assert updated.last_fetched is not None

    def test_run_regulatory_update_logs_summary(self, db_session, caplog):
        """Job logs update summary."""
        source = RegulationSource(
            module="eu_sfdr_csrd",
            name="SFDR Level 1",
            url="https://example.com/sfdr",
        )
        db_session.add(source)
        db_session.commit()

        with patch("scheduler.get_db") as mock_get_db, \
             patch("scheduler.load_sources") as mock_load, \
             patch("scheduler.fetch_if_changed") as mock_fetch:
            caplog.set_level(logging.INFO)
            mock_get_db.return_value = iter([db_session])
            mock_load.return_value = [source]
            mock_fetch.return_value = (False, None)

            run_regulatory_update()

            assert "complete" in caplog.text.lower()
