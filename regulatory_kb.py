"""
Regulatory Knowledge Base management for compliance-agent.

Handles:
- HTTP delta-checking (ETag, Last-Modified, SHA-256) for regulation sources
- Chunking and embedding regulatory text
- Disk persistence (chunks.jsonl + embeddings.npy)
- Per-user notification fan-out on updates
"""

import hashlib
import json
import logging
import os
from pathlib import Path
from typing import Optional

import httpx
import numpy as np
from sqlalchemy.orm import Session

from db import get_db
from models import RegulationSource, Notification, User, utc_now
from retriever import DenseRetriever
from sec import chunk_text

logger = logging.getLogger(__name__)

# Base directory for regulatory KB storage
KB_BASE = Path(os.environ.get("DATA_ROOT", ".")) / "regulatory_kb"
KB_BASE.mkdir(parents=True, exist_ok=True)

# In-memory retriever cache: {module: DenseRetriever}
_retrievers: dict[str, Optional[DenseRetriever]] = {
    "sec": None,
    "eu_sfdr_csrd": None,
    "ca_sb54": None,
}


def load_sources(db: Session) -> list[RegulationSource]:
    """Load all regulation sources from the database."""
    return db.query(RegulationSource).all()


def fetch_if_changed(source: RegulationSource) -> tuple[bool, str | None]:
    """
    Check if a regulatory source has changed using HTTP delta-checking.

    Delta-check strategy:
    1. Send HEAD with If-None-Match (ETag) + If-Modified-Since headers
    2. 304 Not Modified → unchanged, zero cost
    3. 200 → download full content
    4. Compare SHA-256 hash; if identical despite headers, no change
    5. Return (True, text) if changed; (False, None) if unchanged

    Args:
        source: RegulationSource with url, etag, last_modified, content_sha256

    Returns:
        (changed: bool, text: str | None) — (True, raw_text) or (False, None)
    """
    try:
        headers = {}
        if source.etag:
            headers["If-None-Match"] = source.etag
        if source.last_modified:
            headers["If-Modified-Since"] = source.last_modified

        # HEAD request to check for changes
        head_resp = httpx.head(
            source.url, headers=headers, timeout=15, follow_redirects=True
        )

        if head_resp.status_code == 304:
            logger.debug(f"[kb:delta] 304 Not Modified: {source.name}")
            return False, None

        # 200 or other: download full content
        get_resp = httpx.get(
            source.url, timeout=60, follow_redirects=True
        )
        get_resp.raise_for_status()
        raw_text = get_resp.text

        # Compare SHA-256 hash
        new_hash = hashlib.sha256(raw_text.encode()).hexdigest()
        if new_hash == source.content_sha256:
            logger.debug(f"[kb:delta] SHA-256 match (no real change): {source.name}")
            return False, None

        logger.info(f"[kb:delta] Changed: {source.name} (new hash: {new_hash[:8]}...)")
        return True, raw_text

    except Exception as e:
        logger.error(f"[kb:delta] Error fetching {source.name}: {e}")
        return False, None


def ingest_source(
    source: RegulationSource, raw_text: str, db: Session
) -> int:
    """
    Ingest a regulation source: chunk, embed, write to disk, update DB.

    Args:
        source: RegulationSource to ingest
        raw_text: Full regulation text from fetch_if_changed()
        db: Database session

    Returns:
        Number of chunks created
    """
    try:
        # Step 1: Chunk the text (reuse sec.chunk_text)
        chunks = chunk_text(raw_text, chunk_size=1200, overlap=200)
        if not chunks:
            logger.warning(f"[kb:ingest] No chunks for {source.name}")
            return 0

        # Step 2: Build embeddings
        retriever = DenseRetriever()
        for chunk in chunks:
            retriever.add(chunk)
        retriever.build()

        # Step 3: Write to disk
        module_dir = KB_BASE / source.module
        module_dir.mkdir(parents=True, exist_ok=True)

        chunks_file = module_dir / "chunks.jsonl"
        embeddings_file = module_dir / "embeddings.npy"

        with open(chunks_file, "w") as f:
            for chunk in chunks:
                f.write(json.dumps({"text": chunk}) + "\n")

        np.save(embeddings_file, retriever.embeddings)

        # Step 4: Update source metadata in DB
        new_hash = hashlib.sha256(raw_text.encode()).hexdigest()
        source.content_sha256 = new_hash
        source.chunk_count = len(chunks)
        source.last_fetched = utc_now()
        source.last_changed = utc_now()
        db.add(source)
        db.commit()

        logger.info(
            f"[kb:ingest] Ingested {source.name}: {len(chunks)} chunks, "
            f"{retriever.embeddings.shape[0]} embeddings"
        )

        # Step 5: Invalidate in-memory cache so next get_retriever() rebuilds it
        _retrievers[source.module] = None

        return len(chunks)

    except Exception as e:
        logger.error(f"[kb:ingest] Error ingesting {source.name}: {e}")
        return 0


def build_retriever(module: str) -> Optional[DenseRetriever]:
    """
    Load a module's KB from disk and build a DenseRetriever.

    Args:
        module: "sec" | "eu_sfdr_csrd" | "ca_sb54"

    Returns:
        DenseRetriever with chunks and embeddings, or None if KB not found
    """
    try:
        module_dir = KB_BASE / module
        chunks_file = module_dir / "chunks.jsonl"
        embeddings_file = module_dir / "embeddings.npy"

        if not chunks_file.exists() or not embeddings_file.exists():
            logger.debug(f"[kb:build] KB not found for {module}")
            return None

        retriever = DenseRetriever()
        chunks = []
        with open(chunks_file) as f:
            for line in f:
                chunk = json.loads(line)["text"]
                chunks.append(chunk)
                retriever.add(chunk)

        embeddings = np.load(embeddings_file)
        retriever.embeddings = embeddings

        logger.info(f"[kb:build] Loaded {module}: {len(chunks)} chunks")
        return retriever

    except Exception as e:
        logger.error(f"[kb:build] Error building retriever for {module}: {e}")
        return None


def get_retriever(module: str) -> Optional[DenseRetriever]:
    """
    Get a retriever for a module, using in-memory cache.

    Cache is invalidated when ingest_source() completes, so next call rebuilds.

    Args:
        module: "sec" | "eu_sfdr_csrd" | "ca_sb54"

    Returns:
        DenseRetriever or None
    """
    if module not in _retrievers:
        logger.warning(f"[kb:cache] Unknown module: {module}")
        return None

    if _retrievers[module] is None:
        _retrievers[module] = build_retriever(module)

    return _retrievers[module]


def _notify_all_users(
    db: Session, module: str, source: RegulationSource, old_version: Optional[str]
) -> None:
    """
    Create a notification for every user after a regulation source update.

    Args:
        db: Database session
        module: Module name (for Notification.module field)
        source: The updated RegulationSource
        old_version: Previous version_label (may be None)
    """
    try:
        users = db.query(User).all()
        created_count = 0

        for user in users:
            notification = Notification(
                user_id=user.id,
                module=module,
                source_name=source.name,
                title=f"Regulatory update: {source.name}",
                body=(
                    f"{source.name} was updated on {utc_now().date().isoformat()}. "
                    f"Previous version: {old_version or 'unknown'}. "
                    f"Reports run before this date may need re-review under the new regulations."
                ),
            )
            db.add(notification)
            created_count += 1

        db.commit()
        logger.info(f"[kb:notify] Created {created_count} notifications for {source.name}")

    except Exception as e:
        logger.error(f"[kb:notify] Error notifying users for {source.name}: {e}")
