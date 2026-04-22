"""
APScheduler-based background jobs for compliance-agent.

This module manages recurring regulatory update checks and other background tasks
that run independently of HTTP requests.
"""

from apscheduler.schedulers.background import BackgroundScheduler
from apscheduler.triggers.cron import CronTrigger
import logging

logger = logging.getLogger(__name__)

_scheduler: BackgroundScheduler | None = None


def start_scheduler() -> BackgroundScheduler:
    """
    Start the background scheduler with daily regulatory update checks.

    Scheduled jobs:
    - 07:00 UTC daily: Fetch latest regulations, update KB, notify users

    Returns:
        The running scheduler instance
    """
    global _scheduler
    if _scheduler and _scheduler.running:
        logger.info("Scheduler already running")
        return _scheduler

    _scheduler = BackgroundScheduler(timezone="UTC")

    # Daily regulatory update check at 07:00 UTC
    _scheduler.add_job(
        run_regulatory_update,
        CronTrigger(hour=7, minute=0),
        id="regulatory_update",
        name="Regulatory KB update check",
        replace_existing=True,
        misfire_grace_time=3600,  # Allow up to 1 hour grace for missed runs
    )

    _scheduler.start()
    logger.info("Scheduler started with daily regulatory update job at 07:00 UTC")
    return _scheduler


def stop_scheduler() -> None:
    """Gracefully shutdown the scheduler."""
    global _scheduler
    if _scheduler and _scheduler.running:
        _scheduler.shutdown(wait=True)
        _scheduler = None
        logger.info("Scheduler stopped")


def run_regulatory_update() -> None:
    """
    Fetch latest regulatory sources, update KB, notify users of changes.

    This is called daily at 07:00 UTC (or configured REGULATORY_CHECK_HOUR) by the scheduler.

    Steps:
    1. For each RegulationSource (SEC, SFDR/CSRD, SB54):
       - Fetch if changed (HTTP delta-check: ETag, Last-Modified, SHA-256)
       - If changed: re-ingest chunks, re-compute embeddings
       - Notify all users of the change
    2. Log summary to logger
    """
    from db import get_db
    from regulatory_kb import load_sources, fetch_if_changed, ingest_source, _notify_all_users

    db = next(get_db())
    try:
        sources = load_sources(db)
        if not sources:
            logger.info("[scheduler] No regulation sources to check")
            return

        updated_count = 0
        for source in sources:
            try:
                changed, raw_text = fetch_if_changed(source)
                if changed and raw_text:
                    old_version = source.version_label
                    ingest_source(source, raw_text, db)
                    _notify_all_users(db, source.module, source, old_version)
                    updated_count += 1
                    logger.info(f"[scheduler] Updated: {source.name}")
                else:
                    # Update last_fetched even if no change
                    from models import utc_now
                    source.last_fetched = utc_now()
                    db.commit()
                    logger.debug(f"[scheduler] No change: {source.name}")
            except Exception as e:
                logger.error(f"[scheduler] Error processing {source.name}: {e}")
                # Continue to next source — don't crash the whole job

        logger.info(f"[scheduler] Regulatory update complete: {updated_count} source(s) updated")

    except Exception as e:
        logger.error(f"[scheduler] Regulatory update job failed: {e}")
    finally:
        db.close()
