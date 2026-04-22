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

    This is called daily at 07:00 UTC by the scheduler.

    Steps:
    1. For each RegulationSource (SEC, SFDR/CSRD, SB54):
       - Fetch if changed (HTTP delta-check: ETag, Last-Modified, SHA-256)
       - If changed: re-ingest chunks, re-compute embeddings
    2. For each user: create Notification entries for updated sources
    3. Log summary to logger
    """
    # TODO: Implement regulatory KB update logic (Epic 1)
    logger.info("Regulatory update job triggered (stub)")
