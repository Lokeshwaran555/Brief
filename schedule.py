"""APScheduler setup.

Pipeline runs twice daily — 05:00 GST and 13:00 GST. Each run is a
chained sequence:
  ingest_flow → daily_brief_flow → ceo_scan_flow (daily cadence)
Processing only fires after ingest completes; if ingest fails the
downstream steps are skipped (signals won't be fresh enough to brief
or scan honestly).

Auto-investigation of high-priority signals is wired inside
ingest_flow itself, so it lives within the chain.

investigation_sweep stays hourly — lightweight catch-up for any
high-priority signals that slipped past auto-fire.

ceo_scan weekly digest fires once on Sunday 05:30 GST (after
Sunday's morning pipeline run completes).
"""
from __future__ import annotations

import logging

from apscheduler.schedulers.asyncio import AsyncIOScheduler
from apscheduler.triggers.cron import CronTrigger
from apscheduler.triggers.interval import IntervalTrigger

from flows import ceo_scan_flow, daily_brief_flow, ingest_flow, investigation_sweep
from settings import settings

log = logging.getLogger(__name__)

scheduler = AsyncIOScheduler(timezone="Asia/Dubai")


async def _run_pipeline() -> None:
    """Chained pipeline — ingest → daily_brief → ceo_scan (daily).

    Processing fires ONLY when ingest succeeds. The MD's rule: don't
    brief / scan before signals are in. A brief failure doesn't block
    the scan; both run independently after ingest.
    """
    log.info("pipeline: start")
    try:
        ingest_result = await ingest_flow.run()
        log.info("pipeline: ingest ok %s", ingest_result)
    except Exception:
        log.exception("pipeline: ingest failed — skipping downstream brief + scan")
        return

    try:
        brief_result = await daily_brief_flow.run()
        log.info("pipeline: daily_brief ok %s", brief_result)
    except Exception:
        log.exception("pipeline: daily_brief failed — continuing to ceo_scan")

    try:
        scan_result = await ceo_scan_flow.run(cadence="daily")
        log.info("pipeline: ceo_scan daily ok %s", scan_result)
    except Exception:
        log.exception("pipeline: ceo_scan daily failed")

    log.info("pipeline: done")


async def _run_investigation_sweep() -> None:
    try:
        result = await investigation_sweep.run()
        log.info("investigation_sweep ok %s", result)
    except Exception:
        log.exception("investigation_sweep failed")


async def _run_ceo_scan_weekly() -> None:
    try:
        result = await ceo_scan_flow.run(cadence="weekly")
        log.info("ceo_scan_flow weekly ok %s", result)
    except Exception:
        log.exception("ceo_scan_flow weekly failed")


def start() -> None:
    # Twice-daily pipeline run: 05:00 + 13:00 GST.
    # 2026-04-30 (audit-round): reverted from 3× to 2× daily to keep
    # the Tavily / Apify / LinkedIn-scout cost envelope inside the
    # documented budget (~131 calls/run × 2 = ~260/day). The MD's
    # REFRESH button still triggers /run/pipeline directly so an
    # on-demand third run is one click away whenever needed.
    scheduler.add_job(
        _run_pipeline,
        trigger=CronTrigger(hour="5,13", minute=0, timezone="Asia/Dubai"),
        id="pipeline",
        replace_existing=True,
    )
    scheduler.add_job(
        _run_investigation_sweep,
        # Hourly catch-up for any high-priority signals that slipped
        # past auto-fire during the pipeline ingest. Lightweight.
        trigger=IntervalTrigger(hours=1),
        id="investigation_sweep",
        replace_existing=True,
    )
    scheduler.add_job(
        _run_ceo_scan_weekly,
        # Sunday 05:30 GST — weekly digest, runs after Sunday's
        # morning pipeline (05:00) has finished its daily scan.
        trigger=CronTrigger(day_of_week="sun", hour=5, minute=30, timezone="Asia/Dubai"),
        id="ceo_scan_weekly",
        replace_existing=True,
    )
    scheduler.start()
    log.info(
        "scheduler started — pipeline 05:00 + 13:00 GST (chained) · "
        "ceo_scan weekly Sun 05:30 · investigation_sweep hourly"
    )


def shutdown() -> None:
    if scheduler.running:
        scheduler.shutdown(wait=False)
