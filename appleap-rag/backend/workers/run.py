"""Background worker. Drains the `jobs` table.

Run as a separate process from the FastAPI app:

    python -m backend.workers.run

Concurrency is one job at a time per worker process. To scale, start more
processes — Postgres FOR UPDATE SKIP LOCKED keeps them from claiming the
same job. Embedding/parsing is GPU-bound on the L4 anyway, so two or three
processes is realistic ceiling for now.

Loop:
1. Try to claim one job (SKIP LOCKED). If none, sleep POLL_INTERVAL and retry.
2. Run the handler in its own session/transaction.
3. Resolve the job: success → DELETE; retry → release lock + reschedule;
   exhausted/fatal → move to dead_letter_events.
4. Periodically clear stuck locks from crashed workers.

Stop with SIGINT/SIGTERM — current job finishes, then loop exits.
"""

from __future__ import annotations

import asyncio
import logging
import os
import signal
import socket
import sys
from typing import Optional

from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession

from backend.db.connection import async_session
from backend.db.models import DeadLetterEvent
from backend.workers.handlers import FatalError, RetryableError, dispatch

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s %(message)s",
)
logger = logging.getLogger("backend.workers")

POLL_INTERVAL_SECONDS = 1.0
STUCK_JOB_AFTER_MINUTES = 5
STUCK_JOB_SWEEP_INTERVAL_SECONDS = 60
MAX_BACKOFF_SECONDS = 600

_shutdown = False


def _backoff_seconds(attempts: int) -> int:
    """Exponential backoff: 2, 4, 8, 16, ... capped at MAX_BACKOFF_SECONDS."""
    return min(2 ** attempts, MAX_BACKOFF_SECONDS)


async def _claim_job(
    session: AsyncSession,
    worker_id: str,
) -> Optional[dict]:
    """Claim one available job. Returns its row as a dict, or None if queue empty.

    Uses FOR UPDATE SKIP LOCKED so concurrent workers never race on the same
    row. The locked_at/locked_by stamp survives even if this worker crashes
    — the stuck-job sweeper will reclaim those after STUCK_JOB_AFTER_MINUTES.
    """
    row = await session.execute(text("""
        SELECT id, payload, attempts, max_attempts
        FROM jobs
        WHERE locked_at IS NULL AND run_after <= now()
        ORDER BY run_after
        FOR UPDATE SKIP LOCKED
        LIMIT 1
    """))
    job = row.mappings().first()
    if not job:
        return None

    await session.execute(
        text("UPDATE jobs SET locked_at = now(), locked_by = :wid WHERE id = :id"),
        {"wid": worker_id, "id": job["id"]},
    )
    return dict(job)


async def _resolve_job(job: dict, error: Optional[BaseException]) -> None:
    """Final disposition for a processed job. Runs in its own transaction so
    the resolution survives independently of the handler's writes.
    """
    async with async_session() as session:
        async with session.begin():
            if error is None:
                await session.execute(
                    text("DELETE FROM jobs WHERE id = :id"), {"id": job["id"]}
                )
                return

            new_attempts = job["attempts"] + 1
            err_text = str(error)[:1000]
            fatal = isinstance(error, FatalError)
            exhausted = new_attempts >= job["max_attempts"]

            if fatal or exhausted:
                session.add(DeadLetterEvent(
                    attempts=new_attempts,
                    last_error=err_text,
                    payload=job["payload"],
                ))
                await session.execute(
                    text("DELETE FROM jobs WHERE id = :id"), {"id": job["id"]}
                )
                logger.warning(
                    "job_dlq id=%s attempts=%d reason=%s err=%s",
                    job["id"], new_attempts,
                    "fatal" if fatal else "exhausted", err_text,
                )
            else:
                backoff = _backoff_seconds(new_attempts)
                await session.execute(text("""
                    UPDATE jobs
                    SET locked_at = NULL,
                        locked_by = NULL,
                        attempts = :attempts,
                        run_after = now() + (:backoff || ' seconds')::interval,
                        last_error = :err
                    WHERE id = :id
                """), {
                    "attempts": new_attempts,
                    "backoff": str(backoff),
                    "err": err_text,
                    "id": job["id"],
                })
                logger.info(
                    "job_retry id=%s attempts=%d backoff=%ds err=%s",
                    job["id"], new_attempts, backoff, err_text,
                )


async def _process_one(worker_id: str) -> bool:
    """Try to claim and process one job. Returns True if a job was processed,
    False if the queue was empty (caller sleeps).
    """
    # 1. Claim (own transaction so the lock holds even if the handler
    #    crashes the surrounding request).
    async with async_session() as session:
        async with session.begin():
            job = await _claim_job(session, worker_id)
    if job is None:
        return False

    logger.info("job_start id=%s action=%s", job["id"], job["payload"].get("action"))

    # 2. Run handler in its own session.
    handler_error: Optional[BaseException] = None
    try:
        async with async_session() as session:
            async with session.begin():
                await dispatch(session, job["payload"])
    except (RetryableError, FatalError) as e:
        handler_error = e
    except Exception as e:
        # Treat unknown exceptions as retryable — better to retry a transient
        # bug than DLQ a healable one. If it keeps failing, max_attempts
        # routes it to the DLQ anyway.
        handler_error = RetryableError(repr(e))

    # 3. Resolve.
    await _resolve_job(job, handler_error)
    if handler_error is None:
        logger.info("job_done id=%s", job["id"])

    return True


async def _sweep_stuck_jobs() -> int:
    """Release locks held by dead workers. Idempotent — safe to run on a timer.

    Returns the number of jobs reclaimed.
    """
    async with async_session() as session:
        async with session.begin():
            result = await session.execute(text(f"""
                UPDATE jobs
                SET locked_at = NULL, locked_by = NULL
                WHERE locked_at < now() - interval '{STUCK_JOB_AFTER_MINUTES} minutes'
            """))
    n = result.rowcount or 0
    if n:
        logger.warning("stuck_job_sweep reclaimed=%d", n)
    return n


async def _stuck_sweeper_loop() -> None:
    """Background coroutine: runs the sweeper every N seconds until shutdown."""
    while not _shutdown:
        try:
            await _sweep_stuck_jobs()
        except Exception as e:
            logger.error("stuck_sweep failed: %s", e)
        await asyncio.sleep(STUCK_JOB_SWEEP_INTERVAL_SECONDS)


async def _drain_loop(worker_id: str) -> None:
    """Main work loop. Claims jobs as fast as the queue can produce them;
    sleeps POLL_INTERVAL when empty.
    """
    while not _shutdown:
        try:
            had_work = await _process_one(worker_id)
        except Exception as e:
            logger.exception("drain_loop unexpected error: %s", e)
            had_work = False

        if not had_work:
            await asyncio.sleep(POLL_INTERVAL_SECONDS)


def _install_signal_handlers() -> None:
    def _stop(signum, _frame):
        global _shutdown
        logger.info("worker received signal=%d, shutting down", signum)
        _shutdown = True

    signal.signal(signal.SIGINT, _stop)
    if hasattr(signal, "SIGTERM"):
        signal.signal(signal.SIGTERM, _stop)


async def main() -> None:
    worker_id = f"{socket.gethostname()}:{os.getpid()}"
    logger.info("worker starting id=%s", worker_id)

    _install_signal_handlers()

    drain_task = asyncio.create_task(_drain_loop(worker_id))
    sweeper_task = asyncio.create_task(_stuck_sweeper_loop())

    await drain_task
    sweeper_task.cancel()
    try:
        await sweeper_task
    except asyncio.CancelledError:
        pass

    logger.info("worker stopped id=%s", worker_id)


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        sys.exit(0)
