"""Nango webhook receiver.

Receives outbound webhooks from a self-hosted Nango instance and enqueues
work in the `jobs` table for the worker process to drain.

Contract on receipt:
1. Verify HMAC-SHA256 signature against `settings.nango_signing_secret`.
2. Parse JSON body. One Nango webhook delivery may carry N record events.
3. Within ONE transaction:
   - INSERT a `processed_events` row keyed by `event_id` (header or body hash).
     ON CONFLICT DO NOTHING. If 0 rows affected, the delivery is a duplicate
     — return 200 immediately and do nothing else.
   - Otherwise, INSERT one `jobs` row per record event in the batch.
4. COMMIT, return 200.

Failure modes handled:
- Duplicate delivery → idempotent (unique constraint on event_id).
- Crash between the two inserts → impossible (same transaction).
- Unverified payload → 401, never enters the queue.
- Malformed body → 400, never enters the queue.

The receiver is deliberately small — it does no fetching, no parsing, no
embedding. All real work happens in the worker process.
"""

from __future__ import annotations

import hashlib
import hmac
import json
import logging
from typing import Any

from fastapi import APIRouter, Header, HTTPException, Request
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession
from fastapi import Depends

from backend.config import settings
from backend.db.connection import get_session
from backend.db.models import Job, ProcessedEvent

logger = logging.getLogger(__name__)

router = APIRouter()


def _verify_signature(body: bytes, signature: str | None) -> bool:
    """Verify Nango's HMAC-SHA256 signature against the raw body.

    Returns True when the signing secret is empty (dev mode) — production
    deployments MUST set APPLEAP_NANGO_SIGNING_SECRET.
    """
    secret = settings.nango_signing_secret
    if not secret:
        return True
    if not signature:
        return False
    expected = hmac.new(
        secret.encode("utf-8"), body, hashlib.sha256
    ).hexdigest()
    return hmac.compare_digest(expected, signature)


def _derive_event_id(body: bytes, header_id: str | None) -> str:
    """Stable per-delivery ID for idempotency.

    Prefers a provider-supplied delivery header when present, falls back to
    a content hash of the raw body. The fallback is safe because identical
    bodies represent identical work — re-processing them is by definition a
    duplicate.
    """
    if header_id:
        return header_id
    return "sha256:" + hashlib.sha256(body).hexdigest()


def _extract_jobs(payload: dict[str, Any]) -> list[dict[str, Any]]:
    """Translate one Nango webhook payload into N job payloads.

    Nango's per-sync model defines what fields each record carries. We treat
    the payload's `records` (or `data`) array as the unit of work and pass
    each record through to the worker as-is, plus the routing fields the
    worker needs to dispatch:
      - provider          (Nango's providerConfigKey, e.g. 'github')
      - connection_id     (Nango's connection identifier — tenant)
      - action            ('upsert' | 'delete' | 'acl_update' | 'group_membership')
      - source_id         (provider-stable doc id; empty for membership events)
      - record            (the original record, for the worker to consume)

    The `action` is taken from the record itself when present (sync scripts
    set it explicitly), or inferred from `_metadata.deletedAt` if Nango
    flagged a soft-delete. Everything else defaults to 'upsert'.
    """
    provider = payload.get("providerConfigKey") or payload.get("provider") or ""
    connection_id = payload.get("connectionId") or payload.get("connection_id") or ""
    model = payload.get("model") or payload.get("syncName") or ""

    records = (
        payload.get("records")
        or payload.get("data")
        or payload.get("response")
        or []
    )
    if isinstance(records, dict):
        records = [records]

    jobs: list[dict[str, Any]] = []
    for r in records:
        if not isinstance(r, dict):
            continue

        action = r.get("action")
        if not action:
            soft_deleted = (
                r.get("_metadata", {}).get("deletedAt")
                if isinstance(r.get("_metadata"), dict)
                else None
            )
            if soft_deleted:
                action = "delete"
            elif model in ("user_groups", "group_membership"):
                action = "group_membership"
            elif model in ("acl", "permissions"):
                action = "acl_update"
            else:
                action = "upsert"

        source_id = r.get("source_id") or r.get("id") or ""
        if not source_id:
            md = r.get("_metadata")
            if isinstance(md, dict):
                source_id = md.get("recordId", "")

        jobs.append({
            "provider": provider,
            "connection_id": connection_id,
            "model": model,
            "action": action,
            "source_id": source_id,
            "record": r,
        })

    return jobs


@router.post("/webhooks/nango")
async def nango_webhook(
    request: Request,
    x_nango_signature: str | None = Header(default=None, alias="X-Nango-Signature"),
    x_nango_delivery_id: str | None = Header(default=None, alias="X-Nango-Delivery-Id"),
    session: AsyncSession = Depends(get_session),
):
    """Nango webhook entrypoint. See module docstring for the contract."""
    body = await request.body()

    if not _verify_signature(body, x_nango_signature):
        logger.warning("nango_webhook signature verification failed")
        raise HTTPException(status_code=401, detail="Invalid signature")

    try:
        payload = json.loads(body)
    except json.JSONDecodeError:
        raise HTTPException(status_code=400, detail="Invalid JSON")

    if not isinstance(payload, dict):
        raise HTTPException(status_code=400, detail="Expected JSON object")

    event_id = _derive_event_id(body, x_nango_delivery_id)
    provider = payload.get("providerConfigKey") or payload.get("provider") or "unknown"
    connection_id = payload.get("connectionId") or payload.get("connection_id") or ""
    event_type = payload.get("type") or payload.get("syncName") or "unknown"

    # Idempotency: ON CONFLICT DO NOTHING. If the row already exists, this
    # delivery has been seen — short-circuit before enqueueing duplicate work.
    insert_event = await session.execute(
        text("""
            INSERT INTO processed_events (event_id, provider, connection_id, event_type)
            VALUES (:event_id, :provider, :connection_id, :event_type)
            ON CONFLICT (event_id) DO NOTHING
        """),
        {
            "event_id": event_id,
            "provider": provider,
            "connection_id": connection_id,
            "event_type": event_type,
        },
    )
    if (insert_event.rowcount or 0) == 0:
        await session.rollback()
        logger.info("nango_webhook duplicate event_id=%s (skipping)", event_id)
        return {"status": "duplicate", "event_id": event_id}

    job_payloads = _extract_jobs(payload)
    for jp in job_payloads:
        session.add(Job(payload=jp))

    await session.commit()

    logger.info(
        "nango_webhook accepted event_id=%s provider=%s jobs=%d",
        event_id, provider, len(job_payloads),
    )
    return {"status": "accepted", "event_id": event_id, "jobs": len(job_payloads)}
