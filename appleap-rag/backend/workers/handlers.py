"""Job handlers for the worker process.

Each job's `payload.action` routes to one handler. New connector actions
(upsert/delete) register their own handler in Phase 4; ACL and group
membership are connector-agnostic and live here permanently.

Handler contract:
- Runs inside an open AsyncSession (caller commits the surrounding transaction).
- Raises `RetryableError` for transient failures (network, rate limit) —
  worker reschedules with exponential backoff.
- Raises `FatalError` for permanent failures (bad payload, missing fields) —
  worker moves the job straight to dead_letter_events without retrying.
- Returns None on success — worker deletes the job from the queue.
"""

from __future__ import annotations

import logging
from typing import Awaitable, Callable

from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession

from backend.acl import update_document_acl

logger = logging.getLogger(__name__)


class RetryableError(Exception):
    """Transient — worker will retry with exponential backoff."""


class FatalError(Exception):
    """Permanent — worker will route the job straight to the DLQ."""


Handler = Callable[[AsyncSession, dict], Awaitable[None]]
_handlers: dict[str, Handler] = {}


def register_handler(action: str):
    def decorator(fn: Handler) -> Handler:
        if action in _handlers:
            logger.warning("Overriding existing handler for action=%s", action)
        _handlers[action] = fn
        return fn
    return decorator


async def dispatch(session: AsyncSession, payload: dict) -> None:
    """Route a job to its handler. Unknown actions raise FatalError → DLQ."""
    action = payload.get("action")
    handler = _handlers.get(action)
    if handler is None:
        raise FatalError(f"No handler registered for action={action!r}")
    await handler(session, payload)


# ── ACL / membership handlers (connector-agnostic) ─────────────────


@register_handler("acl_update")
async def handle_acl_update(session: AsyncSession, payload: dict) -> None:
    """Apply an ACL change to every chunk of one document. SQL only — no
    re-chunking, no re-embedding. The whole point of the metadata-only path.
    """
    record = payload.get("record") or {}
    source_id = payload.get("source_id") or record.get("source_id")
    acl = record.get("acl")
    if not source_id:
        raise FatalError("acl_update payload missing source_id")
    if not isinstance(acl, dict):
        raise FatalError("acl_update payload missing acl dict")

    chunks_updated = await update_document_acl(session, source_id, acl)
    logger.info(
        "acl_update source_id=%s chunks_updated=%d", source_id, chunks_updated
    )


@register_handler("group_membership")
async def handle_group_membership(session: AsyncSession, payload: dict) -> None:
    """Upsert or remove a (user → group) row in the user_groups cache.

    Record shape:
      {"user_id": "alice@acme.com", "group_id": "...", "operation": "add"|"remove"}
    `provider` falls back to the payload-level provider when not on the record.
    """
    record = payload.get("record") or {}
    user_id = record.get("user_id") or record.get("email")
    group_id = record.get("group_id")
    provider = record.get("provider") or payload.get("provider")
    operation = (record.get("operation") or "add").lower()

    if not (user_id and group_id and provider):
        raise FatalError(
            "group_membership payload missing user_id/group_id/provider"
        )

    if operation == "remove":
        await session.execute(
            text("""
                DELETE FROM user_groups
                WHERE user_id = :u AND group_id = :g AND provider = :p
            """),
            {"u": user_id.lower(), "g": group_id, "p": provider},
        )
        logger.info(
            "group_membership remove user=%s group=%s provider=%s",
            user_id, group_id, provider,
        )
    else:
        await session.execute(
            text("""
                INSERT INTO user_groups (user_id, group_id, provider, updated_at)
                VALUES (:u, :g, :p, now())
                ON CONFLICT (user_id, group_id, provider)
                DO UPDATE SET updated_at = now()
            """),
            {"u": user_id.lower(), "g": group_id, "p": provider},
        )
        logger.info(
            "group_membership add user=%s group=%s provider=%s",
            user_id, group_id, provider,
        )


# ── Connector content handlers (placeholders, replaced in Phase 4) ──
#
# upsert/delete need per-connector logic to fetch the actual document from
# Nango and run it through the parse/chunk/embed pipeline. Until Phase 4
# wires those up, these no-op handlers let smoke tests flow events through
# the queue end-to-end without erroring out.


@register_handler("upsert")
async def handle_upsert_placeholder(session: AsyncSession, payload: dict) -> None:
    logger.info(
        "upsert (placeholder) provider=%s source_id=%s — connector handler "
        "not yet registered (Phase 4)",
        payload.get("provider"), payload.get("source_id"),
    )


@register_handler("delete")
async def handle_delete_placeholder(session: AsyncSession, payload: dict) -> None:
    logger.info(
        "delete (placeholder) provider=%s source_id=%s — connector handler "
        "not yet registered (Phase 4)",
        payload.get("provider"), payload.get("source_id"),
    )
