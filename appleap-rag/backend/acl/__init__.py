"""Access control for retrieved chunks.

Each chunk's metadata JSONB carries an optional `acl` block:
    {
        "acl": {
            "allow_users":   ["alice@acme.com", ...],
            "allow_groups":  ["google-group:eng@acme.com", "github:org/team", ...],
            "allow_domains": ["acme.com", ...],
            "public":        false
        }
    }

Chunks without an `acl` key are treated as public — pre-connector corpus data
stays visible until a connector backfills ACL. Connectors writing into the
pipeline are expected to set this block on every chunk they own.

Group memberships are NOT expanded at ingest. They're stored as opaque group
identifiers on the chunk; the user's group set is loaded from the user_groups
cache at query time and the SQL WHERE clause does an overlap check.
"""

from __future__ import annotations

from sqlalchemy import bindparam, select, text
from sqlalchemy.dialects.postgresql import ARRAY
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.sql.elements import TextClause
from sqlalchemy.types import String as SAString

from backend.db.models import Chunk, Document, UserGroup


async def resolve_user_groups(session: AsyncSession, user_email: str) -> list[str]:
    """Return all group IDs the user belongs to (across providers).

    Empty list = user not in any cached group. Combined with the public/user
    clauses, they'll still see public docs and docs explicitly shared to them.
    """
    rows = await session.execute(
        select(UserGroup.group_id).where(UserGroup.user_id == user_email)
    )
    return [g for (g,) in rows.all()]


def _build_acl_clauses(
    user_email: str,
    user_groups: list[str],
    chunk_table: str,
) -> tuple[list[str], dict[str, object]]:
    """Compose the OR-clauses + parameter dict that decide visibility.

    Returns the list of SQL strings (not yet OR-joined) and a params dict so
    callers can either wrap in text() with bindparams or interpolate into raw
    SQL directly.

    Uses jsonb_exists / jsonb_exists_any function forms instead of the `?`
    and `?|` operators — `?` collides with DBAPI's positional-placeholder
    syntax inside SQLAlchemy text() and silently breaks parameter binding.
    No `::text` casts on bind params for the same reason: `:name::cast`
    fails SA's `:name` regex (negative lookahead for `:`).
    """
    domain = user_email.split("@", 1)[1].lower() if "@" in user_email else ""

    clauses: list[str] = [
        # No ACL key = public corpus default (don't break pre-connector data)
        f"NOT jsonb_exists({chunk_table}.metadata, 'acl')",
        # Explicit public flag
        f"({chunk_table}.metadata->'acl'->>'public') = 'true'",
        # User explicitly listed
        f"({chunk_table}.metadata->'acl'->'allow_users') @> to_jsonb(:acl_user)",
    ]
    params: dict[str, object] = {"acl_user": user_email.lower()}

    if user_groups:
        clauses.append(
            f"jsonb_exists_any({chunk_table}.metadata->'acl'->'allow_groups', :acl_groups)"
        )
        params["acl_groups"] = user_groups

    if domain:
        clauses.append(
            f"({chunk_table}.metadata->'acl'->'allow_domains') @> to_jsonb(:acl_domain)"
        )
        params["acl_domain"] = domain

    return clauses, params


def acl_filter_textclause(
    user_email: str,
    user_groups: list[str],
    chunk_table: str = "chunks",
) -> TextClause:
    """Pre-bound TextClause for use in ORM .where(...) on Chunk.

    All bind params are attached to the clause, so the caller doesn't need to
    pass them separately to session.execute().
    """
    clauses, params = _build_acl_clauses(user_email, user_groups, chunk_table)
    fragment = "(" + " OR ".join(clauses) + ")"

    binds = [bindparam("acl_user", value=params["acl_user"])]
    if "acl_groups" in params:
        binds.append(
            bindparam("acl_groups", value=params["acl_groups"], type_=ARRAY(SAString))
        )
    if "acl_domain" in params:
        binds.append(bindparam("acl_domain", value=params["acl_domain"]))

    return text(fragment).bindparams(*binds)


def acl_filter_raw(
    user_email: str,
    user_groups: list[str],
    chunk_table: str = "c",
) -> tuple[str, dict[str, object]]:
    """SQL fragment + params dict for embedding into raw-SQL queries.

    The keyword search path uses this — its main query is already a text()
    statement so we splice the fragment in directly.
    """
    clauses, params = _build_acl_clauses(user_email, user_groups, chunk_table)
    fragment = "(" + " OR ".join(clauses) + ")"
    return fragment, params


async def update_document_acl(
    session: AsyncSession,
    source_id: str,
    acl: dict,
) -> int:
    """Rewrite the `acl` block on every chunk belonging to a document.

    This is the metadata-only update path: NO chunking, NO embedding, NO
    delete-and-reinsert. Triggered by connector ACL change events.

    Returns the number of chunks updated. 0 if no document with that
    source_id exists (e.g. event arrived before the doc was ingested — the
    next content event will set the ACL fresh).
    """
    doc_row = await session.execute(
        select(Document.id).where(
            Document.metadata_["source_id"].astext == source_id
        )
    )
    doc_id = doc_row.scalar_one_or_none()
    if not doc_id:
        return 0

    # jsonb_set auto-creates the 'acl' key if it doesn't already exist on the
    # chunk — we don't need to handle the missing-key case separately.
    result = await session.execute(
        text("""
            UPDATE chunks
            SET metadata = jsonb_set(metadata, '{acl}', :acl::jsonb, true)
            WHERE document_id = :doc_id
        """),
        {"acl": __import__("json").dumps(acl), "doc_id": doc_id},
    )

    # Mirror onto the document row too so future ACL reads have a single
    # source of truth even if chunks get re-derived.
    await session.execute(
        text("""
            UPDATE documents
            SET metadata = jsonb_set(metadata, '{acl}', :acl::jsonb, true)
            WHERE id = :doc_id
        """),
        {"acl": __import__("json").dumps(acl), "doc_id": doc_id},
    )

    return result.rowcount or 0
