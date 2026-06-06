"""Table catalog — the index the router retrieves over to decide, WITH evidence,
whether a question needs a SQL computation and which table to run it on.

One catalog entry per queryable table (a CSV/TSV, or one Excel sheet), built at
ingest from the document's Phase-1 `table_schema`. Each entry's `description`
(filename + columns + row count) is embedded, so finding the relevant table(s)
for a query is a top-K pgvector search — the same retrieve-then-decide pattern
as RAG, applied to schemas instead of text. This is what keeps routing scalable
and accurate: the routing decision is grounded in the FEW retrieved candidate
tables, never the whole catalog, and never a blind guess.

Population happens at (re-)ingest (index_document_tables, called from the ingest
handlers). Existing pre-Phase-1 documents gain entries when re-ingested.
retrieve_tables() is the read side, consumed by the Phase-4 routing decision.
"""

from __future__ import annotations

import logging
from typing import Any

from sqlalchemy import and_, or_, select
from sqlalchemy.ext.asyncio import AsyncSession

from backend.db.models import Document, TableCatalog
from backend.embedding.embedder import embed_batch, embed_text
# Reuse the SAME table-naming as the SQL builder so a catalog entry's
# table_name matches what load_tables() will produce at query time.
from backend.retrieval.tabular_sql import _table_name

logger = logging.getLogger(__name__)


# ── Build (pure) ──────────────────────────────────────────────────────────


def build_catalog_entries(
    table_schema: dict[str, Any] | None, filename: str
) -> list[dict[str, Any]]:
    """Turn a document's `table_schema` into catalog entry dicts. Pure — no DB,
    no embedding. Iterates the schema's tables in order and dedupes table names
    exactly as load_tables() does, so names line up between catalog and engine.
    """
    tables = (table_schema or {}).get("tables") or []
    entries: list[dict[str, Any]] = []
    used: set[str] = set()
    for t in tables:
        cols = t.get("columns") or []
        if not cols:
            continue
        sheet = t.get("sheet_name")
        row_count = int(t.get("row_count", 0) or 0)
        name = _table_name(sheet, used)
        entries.append({
            "table_name": name,
            "sheet_name": sheet,
            "columns": cols,
            "row_count": row_count,
            "description": _describe(filename, sheet, cols, row_count),
        })
    return entries


def _describe(
    filename: str, sheet: str | None, cols: list[dict], row_count: int
) -> str:
    """Natural-language description embedded for retrieval AND shown to the
    router as the candidate. Deterministic (no LLM) — column names carry the
    signal, so NL phrasing keeps it close to Nomic's query distribution."""
    col_list = ", ".join(str(c.get("name", "")) for c in cols)
    sheet_part = f' (sheet "{sheet}")' if sheet else ""
    return (
        f"Tabular data from {filename}{sheet_part} with {row_count} rows. "
        f"Columns: {col_list}."
    )


# ── Index (write side, called at ingest) ───────────────────────────────────


async def index_document_tables(
    session: AsyncSession,
    document_id: str,
    filename: str,
    doc_meta: dict[str, Any] | None,
) -> int:
    """Build + embed + add catalog rows for one ingested document. Returns the
    number of tables indexed (0 for non-tabular docs). Adds to the session but
    does NOT commit — the caller controls the transaction. Embedding happens
    before any session.add so an embed failure leaves the session clean.
    """
    entries = build_catalog_entries((doc_meta or {}).get("table_schema"), filename)
    if not entries:
        return 0

    embeddings = await embed_batch([e["description"] for e in entries])
    for entry, emb in zip(entries, embeddings):
        session.add(TableCatalog(
            document_id=document_id,
            table_name=entry["table_name"],
            sheet_name=entry["sheet_name"],
            columns=entry["columns"],
            row_count=entry["row_count"],
            description=entry["description"],
            embedding=emb,
        ))
    return len(entries)


# ── Retrieve (read side, consumed by Phase-4 routing) ───────────────────────


async def retrieve_tables(
    session: AsyncSession,
    query: str,
    conversation_id: str | None = None,
    top_k: int = 5,
    max_distance: float | None = None,
) -> list[dict[str, Any]]:
    """Top-K candidate tables for `query`, ranked by pgvector cosine distance,
    scoped to corpus tables + this conversation's attachment tables (never
    another conversation's). Returns lightweight dicts the router/SQL step use.

    `max_distance` (cosine distance, 0=identical .. 2=opposite) optionally drops
    weak matches, so when nothing relevant exists the SQL option isn't offered
    and the decision falls to plain retrieval.
    """
    q_emb = await embed_text(query)

    if conversation_id:
        scope = or_(
            Document.source_type == "corpus",
            and_(
                Document.source_type == "attachment",
                Document.conversation_id == conversation_id,
            ),
        )
    else:
        scope = Document.source_type == "corpus"

    distance = TableCatalog.embedding.cosine_distance(q_emb).label("distance")
    stmt = (
        select(TableCatalog, distance)
        .join(Document, TableCatalog.document_id == Document.id)
        .where(scope)
        .order_by(distance)
        .limit(top_k)
    )
    rows = (await session.execute(stmt)).all()

    out: list[dict[str, Any]] = []
    for tc, dist in rows:
        if max_distance is not None and dist is not None and dist > max_distance:
            continue
        out.append({
            "document_id": tc.document_id,
            "table_name": tc.table_name,
            "sheet_name": tc.sheet_name,
            "columns": tc.columns,
            "row_count": tc.row_count,
            "description": tc.description,
            "distance": float(dist) if dist is not None else None,
        })
    return out
