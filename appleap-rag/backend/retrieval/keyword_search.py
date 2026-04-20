from __future__ import annotations

import logging
import re

from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession

from backend.config import settings
from backend.db.models import Chunk

logger = logging.getLogger(__name__)

# Minimal token extractor: alphanumeric runs, 2+ chars. Anything else
# (punctuation, tsquery operators) is stripped — so a user question never
# forms an invalid to_tsquery expression.
_TOKEN_RE = re.compile(r"[A-Za-z0-9]+")


def _build_or_tsquery(query: str) -> str:
    """Turn a free-form question into an OR-connected tsquery string.

    Why OR: websearch_to_tsquery/plainto_tsquery default to AND, which means
    every stemmed token must appear in a chunk for it to match. For long
    natural-language questions (8-10 content words), almost nothing matches.
    We want ts_rank_cd to rank chunks that match more tokens higher — not
    to filter out chunks that miss any single token.
    """
    tokens: list[str] = []
    seen: set[str] = set()
    for raw in _TOKEN_RE.findall(query):
        tok = raw.lower()
        if len(tok) < 2 or tok in seen:
            continue
        seen.add(tok)
        tokens.append(tok)
    return " | ".join(tokens)


async def keyword_search(
    query: str,
    session: AsyncSession,
    top_k: int = settings.top_k,
    conversation_id: str | None = None,
) -> list[Chunk]:
    """BM25-style keyword search using PostgreSQL full-text search with OR semantics.

    Uses ts_rank_cd (cover density ranking) — chunks containing more query
    tokens rank higher. Builds the tsquery with `|` (OR) so long questions
    don't require every stem to appear in a single chunk.

    Length normalization (flag 1): divides rank by `1 + log(doc length)`.
    Without this, density-biased ts_rank_cd lets long repetitive files
    (Helm values with N services × M knobs) outrank short focused chunks
    (a single K8s manifest) because the long file accumulates more token
    hits. The log-scale penalty evens this out without brutally crushing
    medium-sized chunks.

    Scope (via JOIN documents):
    - conversation_id=None → corpus only
    - conversation_id set  → corpus + attachments for that conversation
    """
    tsquery = _build_or_tsquery(query)
    if not tsquery:
        return []

    if conversation_id:
        scope_clause = (
            "AND (d.source_type = 'corpus' "
            "OR (d.source_type = 'attachment' AND d.conversation_id = :conv_id))"
        )
    else:
        scope_clause = "AND d.source_type = 'corpus'"

    stmt = text(f"""
        SELECT c.id, c.document_id, c.content, c.chunk_index, c.embedding,
               c.metadata, c.created_at,
               ts_rank_cd(c.search_vector, to_tsquery('english', :tsquery), 1) AS rank
        FROM chunks c
        JOIN documents d ON c.document_id = d.id
        WHERE c.search_vector @@ to_tsquery('english', :tsquery)
        {scope_clause}
        ORDER BY rank DESC
        LIMIT :top_k
    """)

    params: dict = {"tsquery": tsquery, "top_k": top_k}
    if conversation_id:
        params["conv_id"] = conversation_id

    try:
        result = await session.execute(stmt, params)
        rows = result.fetchall()
    except Exception as e:
        # Defensive: if to_tsquery ever rejects the generated expression
        # (e.g., every token was a stop word), fall back to empty results
        # rather than failing the whole query.
        logger.warning("keyword_search tsquery failed for %r: %s", tsquery, e)
        return []

    chunks: list[Chunk] = []
    for row in rows:
        chunk = Chunk(
            id=row.id,
            document_id=row.document_id,
            content=row.content,
            chunk_index=row.chunk_index,
            embedding=row.embedding,
            metadata_=row.metadata,
            created_at=row.created_at,
        )
        chunks.append(chunk)

    return chunks
