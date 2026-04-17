from __future__ import annotations

import asyncio
import logging

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from backend.config import settings
from backend.db.models import Chunk
from backend.embedding.embedder import embed_text
from backend.retrieval.keyword_search import keyword_search
from backend.retrieval.reranker import rerank

logger = logging.getLogger(__name__)

# Asymmetric over-fetch — calibrated to retrieval_diagnostic.py observations:
#   - Vector retrieval hits in the top ~10 or not at all; going wider is
#     wasted compute for the reranker.
#   - Keyword retrieval has slower decay — correct chunks have been observed
#     at ranks 38 and 45 — so we need a wider keyword net.
VECTOR_OVERFETCH = 15
KEYWORD_OVERFETCH = 50


async def search(
    query: str,
    session: AsyncSession,
    top_k: int = settings.top_k,
    neighbor_window: int = settings.neighbor_window,
) -> list[Chunk]:
    """Hybrid search: vector + keyword → union → cross-encoder rerank → top_k.

    1. Vector search → top VECTOR_OVERFETCH by cosine distance
    2. Keyword search → top KEYWORD_OVERFETCH by ts_rank_cd
       (both retrievers run in parallel)
    3. Union and dedupe by chunk_id — retrieval = coverage
    4. Cross-encoder rerank all candidates → top_k — reranker = precision
    5. Optional neighbor expansion (±neighbor_window)
    6. Final results sorted by (document_id, chunk_index) for reading order

    No RRF: a cross-encoder overrides whatever order RRF produces, so RRF
    becomes dead weight. See CLAUDE.md V8 Phase 2.
    """
    vector_task = _vector_search(query, session, top_k=VECTOR_OVERFETCH)
    keyword_task = keyword_search(query, session, top_k=KEYWORD_OVERFETCH)
    vector_results, keyword_results = await asyncio.gather(vector_task, keyword_task)

    # Union + dedupe by chunk_id. First occurrence wins; order doesn't matter
    # because the reranker re-scores everything.
    pool: dict[str, Chunk] = {}
    for c in vector_results:
        pool.setdefault(c.id, c)
    for c in keyword_results:
        pool.setdefault(c.id, c)
    candidates = list(pool.values())

    logger.info(
        "Hybrid pool: %d vector + %d keyword = %d unique candidates",
        len(vector_results), len(keyword_results), len(candidates),
    )

    reranked = await rerank(query, candidates, top_k=top_k)

    if reranked and neighbor_window > 0:
        reranked = await _expand_neighbors(reranked, session, neighbor_window)

    return reranked


async def _vector_search(
    query: str,
    session: AsyncSession,
    top_k: int,
) -> list[Chunk]:
    """Pure vector similarity search via pgvector cosine distance."""
    query_embedding = await embed_text(query)

    stmt = (
        select(Chunk)
        .order_by(Chunk.embedding.cosine_distance(query_embedding))
        .limit(top_k)
    )
    result = await session.execute(stmt)
    return list(result.scalars().all())


async def _expand_neighbors(
    hits: list[Chunk],
    session: AsyncSession,
    window: int,
) -> list[Chunk]:
    """Pull ±window adjacent chunks from the same document for each hit."""
    seen_ids: set[str] = {h.id for h in hits}
    neighbors: list[Chunk] = []

    for hit in hits:
        low = max(0, hit.chunk_index - window)
        high = hit.chunk_index + window
        stmt = (
            select(Chunk)
            .where(
                Chunk.document_id == hit.document_id,
                Chunk.chunk_index >= low,
                Chunk.chunk_index <= high,
            )
        )
        result = await session.execute(stmt)
        for chunk in result.scalars().all():
            if chunk.id not in seen_ids:
                seen_ids.add(chunk.id)
                neighbors.append(chunk)

    all_chunks = hits + neighbors
    all_chunks.sort(key=lambda c: (c.document_id, c.chunk_index))
    return all_chunks
