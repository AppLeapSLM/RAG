from __future__ import annotations

import logging

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from backend.config import settings
from backend.db.models import Chunk
from backend.embedding.embedder import embed_text
from backend.retrieval.keyword_search import keyword_search

logger = logging.getLogger(__name__)

# RRF constant — standard value from the original paper
RRF_K = 60

# Each retriever over-fetches this many candidates before RRF fusion.
# A wider candidate pool means the correct chunk has more chances to be
# surfaced by at least one retriever, even if its raw rank is modest in
# either individual system. Downstream we still return `top_k` to the caller.
RETRIEVER_OVERFETCH = 30


async def search(
    query: str,
    session: AsyncSession,
    top_k: int = settings.top_k,
    neighbor_window: int = settings.neighbor_window,
) -> list[Chunk]:
    """Hybrid search: vector similarity + BM25 keyword search, fused with RRF.

    1. Vector search → RETRIEVER_OVERFETCH closest chunks by cosine distance
    2. Keyword search → RETRIEVER_OVERFETCH best BM25 matches
    3. Reciprocal Rank Fusion to combine both result sets → top_k
    4. Optional neighbor expansion (±neighbor_window)
    5. Return top_k final results sorted by (document_id, chunk_index)
    """
    # Each retriever over-fetches — the final RRF narrows to top_k.
    pool_size = max(top_k, RETRIEVER_OVERFETCH)
    vector_results = await _vector_search(query, session, top_k=pool_size)
    kw_results = await keyword_search(query, session, top_k=pool_size)

    logger.info(
        "Hybrid search: %d vector hits, %d keyword hits",
        len(vector_results),
        len(kw_results),
    )

    # Fuse with RRF
    fused = _reciprocal_rank_fusion(vector_results, kw_results, top_k=top_k)

    # Optional neighbor expansion
    if fused and neighbor_window > 0:
        fused = await _expand_neighbors(fused, session, neighbor_window)

    return fused


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


def _reciprocal_rank_fusion(
    vector_results: list[Chunk],
    keyword_results: list[Chunk],
    top_k: int,
) -> list[Chunk]:
    """Combine two ranked lists using Reciprocal Rank Fusion.

    RRF score = sum of 1/(k + rank) across all lists where the item appears.
    Higher score = appeared in both lists or ranked highly in one.
    """
    scores: dict[str, float] = {}
    chunk_map: dict[str, Chunk] = {}

    for rank, chunk in enumerate(vector_results):
        scores[chunk.id] = scores.get(chunk.id, 0) + 1 / (RRF_K + rank + 1)
        chunk_map[chunk.id] = chunk

    for rank, chunk in enumerate(keyword_results):
        scores[chunk.id] = scores.get(chunk.id, 0) + 1 / (RRF_K + rank + 1)
        chunk_map[chunk.id] = chunk

    # Sort by RRF score descending, take top_k
    ranked_ids = sorted(scores, key=lambda cid: scores[cid], reverse=True)[:top_k]

    # Sort final results by (document_id, chunk_index) for reading order
    results = [chunk_map[cid] for cid in ranked_ids]
    results.sort(key=lambda c: (c.document_id, c.chunk_index))

    return results


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
