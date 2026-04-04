from __future__ import annotations

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from backend.config import settings
from backend.db.models import Chunk
from backend.embedding.embedder import embed_text


async def search(
    query: str,
    session: AsyncSession,
    top_k: int = settings.top_k,
    neighbor_window: int = settings.neighbor_window,
) -> list[Chunk]:
    """Embed the query, find closest chunks, then expand with neighbors.

    1. Vector search → top_k closest chunks
    2. For each hit, pull ±neighbor_window adjacent chunks from the same document
    3. Deduplicate and sort by (document_id, chunk_index) for coherent reading order
    """
    query_embedding = await embed_text(query)

    # Step 1: vector similarity search
    stmt = (
        select(Chunk)
        .order_by(Chunk.embedding.cosine_distance(query_embedding))
        .limit(top_k)
    )
    result = await session.execute(stmt)
    hits = list(result.scalars().all())

    if not hits or neighbor_window <= 0:
        return hits

    # Step 2: expand neighbors — collect (doc_id, chunk_index) ranges to fetch
    seen_ids: set[str] = {h.id for h in hits}
    neighbor_ranges: list[tuple[str, int, int]] = []

    for hit in hits:
        low = max(0, hit.chunk_index - neighbor_window)
        high = hit.chunk_index + neighbor_window
        neighbor_ranges.append((hit.document_id, low, high))

    # Batch-fetch all neighbors
    neighbors: list[Chunk] = []
    for doc_id, low, high in neighbor_ranges:
        stmt = (
            select(Chunk)
            .where(
                Chunk.document_id == doc_id,
                Chunk.chunk_index >= low,
                Chunk.chunk_index <= high,
            )
        )
        result = await session.execute(stmt)
        for chunk in result.scalars().all():
            if chunk.id not in seen_ids:
                seen_ids.add(chunk.id)
                neighbors.append(chunk)

    # Step 3: merge and sort by (document_id, chunk_index) for reading order
    all_chunks = hits + neighbors
    all_chunks.sort(key=lambda c: (c.document_id, c.chunk_index))

    return all_chunks
