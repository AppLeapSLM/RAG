from __future__ import annotations

from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession

from backend.config import settings
from backend.db.models import Chunk


async def keyword_search(
    query: str,
    session: AsyncSession,
    top_k: int = settings.top_k,
) -> list[Chunk]:
    """BM25-style keyword search using PostgreSQL full-text search.

    Uses ts_rank_cd (cover density ranking) which approximates BM25 behavior.
    Requires the search_vector column and GIN index created in lifespan.
    """
    stmt = text("""
        SELECT id, document_id, content, chunk_index, embedding, metadata, created_at,
               ts_rank_cd(search_vector, websearch_to_tsquery('english', :query)) AS rank
        FROM chunks
        WHERE search_vector @@ websearch_to_tsquery('english', :query)
        ORDER BY rank DESC
        LIMIT :top_k
    """)

    result = await session.execute(stmt, {"query": query, "top_k": top_k})
    rows = result.fetchall()

    # Map raw rows back to Chunk objects
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
