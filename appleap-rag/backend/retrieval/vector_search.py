from sqlalchemy import select, text
from sqlalchemy.ext.asyncio import AsyncSession

from backend.config import settings
from backend.db.models import Chunk
from backend.embedding.embedder import embed_text


async def search(
    query: str, session: AsyncSession, top_k: int = settings.top_k
) -> list[Chunk]:
    """Embed the query and find the closest chunks via pgvector cosine distance."""
    query_embedding = await embed_text(query)

    # pgvector cosine distance operator: <=>
    stmt = (
        select(Chunk)
        .order_by(Chunk.embedding.cosine_distance(query_embedding))
        .limit(top_k)
    )

    result = await session.execute(stmt)
    return list(result.scalars().all())
