from __future__ import annotations

import asyncio
import logging

from sqlalchemy import and_, or_, select
from sqlalchemy.ext.asyncio import AsyncSession

from backend.acl import acl_filter_textclause, resolve_user_groups
from backend.config import settings
from backend.db.models import Chunk, Document
from backend.embedding.embedder import embed_text
from backend.retrieval.keyword_search import keyword_search
from backend.retrieval.reranker import rerank

logger = logging.getLogger(__name__)

# Asymmetric over-fetch — calibrated to retrieval_diagnostic.py observations:
#   - Vector retrieval hits in the top ~10 or not at all; going much wider
#     is wasted compute for the reranker. Kept at 25 to round up from our
#     earlier 15 with minimal cost.
#   - Keyword retrieval has slower decay — correct chunks observed at ranks
#     38, 45, and 64 post-normalization — so we need a wider keyword net.
VECTOR_OVERFETCH = 25
KEYWORD_OVERFETCH = 75


async def search(
    query: str,
    session: AsyncSession,
    top_k: int = settings.top_k,
    neighbor_window: int = settings.neighbor_window,
    conversation_id: str | None = None,
    user_email: str | None = None,
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

    Scope:
    - When `conversation_id` is None: corpus only.
    - When set: corpus + chunked attachments tagged to that conversation.

    ACL:
    - When `user_email` is set, both retrievers filter by chunks.metadata->'acl'
      using the user + their cached groups. Chunks without an `acl` key are
      treated as public.
    - When `user_email` is None (no header sent), ACL filtering is bypassed
      entirely. This keeps the eval harness and pre-auth dev paths working;
      auth middleware is expected to set the header in production.
    """
    user_groups: list[str] = []
    if user_email:
        user_groups = await resolve_user_groups(session, user_email)

    vector_task = _vector_search(
        query, session, top_k=VECTOR_OVERFETCH,
        conversation_id=conversation_id,
        user_email=user_email, user_groups=user_groups,
    )
    keyword_task = keyword_search(
        query, session, top_k=KEYWORD_OVERFETCH,
        conversation_id=conversation_id,
        user_email=user_email, user_groups=user_groups,
    )
    # Force-include every chunk from this conversation's attachments. Vector
    # and keyword retrievers can miss attachment chunks (corpus often
    # out-ranks them on generic queries — see CMDB-03 / "Based on
    # DeviceItemID" regression). Sending all attachment chunks straight to
    # the reranker is bounded by the 5MB chat-upload cap and lets the
    # reranker make the final call.
    attachment_task = _all_attachment_chunks(
        session, conversation_id=conversation_id,
        user_email=user_email, user_groups=user_groups,
    )
    vector_results, keyword_results, attachment_results = await asyncio.gather(
        vector_task, keyword_task, attachment_task,
    )

    # Union + dedupe by chunk_id. First occurrence wins; order doesn't matter
    # because the reranker re-scores everything.
    pool: dict[str, Chunk] = {}
    for c in vector_results:
        pool.setdefault(c.id, c)
    for c in keyword_results:
        pool.setdefault(c.id, c)
    for c in attachment_results:
        pool.setdefault(c.id, c)
    candidates = list(pool.values())

    logger.info(
        "Hybrid pool: %d vector + %d keyword + %d attachment = %d unique candidates",
        len(vector_results), len(keyword_results), len(attachment_results), len(candidates),
    )

    reranked = await rerank(query, candidates, top_k=top_k)

    if reranked and neighbor_window > 0:
        reranked = await _expand_neighbors(reranked, session, neighbor_window)

    return reranked


async def _vector_search(
    query: str,
    session: AsyncSession,
    top_k: int,
    conversation_id: str | None = None,
    user_email: str | None = None,
    user_groups: list[str] | None = None,
) -> list[Chunk]:
    """Pure vector similarity search via pgvector cosine distance.

    Scopes results to corpus docs plus (optionally) chunked attachments
    tagged to `conversation_id`. Never returns other conversations' attachments.

    Applies ACL filter when `user_email` is set; bypassed otherwise.
    """
    query_embedding = await embed_text(query)

    if conversation_id:
        scope_filter = or_(
            Document.source_type == "corpus",
            and_(
                Document.source_type == "attachment",
                Document.conversation_id == conversation_id,
            ),
        )
    else:
        scope_filter = Document.source_type == "corpus"

    stmt = (
        select(Chunk)
        .join(Document, Chunk.document_id == Document.id)
        .where(scope_filter)
    )

    if user_email:
        stmt = stmt.where(
            acl_filter_textclause(user_email, user_groups or [], chunk_table="chunks")
        )

    stmt = (
        stmt.order_by(Chunk.embedding.cosine_distance(query_embedding))
        .limit(top_k)
    )
    result = await session.execute(stmt)
    return list(result.scalars().all())


async def _all_attachment_chunks(
    session: AsyncSession,
    conversation_id: str | None = None,
    user_email: str | None = None,
    user_groups: list[str] | None = None,
) -> list[Chunk]:
    """Every chunk from this conversation's chunked attachments. Empty when
    no conversation_id is set (eval / corpus-only path) or no attachments
    exist. Applies the same ACL filter as the other retrievers.
    """
    if not conversation_id:
        return []
    stmt = (
        select(Chunk)
        .join(Document, Chunk.document_id == Document.id)
        .where(Document.source_type == "attachment")
        .where(Document.conversation_id == conversation_id)
    )
    if user_email:
        stmt = stmt.where(
            acl_filter_textclause(user_email, user_groups or [], chunk_table="chunks")
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
