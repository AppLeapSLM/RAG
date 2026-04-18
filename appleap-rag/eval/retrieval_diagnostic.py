"""Retrieval diagnostic — runs vector and keyword search independently against
a set of known-failing eval queries and reports where the expected chunk ranks
in each. Tells us which retriever is blind vs. just ranking badly.

Run locally on the VM (where the DB is) with:
    PYTHONPATH=. python3.13 -m eval.retrieval_diagnostic
"""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
from typing import Any

from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession

from backend.config import settings
from backend.db.connection import async_session
from backend.db.models import Chunk
from backend.embedding.embedder import embed_text


DIAGNOSTIC_CASES: list[dict[str, Any]] = [
    {
        "id": "K8S-01",
        "query": "What are the CPU and memory resource limits for the API gateway Kubernetes deployment?",
        "expected_filenames": ["api-gateway.yaml"],
    },
    {
        "id": "TF-01",
        "query": "What are the EKS cluster names used in NovaCrest's production Terraform configuration?",
        "expected_filenames": ["eks.tf.json"],
    },
    {
        "id": "CMDB-01",
        "query": "What port does the ingest-service run on and what language is it written in?",
        "expected_filenames": ["cmdb-production.csv"],
    },
    {
        "id": "PUPPET-01",
        "query": "What is the dependency chain order in the Puppet base module?",
        "expected_filenames": ["init.pp"],
    },
    {
        "id": "ARCH-01",
        "query": "How many metrics per minute does the NovaCrest Insights Platform process?",
        "expected_filenames": ["architecture-architecture-overview-novacrest-insights-platform.md"],
    },
]

TOP_N = 50  # deep pool so we can see where expected chunks land


def _doc_filename(chunk: Chunk) -> str:
    """Pull original filename from chunk metadata or title fallback."""
    meta = chunk.metadata_ or {}
    return meta.get("original_filename") or meta.get("title") or ""


def _matches_expected(chunk: Chunk, expected: list[str]) -> bool:
    name = _doc_filename(chunk).lower()
    return any(exp.lower() in name for exp in expected)


async def _vector_search_ranked(
    query: str, session: AsyncSession, top_k: int
) -> list[tuple[Chunk, float]]:
    """Vector search returning (chunk, cosine_distance) tuples for top_k."""
    qvec = await embed_text(query)
    stmt = text("""
        SELECT id, document_id, content, chunk_index, embedding, metadata, created_at,
               embedding <=> CAST(:qvec AS vector) AS distance
        FROM chunks
        ORDER BY distance ASC
        LIMIT :top_k
    """)
    # pgvector literal: cast the list to vector
    result = await session.execute(
        stmt, {"qvec": str(qvec), "top_k": top_k}
    )
    rows = result.fetchall()
    out = []
    for row in rows:
        chunk = Chunk(
            id=row.id, document_id=row.document_id, content=row.content,
            chunk_index=row.chunk_index, embedding=row.embedding,
            metadata_=row.metadata, created_at=row.created_at,
        )
        out.append((chunk, float(row.distance)))
    return out


async def _keyword_search_ranked(
    query: str, session: AsyncSession, top_k: int
) -> list[tuple[Chunk, float]]:
    """Keyword search with OR semantics (matches production keyword_search.py)."""
    from backend.retrieval.keyword_search import _build_or_tsquery

    tsquery = _build_or_tsquery(query)
    if not tsquery:
        return []
    stmt = text("""
        SELECT id, document_id, content, chunk_index, embedding, metadata, created_at,
               ts_rank_cd(search_vector, to_tsquery('english', :tsquery), 1) AS rank
        FROM chunks
        WHERE search_vector @@ to_tsquery('english', :tsquery)
        ORDER BY rank DESC
        LIMIT :top_k
    """)
    result = await session.execute(stmt, {"tsquery": tsquery, "top_k": top_k})
    rows = result.fetchall()
    out = []
    for row in rows:
        chunk = Chunk(
            id=row.id, document_id=row.document_id, content=row.content,
            chunk_index=row.chunk_index, embedding=row.embedding,
            metadata_=row.metadata, created_at=row.created_at,
        )
        out.append((chunk, float(row.rank)))
    return out


def _rank_of_expected(ranked: list[tuple[Chunk, float]], expected: list[str]) -> int | None:
    for i, (chunk, _) in enumerate(ranked):
        if _matches_expected(chunk, expected):
            return i + 1  # 1-indexed
    return None


async def _diagnose_one(case: dict[str, Any], session: AsyncSession) -> None:
    print(f"\n{'=' * 78}")
    print(f"  [{case['id']}]  {case['query']}")
    print(f"  Expected: {case['expected_filenames']}")
    print("-" * 78)

    vec = await _vector_search_ranked(case["query"], session, TOP_N)
    kw = await _keyword_search_ranked(case["query"], session, TOP_N)

    vec_rank = _rank_of_expected(vec, case["expected_filenames"])
    kw_rank = _rank_of_expected(kw, case["expected_filenames"])

    print(f"  VECTOR  (top {TOP_N}): expected at rank {vec_rank or '— NOT FOUND'}")
    print(f"  KEYWORD (top {TOP_N}): expected at rank {kw_rank or '— NOT FOUND'}")

    print("\n  Vector top 5:")
    for i, (c, d) in enumerate(vec[:5]):
        marker = "★" if _matches_expected(c, case["expected_filenames"]) else " "
        preview = c.content.replace("\n", " ")[:110]
        print(f"    {marker} {i+1:2d}. dist={d:.4f}  {_doc_filename(c)[:40]:40s}  | {preview}")

    print("\n  Keyword top 5:")
    for i, (c, r) in enumerate(kw[:5]):
        marker = "★" if _matches_expected(c, case["expected_filenames"]) else " "
        preview = c.content.replace("\n", " ")[:110]
        print(f"    {marker} {i+1:2d}. rank={r:.4f}  {_doc_filename(c)[:40]:40s}  | {preview}")

    # If expected is deeper in the pool, show its chunk preview
    if vec_rank and vec_rank > 5:
        c, d = vec[vec_rank - 1]
        print(f"\n  Expected in vector@{vec_rank} (dist={d:.4f}):")
        print(f"    {c.content[:220]}")
    elif vec_rank is None:
        print(f"\n  Expected NOT in vector top {TOP_N} — deep retrieval problem.")

    if kw_rank and kw_rank > 5:
        c, r = kw[kw_rank - 1]
        print(f"\n  Expected in keyword@{kw_rank} (rank={r:.4f}):")
        print(f"    {c.content[:220]}")
    elif kw_rank is None:
        print(f"  Expected NOT in keyword top {TOP_N} — BM25 is blind to this chunk.")


async def main():
    async with async_session() as session:
        for case in DIAGNOSTIC_CASES:
            await _diagnose_one(case, session)


if __name__ == "__main__":
    asyncio.run(main())
