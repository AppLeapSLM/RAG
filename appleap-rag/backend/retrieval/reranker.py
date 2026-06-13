"""Cross-encoder reranker — Phase 2.

Replaces RRF: given a unioned candidate pool from vector + keyword retrieval,
the cross-encoder scores each (query, chunk) pair and returns the top_k by
relevance.

- Model: BAAI/bge-reranker-v2-m3 (~568M, multilingual, strong on technical text).
- Lazy-loaded module-level singleton — first call pays the load cost (~2-5s cold).
- Fail loud: if load or inference fails, the exception propagates. No RRF
  fallback by design — fallbacks rot when unexercised.
"""

from __future__ import annotations

import asyncio
import json
import logging
import math
import os
import threading
from typing import Any

from backend.config import settings
from backend.db.models import Chunk

logger = logging.getLogger(__name__)

_MODEL_NAME = "BAAI/bge-reranker-v2-m3"
_BATCH_SIZE = 32

# Invert the CrossEncoder's default sigmoid so the relative-to-top band runs on
# raw logits (where strong matches stay far above near-duplicates).
_LOGIT_CLAMP = 1e-7


def _to_logit(s: float) -> float:
    s = min(max(float(s), _LOGIT_CLAMP), 1.0 - _LOGIT_CLAMP)
    return math.log(s / (1.0 - s))

_model: Any = None
_model_lock = threading.Lock()


def _get_model():
    """Load the CrossEncoder on first use. Thread-safe; later callers reuse."""
    global _model
    if _model is not None:
        return _model

    with _model_lock:
        if _model is not None:
            return _model

        logger.info("Loading cross-encoder %s ...", _MODEL_NAME)
        from sentence_transformers import CrossEncoder  # type: ignore

        _model = CrossEncoder(_MODEL_NAME)
        logger.info(
            "Cross-encoder loaded (device=%s)",
            getattr(_model, "device", "unknown"),
        )
        return _model


def _score_sync(query: str, texts: list[str]) -> list[float]:
    """Score (query, text) pairs synchronously. Caller wraps in asyncio.to_thread."""
    model = _get_model()
    pairs = [(query, t) for t in texts]
    scores = model.predict(pairs, batch_size=_BATCH_SIZE, show_progress_bar=False)
    return [float(s) for s in scores]


def _select_count(scores_desc: list[float], floor: int, max_k: int, delta: float) -> int:
    """How many of the score-sorted candidates to keep (relative-to-top band).

    Keep every chunk whose score is within `delta` (logit space) of the top
    score, then clamp to [floor, max_k]. Relative-to-top — not an absolute
    threshold — so it adapts to each query's own score scale (a sharp query
    tops out near +8, a vague one near -2; the band is measured from that top).

    floor == old fixed top_k, so the kept set is always a superset of the old
    top-`floor` → recall is monotonic vs the fixed-k baseline (never worse).

    To swap in the largest-gap rule instead, replace the `band` line with a
    gap scan over scores_desc[floor:] and cut at the maximum consecutive drop.
    """
    n = len(scores_desc)
    if n == 0:
        return 0
    top = scores_desc[0]
    band = sum(1 for s in scores_desc if s >= top - delta)
    n_keep = max(band, min(floor, n))   # never below the floor
    n_keep = min(n_keep, max_k)          # never above the ceiling
    return n_keep


async def rerank(
    query: str,
    chunks: list[Chunk],
    floor: int | None = None,
    max_k: int | None = None,
    delta: float | None = None,
) -> list[Chunk]:
    """Rerank a candidate pool with the cross-encoder, then dynamically select
    how many to keep via a relative-to-top score band (see `_select_count`).

    Returns the kept chunks sorted by (document_id, chunk_index) for reading
    order — matches the convention the generation layer expects.
    """
    if not chunks:
        return []

    floor = settings.rerank_floor if floor is None else floor
    max_k = settings.rerank_max_k if max_k is None else max_k
    delta = settings.rerank_relative_delta if delta is None else delta

    texts = [c.content for c in chunks]
    scores = await asyncio.to_thread(_score_sync, query, texts)

    # bge-reranker-v2-m3 (via sentence-transformers) returns SIGMOID scores in
    # (0,1). Near the top these saturate — many near-duplicate docs all read
    # ~0.99 — so a relative-to-top band on sigmoid scores can't separate the one
    # true answer from look-alikes (every precise query over-fetched to the
    # ceiling). Band in LOGIT space instead: an exact match sits far above the
    # near-duplicates (e.g. 9.3 vs 3.6 where sigmoid showed 0.9999 vs 0.973).
    scores = [_to_logit(s) for s in scores]

    scored = sorted(zip(chunks, scores), key=lambda p: p[1], reverse=True)

    # Diagnostic dump (env-gated, no behavior change): record the FULL ranked
    # candidate list per query as (sigmoid, raw_logit, content_prefix) so delta
    # can be tuned OFFLINE over a grid in both sigmoid and logit space.
    if os.getenv("APPLEAP_RERANK_DUMP"):
        try:
            def _logit(s: float) -> float:
                s = min(max(s, 1e-7), 1.0 - 1e-7)
                return math.log(s / (1.0 - s))
            rec = {
                "query": query,
                "scored": [[round(float(s), 5), round(_logit(s), 4), c.content[:110]]
                           for c, s in scored],
            }
            with open(os.getenv("APPLEAP_RERANK_DUMP"), "a") as fh:
                fh.write(json.dumps(rec) + "\n")
        except Exception:
            logger.exception("rerank dump failed (non-fatal)")

    n_keep = _select_count([s for _, s in scored], floor, max_k, delta)

    logger.info(
        "Rerank select: %d candidates -> kept %d (top=%.3f, band-cut=%.3f, floor=%d, max_k=%d, delta=%.2f)",
        len(scored), n_keep, scored[0][1], scored[0][1] - delta, floor, max_k, delta,
    )

    top = [c for c, _ in scored[:n_keep]]
    top.sort(key=lambda c: (c.document_id, c.chunk_index))
    return top
