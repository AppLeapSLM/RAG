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
import logging
import threading
from typing import Any

from backend.db.models import Chunk

logger = logging.getLogger(__name__)

_MODEL_NAME = "BAAI/bge-reranker-v2-m3"
_BATCH_SIZE = 32

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


async def rerank(query: str, chunks: list[Chunk], top_k: int) -> list[Chunk]:
    """Rerank a candidate pool with the cross-encoder.

    Returns the top_k chunks sorted by (document_id, chunk_index) for reading
    order — matches the convention the generation layer expects.
    """
    if not chunks:
        return []

    texts = [c.content for c in chunks]
    scores = await asyncio.to_thread(_score_sync, query, texts)

    scored = sorted(zip(chunks, scores), key=lambda p: p[1], reverse=True)
    top = [c for c, _ in scored[:top_k]]
    top.sort(key=lambda c: (c.document_id, c.chunk_index))
    return top
