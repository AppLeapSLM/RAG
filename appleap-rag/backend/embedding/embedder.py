import logging

import httpx

from backend.config import settings

logger = logging.getLogger(__name__)


def _sanitize(text: str) -> str:
    """Strip characters that Ollama's JSON layer rejects. NULL bytes are
    valid in JSON syntactically but rejected by many Go-based parsers."""
    return text.replace("\x00", "")


async def embed_text(text: str) -> list[float]:
    """Embed a single text string using Nomic via Ollama."""
    async with httpx.AsyncClient() as client:
        response = await client.post(
            f"{settings.ollama_base_url}/api/embed",
            json={"model": settings.embedding_model, "input": _sanitize(text)},
            timeout=60.0,
        )
        response.raise_for_status()
        return response.json()["embeddings"][0]


async def embed_batch(texts: list[str], batch_size: int = 64) -> list[list[float]]:
    """Embed multiple texts. Ollama caps inputs/tokens per request, so we
    send in chunks of `batch_size` sequentially and concatenate. A 17 MB
    source file producing ~5,400 chunks exceeded the cap and failed with
    HTTP 400; small files (≤ batch_size chunks) still complete in one POST.

    On HTTP error, logs the batch index, batch sizes, and Ollama's
    response body so we can diagnose without re-running the whole ingest.
    """
    out: list[list[float]] = []
    async with httpx.AsyncClient() as client:
        for i in range(0, len(texts), batch_size):
            batch = [_sanitize(t) for t in texts[i:i + batch_size]]
            response = await client.post(
                f"{settings.ollama_base_url}/api/embed",
                json={"model": settings.embedding_model, "input": batch},
                timeout=120.0,
            )
            if response.status_code >= 400:
                lens = [len(t) for t in batch]
                logger.error(
                    "embed_batch HTTP %d on batch_idx=%d (offset %d, size %d, "
                    "char-lens min=%d max=%d total=%d). Response body: %r",
                    response.status_code, i // batch_size, i, len(batch),
                    min(lens), max(lens), sum(lens),
                    response.text[:1000],
                )
                response.raise_for_status()
            out.extend(response.json()["embeddings"])
    return out
