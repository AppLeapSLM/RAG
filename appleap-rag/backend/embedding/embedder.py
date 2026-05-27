import logging

import httpx

from backend.config import settings

logger = logging.getLogger(__name__)

# Adaptive embed: starting batch size. On per-batch context-length 400s
# from Ollama, the batch is halved recursively until it fits. 32 is small
# enough that the dx-netops 156MB-PDF case ingests in one pass for most
# batches; large enough that small files still complete in one POST.
DEFAULT_BATCH_SIZE = 32

# Below this length, a chunk text isn't worth splitting further — if it's
# STILL hitting the per-input cap below this, something's badly wrong and
# we let the error propagate rather than chase an infinite recursion.
MIN_SPLIT_CHARS = 100


def _sanitize(text: str) -> str:
    """Strip characters that Ollama's JSON layer rejects. NULL bytes are
    valid in JSON syntactically but rejected by many Go-based parsers."""
    return text.replace("\x00", "")


def _is_context_length_error(response: httpx.Response) -> bool:
    """Ollama returns 400 with this specific message when an input (or the
    aggregate of a batch) exceeds the model's context window. Distinguish
    from other 400s so we don't swallow real bugs."""
    return (
        response.status_code == 400
        and "context length" in response.text
    )


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


async def _embed_with_adaptive_retry(
    client: httpx.AsyncClient, batch: list[str],
) -> list[list[float]]:
    """Send a batch to Ollama, handling context-length overflows by
    recursively halving (per-batch overflow) or splitting the text and
    mean-pooling (per-input overflow). Other errors propagate."""
    response = await client.post(
        f"{settings.ollama_base_url}/api/embed",
        json={"model": settings.embedding_model, "input": batch},
        timeout=120.0,
    )

    if response.status_code < 400:
        return response.json()["embeddings"]

    if not _is_context_length_error(response):
        lens = [len(t) for t in batch]
        logger.error(
            "embed_batch HTTP %d (size %d, char-lens min=%d max=%d total=%d). Body: %r",
            response.status_code, len(batch),
            min(lens), max(lens), sum(lens),
            response.text[:1000],
        )
        response.raise_for_status()

    if len(batch) > 1:
        mid = len(batch) // 2
        logger.info(
            "embed_batch context overflow on batch size %d; halving to %d + %d",
            len(batch), mid, len(batch) - mid,
        )
        left = await _embed_with_adaptive_retry(client, batch[:mid])
        right = await _embed_with_adaptive_retry(client, batch[mid:])
        return left + right

    # Single input still over the per-input cap. Split the text into halves,
    # embed each, mean-pool back to one vector. Preserves the 1:1
    # input→embedding contract so callers don't see the split.
    text = batch[0]
    if len(text) < MIN_SPLIT_CHARS:
        logger.error(
            "embed_batch chunk under %d chars still over context length: %r",
            MIN_SPLIT_CHARS, text[:200],
        )
        response.raise_for_status()

    mid = len(text) // 2
    logger.warning(
        "embed_batch mean-pooling oversized chunk (%d chars) → 2x %d chars",
        len(text), mid,
    )
    left = await _embed_with_adaptive_retry(client, [text[:mid]])
    right = await _embed_with_adaptive_retry(client, [text[mid:]])
    pooled = [(a + b) / 2.0 for a, b in zip(left[0], right[0])]
    return [pooled]


async def embed_batch(
    texts: list[str], batch_size: int = DEFAULT_BATCH_SIZE,
) -> list[list[float]]:
    """Embed multiple texts. Ollama's nomic-embed-text caps at 2048 tokens
    per input AND ~50K tokens per batch (the GGUF hardcodes context length;
    Modelfile PARAMETER num_ctx and options.num_ctx are ignored for /api/embed).

    Adaptive behavior on context-length 400:
      - per-batch overflow → recursively halve the batch and retry
      - per-input overflow (single chunk still rejected) → split the text
        in halves, embed each, mean-pool the two embeddings into one
        (preserves N inputs → N embeddings contract)

    Common case (most chunks well under the cap) completes in one POST.
    """
    out: list[list[float]] = []
    async with httpx.AsyncClient() as client:
        for i in range(0, len(texts), batch_size):
            batch = [_sanitize(t) for t in texts[i:i + batch_size]]
            out.extend(await _embed_with_adaptive_retry(client, batch))
    return out
