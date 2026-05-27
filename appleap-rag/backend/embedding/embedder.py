import httpx

from backend.config import settings


async def embed_text(text: str) -> list[float]:
    """Embed a single text string using Nomic via Ollama."""
    async with httpx.AsyncClient() as client:
        response = await client.post(
            f"{settings.ollama_base_url}/api/embed",
            json={"model": settings.embedding_model, "input": text},
            timeout=60.0,
        )
        response.raise_for_status()
        return response.json()["embeddings"][0]


async def embed_batch(texts: list[str], batch_size: int = 64) -> list[list[float]]:
    """Embed multiple texts. Ollama caps inputs/tokens per request, so we
    send in chunks of `batch_size` sequentially and concatenate. A 17 MB
    source file producing ~5,400 chunks exceeded the cap and failed with
    HTTP 400; small files (≤ batch_size chunks) still complete in one POST.
    """
    out: list[list[float]] = []
    async with httpx.AsyncClient() as client:
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            response = await client.post(
                f"{settings.ollama_base_url}/api/embed",
                json={"model": settings.embedding_model, "input": batch},
                timeout=120.0,
            )
            response.raise_for_status()
            out.extend(response.json()["embeddings"])
    return out
