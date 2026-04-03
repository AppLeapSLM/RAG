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


async def embed_batch(texts: list[str]) -> list[list[float]]:
    """Embed multiple texts in one call."""
    async with httpx.AsyncClient() as client:
        response = await client.post(
            f"{settings.ollama_base_url}/api/embed",
            json={"model": settings.embedding_model, "input": texts},
            timeout=120.0,
        )
        response.raise_for_status()
        return response.json()["embeddings"]
