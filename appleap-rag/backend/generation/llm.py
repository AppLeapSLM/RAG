import httpx

from backend.config import settings

SYSTEM_PROMPT = (
    "You are an IT Operations assistant for an enterprise. "
    "Answer the user's question using ONLY the provided context. "
    "If the context does not contain enough information, say so — "
    "do not make up answers. Cite which parts of the context you used."
)


async def generate_answer(question: str, context_chunks: list[str]) -> str:
    """Send retrieved context + question to Phi-4 via Ollama and return the answer."""
    context_block = "\n\n---\n\n".join(
        f"[Chunk {i + 1}]\n{chunk}" for i, chunk in enumerate(context_chunks)
    )

    prompt = (
        f"Context:\n{context_block}\n\n"
        f"---\n\n"
        f"Question: {question}\n\n"
        f"Answer:"
    )

    async with httpx.AsyncClient() as client:
        response = await client.post(
            f"{settings.ollama_base_url}/api/generate",
            json={
                "model": settings.llm_model,
                "system": SYSTEM_PROMPT,
                "prompt": prompt,
                "stream": False,
            },
            timeout=600.0,
        )
        response.raise_for_status()
        return response.json()["response"]
