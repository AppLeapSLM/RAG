from __future__ import annotations

from collections import defaultdict

import httpx

from backend.config import settings
from backend.db.models import Chunk

SYSTEM_PROMPT = (
    "You are AppLeap, a technical support assistant for IT Operations. "
    "You answer questions using ONLY the company documents provided in the context.\n\n"
    "Rules:\n"
    "1. If context from company documents is provided, answer strictly from that context. "
    "Cite the document name and section. Do not add information from your own knowledge.\n"
    "2. If the context does not contain the answer, say: "
    '"I could not find this information in the available documents." '
    "Do not guess or fill in gaps with general knowledge.\n"
    "3. For casual messages with no context provided, respond naturally like a helpful colleague.\n"
    "4. Be concise and direct."
)


def build_context_block(chunks: list[Chunk]) -> str:
    """Build a document-aware context block from retrieved chunks.

    Groups chunks by source document and preserves reading order
    (chunks are already sorted by document_id, chunk_index from retrieval).
    Respects the max_context_chars cap.
    """
    if not chunks:
        return ""

    # Group chunks by document, preserving order
    doc_groups: dict[str, list[Chunk]] = defaultdict(list)
    for chunk in chunks:
        doc_groups[chunk.document_id].append(chunk)

    sections: list[str] = []
    total_chars = 0

    for doc_id, doc_chunks in doc_groups.items():
        # Extract document title from the first chunk's metadata
        meta = doc_chunks[0].metadata_ or {}
        doc_title = meta.get("title", "Unknown Document")

        doc_section = f'From "{doc_title}":\n'

        for chunk in doc_chunks:
            chunk_meta = chunk.metadata_ or {}
            section_name = chunk_meta.get("section", "")

            # Build chunk label
            if section_name:
                label = f"  [{section_name}]"
            else:
                pos = chunk_meta.get("chunk_position", "")
                total = chunk_meta.get("total_chunks", "")
                if pos and total:
                    label = f"  [Part {pos} of {total}]"
                else:
                    label = f"  [Chunk]"

            chunk_text = f"{label}\n  {chunk.content}\n"

            # Enforce context cap
            addition = len(doc_section) + len(chunk_text) if not sections else len(chunk_text)
            if total_chars + addition > settings.max_context_chars:
                break
            doc_section += chunk_text
            total_chars += len(chunk_text)

        sections.append(doc_section)

        if total_chars >= settings.max_context_chars:
            break

    return "\n---\n\n".join(sections)


async def generate_answer(question: str, chunks: list[Chunk]) -> str:
    """Send retrieved context + question to Phi-4 via Ollama and return the answer.

    Accepts full Chunk objects (not just strings) to enable document-aware
    context assembly.
    """
    context_block = build_context_block(chunks)

    if context_block:
        prompt = (
            f"Context from company documents:\n\n"
            f"{context_block}\n"
            f"---\n\n"
            f"Question: {question}\n\n"
            f"Answer:"
        )
    else:
        prompt = (
            f"No company documents were found for this question.\n\n"
            f"Question: {question}\n\n"
            f"Answer from your general knowledge:"
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
