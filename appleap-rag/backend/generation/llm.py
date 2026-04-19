from __future__ import annotations

import logging
from collections import defaultdict

import httpx

from backend.config import settings
from backend.db.models import Chunk

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = (
    "You are AppLeap, a technical support assistant for IT Operations. "
    "Your primary directive is to provide perfectly accurate answers based "
    "strictly on the internal company documents provided in the context.\n\n"
    "RULES:\n"
    "1. STRICT GROUNDING: Answer using ONLY information from the provided context. "
    "Do not use your pre-trained knowledge to fill in gaps or speculate.\n"
    "2. REFUSAL: If the context does not contain the information needed to answer "
    'the query, output exactly: "I could not find this information in the '
    'available documents."\n'
    "3. CITATION: Every factual claim must cite its source using: "
    "[Document: X | Section: Y].\n"
    "4. CASUAL CONVERSATION: If the query is a greeting, farewell, or casual remark, "
    "ignore the context and respond naturally as a helpful colleague.\n"
    "5. FORMAT: Use Markdown (bold, code blocks, bullet points) for technical content. "
    "Be concise and direct."
)

REWRITE_PROMPT = (
    "You are a query rewriter for an IT operations RAG system. "
    "Given a conversation history and the user's latest follow-up question, "
    "rewrite the follow-up into a fully self-contained standalone question "
    "that captures all necessary context from the conversation.\n\n"
    "RULES:\n"
    "- Resolve all pronouns (it, they, that, this, etc.) to their explicit referents.\n"
    "- Preserve the original intent and specificity of the question.\n"
    "- If the question is already self-contained, return it unchanged.\n"
    "- Output ONLY the rewritten question — no explanation, no preamble.\n"
)


# ── Low-level LLM call ─────────────────────────────────────────────


async def _llm_generate(system: str, prompt: str, model: str | None = None) -> str:
    """Send a prompt to the LLM and return the response text.

    This is the single point of contact with the inference backend.
    Currently uses Ollama /api/generate. When the backend changes,
    only this function needs updating.
    """
    model = model or settings.llm_model

    async with httpx.AsyncClient() as client:
        response = await client.post(
            f"{settings.ollama_base_url}/api/generate",
            json={
                "model": model,
                "system": system,
                "prompt": prompt,
                "stream": False,
                "options": {"num_ctx": settings.llm_num_ctx},
            },
            timeout=600.0,
        )
        response.raise_for_status()
        data = response.json()
        logger.info(
            "llm_generate model=%s num_ctx=%d prompt_tokens=%s eval_tokens=%s",
            model,
            settings.llm_num_ctx,
            data.get("prompt_eval_count"),
            data.get("eval_count"),
        )
        return data["response"]


# ── Query rewriting ────────────────────────────────────────────────


def _format_history_for_rewrite(history: list[dict]) -> str:
    """Format conversation history for the query rewriting prompt.

    Each entry in history is {"role": "user"|"assistant", "content": "..."}.
    """
    lines = []
    for msg in history:
        role = "User" if msg["role"] == "user" else "Assistant"
        lines.append(f"{role}: {msg['content']}")
    return "\n".join(lines)


async def rewrite_query(question: str, history: list[dict]) -> str:
    """Rewrite a follow-up question to be self-contained using conversation history.

    If there's no history, returns the question unchanged.
    history is a list of {"role": "user"|"assistant", "content": "..."} dicts.
    """
    if not history:
        return question

    history_text = _format_history_for_rewrite(history)

    prompt = (
        f"Conversation history:\n{history_text}\n\n"
        f"Follow-up question: {question}\n\n"
        f"Standalone question:"
    )

    rewritten = await _llm_generate(REWRITE_PROMPT, prompt)
    return rewritten.strip()


# ── Context building ───────────────────────────────────────────────


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


# ── History formatting ─────────────────────────────────────────────


def build_history_block(history: list[dict]) -> str:
    """Format conversation history for inclusion in the generation prompt.

    history is a list of {"role": "user"|"assistant", "content": "..."} dicts,
    ordered chronologically (oldest first).
    """
    if not history:
        return ""

    lines = []
    for msg in history:
        if msg["role"] == "user":
            lines.append(f"User: {msg['content']}")
        else:
            lines.append(f"Assistant: {msg['content']}")

    return "Previous conversation:\n" + "\n\n".join(lines)


# ── Answer generation ──────────────────────────────────────────────


async def generate_answer(
    question: str,
    chunks: list[Chunk],
    history: list[dict] | None = None,
) -> str:
    """Send retrieved context + question (with optional conversation history)
    to the LLM and return the answer.

    history is a list of {"role": "user"|"assistant", "content": "..."} dicts.
    """
    context_block = build_context_block(chunks)
    history_block = build_history_block(history) if history else ""

    parts: list[str] = []

    # Include conversation history so the LLM has full context
    if history_block:
        parts.append(history_block)
        parts.append("---\n")

    if context_block:
        parts.append(f"Context from company documents:\n\n{context_block}")
        parts.append("---\n")
        parts.append(f"Question: {question}\n\nAnswer:")
    else:
        parts.append(f"No company documents were found for this question.\n\n")
        parts.append(f"Question: {question}\n\nAnswer from your general knowledge:")

    prompt = "\n".join(parts)

    return await _llm_generate(SYSTEM_PROMPT, prompt)
