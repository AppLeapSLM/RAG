from __future__ import annotations

import json
import logging
from collections import defaultdict
from typing import AsyncIterator

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
    "You are a query interpreter for an IT operations RAG system. Given a "
    "conversation history and the user's latest follow-up, you must decide "
    "whether the follow-up can be answered as-is (after pronoun resolution) "
    "or whether it is too vague / ambiguous to answer confidently. Output "
    "EXACTLY ONE line, starting with one of the two prefixes:\n\n"
    "QUERY: <a fully self-contained standalone question, with pronouns "
    "resolved and needed context folded in from the history>\n"
    "CLARIFY: <one short, focused question to ask the user — at most one "
    "sentence — when the follow-up is genuinely ambiguous or missing a "
    "required entity (e.g., which service, which environment, which time "
    "range)>\n\n"
    "RULES:\n"
    "- Strongly prefer QUERY. Only use CLARIFY when the question cannot be "
    "answered without more information from the user — clarifications cost "
    "the user a turn.\n"
    "- Resolve all pronouns (it, they, that, this, etc.) to their explicit "
    "referents from the history when possible. Coreference is NOT a reason "
    "to clarify if the referent is unambiguous from prior turns.\n"
    "- If the follow-up is already self-contained, output QUERY with it "
    "unchanged.\n"
    "- NEVER output anything except the QUERY: or CLARIFY: line. No "
    "explanation, no preamble, no extra lines.\n"
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
        pt = data.get("prompt_eval_count")
        et = data.get("eval_count")
        # 14000 = 86% of Phi-4's 16K ceiling. Beyond this Ollama starts to
        # silently truncate the prompt and the answer quality degrades.
        if pt is not None and pt > 14000:
            logger.warning(
                "llm_generate prompt_tokens=%d approaching num_ctx=%d (truncation risk)",
                pt, settings.llm_num_ctx,
            )
        else:
            logger.info(
                "llm_generate model=%s num_ctx=%d prompt_tokens=%s eval_tokens=%s",
                model, settings.llm_num_ctx, pt, et,
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


async def rewrite_query(
    question: str, history: list[dict],
) -> tuple[str, str]:
    """Decide whether the follow-up is answerable or needs clarification,
    and return one of:

      ("query",   <self-contained rewritten question>)  → proceed to retrieval
      ("clarify", <question to ask the user>)            → skip retrieval, stop

    Turn 1 (no history) always returns ("query", question) unchanged —
    clarification on a brand-new turn is out of scope to avoid adding an
    LLM call to the cold path.

    If the LLM ignores the QUERY:/CLARIFY: format, the response is treated
    as ("query", raw_response) so behavior degrades gracefully to the
    pre-clarification baseline rather than blocking the user.
    """
    if not history:
        return ("query", question)

    history_text = _format_history_for_rewrite(history)

    prompt = (
        f"Conversation history:\n{history_text}\n\n"
        f"Follow-up question: {question}\n\n"
        f"Your output:"
    )

    raw = (await _llm_generate(REWRITE_PROMPT, prompt)).strip()

    # Strip a leading code fence in case the model wraps its output.
    if raw.startswith("```"):
        raw = raw.strip("`").strip()

    upper = raw.upper()
    if upper.startswith("CLARIFY:"):
        text = raw[len("CLARIFY:"):].strip()
        if not text:
            logger.warning("rewrite_query empty CLARIFY payload — falling back to QUERY")
            return ("query", question)
        logger.info("rewrite_query CLARIFY: %s", text[:200])
        return ("clarify", text)

    if upper.startswith("QUERY:"):
        text = raw[len("QUERY:"):].strip()
        if not text:
            return ("query", question)
        return ("query", text)

    # Malformed — the model ignored the prefix contract. Fall back to the
    # pre-clarification behavior: assume the response IS the rewritten
    # question. Worst case is current production behavior.
    logger.warning("rewrite_query missing prefix; using raw=%r", raw[:200])
    return ("query", raw)


# ── Context building ───────────────────────────────────────────────


def build_context_block(
    chunks: list[Chunk],
    inline_attachments: list | None = None,
) -> str:
    """Build a document-aware context block from retrieved chunks.

    Groups chunks by source document and preserves reading order
    (chunks are already sorted by document_id, chunk_index from retrieval).
    Respects the max_context_chars cap.

    `inline_attachments` — optional list of `ConversationInlineAttachment`
    rows whose full text is prepended as the first section, before any
    retrieved chunks. User-attached files take priority over retrieval.
    Inline attachments bypass the `max_context_chars` cap (they are the
    user's own context and must appear in full — overflow is monitored
    via the prompt_tokens warning in `_llm_generate`).
    """
    sections: list[str] = []

    if inline_attachments:
        for a in inline_attachments:
            sections.append(
                f'From user-attached file "{a.filename}":\n{a.text_content}'
            )

    if not chunks:
        return "\n---\n\n".join(sections) if sections else ""

    # Group chunks by document, preserving order
    doc_groups: dict[str, list[Chunk]] = defaultdict(list)
    for chunk in chunks:
        doc_groups[chunk.document_id].append(chunk)

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
    inline_attachments: list | None = None,
) -> str:
    """Send retrieved context + question (with optional conversation history)
    to the LLM and return the answer.

    history is a list of {"role": "user"|"assistant", "content": "..."} dicts.
    inline_attachments is a list of ConversationInlineAttachment rows whose
    full text is prepended to the context block ahead of retrieved chunks.
    """
    context_block = build_context_block(chunks, inline_attachments=inline_attachments)
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


# ── Streaming generation ───────────────────────────────────────────


async def _llm_generate_stream(
    system: str, prompt: str, model: str | None = None,
) -> AsyncIterator[str]:
    """Stream tokens from Ollama. Yields each delta as it arrives.

    Ollama's /api/generate with stream=true returns NDJSON: one JSON
    object per line, each carrying a `response` field with the next token
    fragment, plus a final object with done=true and stats.
    """
    model = model or settings.llm_model

    async with httpx.AsyncClient(timeout=600.0) as client:
        async with client.stream(
            "POST",
            f"{settings.ollama_base_url}/api/generate",
            json={
                "model": model,
                "system": system,
                "prompt": prompt,
                "stream": True,
                "options": {"num_ctx": settings.llm_num_ctx},
            },
        ) as response:
            response.raise_for_status()
            final_payload: dict | None = None
            async for line in response.aiter_lines():
                if not line:
                    continue
                try:
                    payload = json.loads(line)
                except json.JSONDecodeError:
                    logger.warning("llm_generate_stream skip non-json line=%r", line[:200])
                    continue
                chunk = payload.get("response", "")
                if chunk:
                    yield chunk
                if payload.get("done"):
                    final_payload = payload
                    break

            if final_payload is not None:
                pt = final_payload.get("prompt_eval_count")
                et = final_payload.get("eval_count")
                if pt is not None and pt > 14000:
                    logger.warning(
                        "llm_generate_stream prompt_tokens=%d approaching num_ctx=%d (truncation risk)",
                        pt, settings.llm_num_ctx,
                    )
                else:
                    logger.info(
                        "llm_generate_stream model=%s num_ctx=%d prompt_tokens=%s eval_tokens=%s",
                        model, settings.llm_num_ctx, pt, et,
                    )


async def generate_answer_stream(
    question: str,
    chunks: list[Chunk],
    history: list[dict] | None = None,
    inline_attachments: list | None = None,
) -> AsyncIterator[str]:
    """Streaming counterpart of `generate_answer`. Builds the same prompt
    and yields token deltas as Ollama generates them.
    """
    context_block = build_context_block(chunks, inline_attachments=inline_attachments)
    history_block = build_history_block(history) if history else ""

    parts: list[str] = []
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

    async for delta in _llm_generate_stream(SYSTEM_PROMPT, prompt):
        yield delta
