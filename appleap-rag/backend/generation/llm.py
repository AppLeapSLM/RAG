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
    "You rewrite the user's latest message into a single self-contained question "
    "for an IT-operations assistant, using the conversation history.\n"
    "RULES:\n"
    "- Resolve references (it/they/that/this/these) to their explicit referents "
    "from the history.\n"
    "- KEEP THE USER'S OWN WORDING. Do NOT invent or guess table names, column "
    "names, or values, and do not rephrase into database/schema terms or write "
    "SQL — a later step maps the question to the data.\n"
    "- If the message is already self-contained, output it unchanged.\n"
    "- Output ONLY the rewritten question — no prefix, label, or explanation.\n"
)


# ── Low-level LLM call ─────────────────────────────────────────────


async def _llm_generate(
    system: str,
    prompt: str,
    model: str | None = None,
    temperature: float | None = None,
) -> str:
    """Send a prompt to the LLM and return the response text.

    This is the single point of contact with the inference backend.
    Currently uses Ollama /api/generate. When the backend changes,
    only this function needs updating.

    `temperature`: set 0.0 for structured/decision calls (routing, SQL agent)
    so they are near-deterministic — these are reasoning tasks, not creative
    writing, and default sampling makes them inconsistent run-to-run.
    """
    model = model or settings.llm_model

    options: dict = {"num_ctx": settings.llm_num_ctx}
    if temperature is not None:
        options["temperature"] = temperature

    async with httpx.AsyncClient() as client:
        response = await client.post(
            f"{settings.ollama_base_url}/api/generate",
            json={
                "model": model,
                "system": system,
                "prompt": prompt,
                "stream": False,
                "options": options,
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
    question: str,
    history: list[dict],
    attachment_filenames: list[str] | None = None,
) -> str:
    """Rewrite the latest message into a single self-contained question —
    resolving references from the history and KEEPING the user's wording (no
    invented table/column names, no SQL). Returns the rewritten question string.

    Routing is a SEPARATE, table-grounded step (decide_route); this does only the
    rewrite. Callers skip it on the cold path (no history and no attachments) —
    there is nothing to resolve."""
    attachment_filenames = attachment_filenames or []

    history_text = (
        _format_history_for_rewrite(history) if history else "(no prior turns)"
    )

    attachment_block = ""
    if attachment_filenames:
        names = "\n".join(f"- {fn}" for fn in attachment_filenames)
        attachment_block = f"\n\nFiles attached to this conversation:\n{names}"

    prompt = (
        f"Conversation history:\n{history_text}"
        f"{attachment_block}\n\n"
        f"Latest message: {question}\n\n"
        f"Rewritten question:"
    )

    raw = (await _llm_generate(REWRITE_PROMPT, prompt, temperature=0.0)).strip()
    if raw.startswith("```"):
        raw = raw.strip("`").strip()
    return raw or question


# ── Tabular-SQL routing + generation (Phase 4) ─────────────────────

ROUTE_PROMPT = (
    "You route a user's question in an IT-operations assistant. You are given the "
    "question and the data tables that most closely match it (with their "
    "columns). Choose ONE route and output ONE JSON object.\n\n"
    "ROUTES:\n"
    '- "sql": answer by computing over ONE of the listed tables — a count, sum, '
    "average, min/max, total, 'how many', or filtering/grouping over many rows. "
    "The system finds the exact columns and values itself. Give the table number.\n"
    '- "rag": answer by searching the documents — prose/text, a single-fact '
    "lookup, or when none of the listed tables hold the needed data.\n"
    '- "clarify": ask the user ONE short question — ONLY as a last resort, when '
    "no attempt could succeed without information that ONLY the user can give.\n\n"
    "PRINCIPLES:\n"
    "- Prefer acting (sql/rag) over asking; an attempt is cheap and recoverable.\n"
    "- Use sql when the answer needs a calculation over many/all rows of a listed "
    "table; use rag for prose or single-fact questions.\n"
    "- Do NOT clarify about things the system can discover itself (which table, "
    "column, or value) — that is what sql/rag do.\n"
    '- Output ONLY this JSON object: {"route": "sql|rag|clarify", "table": '
    '<number or null>, "clarification": <string or null>, "reason": "<short>"}'
)

SQL_AGENT_PROMPT = (
    "You are a data analyst answering a question by querying ONE in-memory "
    "DuckDB database (read-only). You work in steps, choosing a tool each step.\n\n"
    "Output EXACTLY ONE JSON object:\n"
    '  {"thought": "<one short sentence>", "action": '
    '"run"|"find_value"|"final"|"giveup", '
    '"sql": "<a single SELECT, for run/final>", "value": "<for find_value>"}\n\n'
    "TOOLS:\n"
    '- "run": execute a diagnostic SELECT to inspect the data before deciding '
    "(e.g. SELECT DISTINCT, or check a type/range).\n"
    '- "find_value": locate which column(s) contain a literal value from the '
    'question — put the value in the "value" field. It checks ALL columns at '
    "once, so use it (instead of guessing a column) whenever you must filter by "
    "a value and the right column is not obvious from the example values.\n"
    '- "final": the query whose result IS the answer. Use only when confident '
    "the column(s) and value(s) are correct.\n"
    '- "giveup": the available table(s) genuinely cannot answer the question.\n\n'
    "RULES:\n"
    "- Before filtering or grouping by a specific value from the question, make "
    "sure you know which column holds it — use find_value if it is not obvious.\n"
    "- One single SELECT (a leading WITH is fine). Quote identifiers with double "
    "quotes (columns may contain spaces). No INSERT/UPDATE/DELETE/DDL, no "
    "semicolons, no file/external functions.\n"
    "- Output ONLY the JSON object."
)


def _extract_json(text: str) -> dict | None:
    """Pull the JSON object out of a model response that may include reasoning
    before/after it. Returns None if nothing parses."""
    try:
        return json.loads(text)
    except (ValueError, TypeError):
        pass
    start, end = text.find("{"), text.rfind("}")
    if start != -1 and end > start:
        try:
            return json.loads(text[start:end + 1])
        except (ValueError, TypeError):
            return None
    return None


def _strip_sql(text: str) -> str:
    """Strip code fences / a leading 'SQL:' label from a generated SQL string."""
    t = text.strip()
    if t.startswith("```"):
        t = t.strip("`")
        # drop an optional language tag on the first line (```sql)
        if "\n" in t:
            first, rest = t.split("\n", 1)
            if first.strip().lower() in ("sql", "duckdb"):
                t = rest
    if t.lower().startswith("sql:"):
        t = t[4:]
    return t.strip()


async def decide_route(
    question: str,
    candidates: list[dict],
    prev_was_clarification: bool = False,
) -> dict:
    """Grounded 3-way route decision, informed by the candidate tables. Returns
    {"route": "sql"|"rag"|"clarify", "table": <1-based index|None>,
    "clarification": <str|None>, "reason": <str>}.

    Safety: degrades to "rag" on any parse problem; a "sql" route needs a valid
    table index or it falls to "rag"; and a "clarify" is suppressed if we already
    clarified last turn (loop-breaker — the user can always escape by replying)."""
    if candidates:
        lines = []
        for i, c in enumerate(candidates, 1):
            cols = ", ".join(
                f'{col.get("name")} ({col.get("type", "text")})'
                for col in (c.get("columns") or [])
            )
            lines.append(f'[{i}] "{c.get("table_name")}" — {c.get("description","")} Columns: {cols}')
        tables_block = "Matching data tables:\n" + "\n".join(lines)
    else:
        tables_block = "Matching data tables: (none)"

    prompt = f"Question: {question}\n\n{tables_block}\n\nDecide the route. Output the JSON object."
    raw = (await _llm_generate(ROUTE_PROMPT, prompt, temperature=0.0)).strip()
    data = _extract_json(raw) or {}

    route = str(data.get("route", "")).lower().strip()
    if route not in ("sql", "rag", "clarify"):
        route = "rag"

    if route == "clarify" and prev_was_clarification:
        logger.info("decide_route CLARIFY suppressed (prev turn was a clarification) — using rag")
        route = "rag"

    table = data.get("table") if isinstance(data.get("table"), int) else None
    if route == "sql" and not (table and candidates and 1 <= table <= len(candidates)):
        logger.info("decide_route sql without a valid table -> rag")
        route, table = "rag", None

    clarification = str(data.get("clarification") or "") if route == "clarify" else None
    if route == "clarify" and not clarification:
        route = "rag"  # wanted to clarify but gave no question — attempt instead

    logger.info("decide_route route=%s table=%s reason=%s", route, table, str(data.get("reason", ""))[:120])
    return {"route": route, "table": table, "clarification": clarification, "reason": str(data.get("reason", ""))}


async def sql_agent_step(
    question: str, schema_text: str, transcript: list[dict]
) -> dict:
    """One step of the SQL agent loop. Given the schema, the question, and the
    transcript of prior (query → result/error) steps, decide the next action.
    Returns {"action": "run"|"final"|"giveup", "sql": str, "thought": str}.
    Degrades to giveup on parse failure — caller falls back to RAG."""
    parts = [f"Database schema (DuckDB):\n{schema_text}", f"\nQuestion: {question}"]
    if transcript:
        parts.append("\nSteps so far:")
        for i, step in enumerate(transcript, 1):
            if "observation" in step:  # a tool result (e.g. find_value)
                parts.append(f'[{i}] find_value("{step["value"]}") -> {step["observation"]}')
            elif "error" in step:
                parts.append(f"[{i}] SQL: {step['sql']}\n    ERROR: {step['error']}")
            else:
                parts.append(f"[{i}] SQL: {step['sql']}\n    RESULT:\n{step['result_preview']}")
    parts.append("\nYour next step (one JSON object):")

    raw = await _llm_generate(SQL_AGENT_PROMPT, "\n".join(parts), temperature=0.0)
    data = _extract_json(raw) or {}
    action = str(data.get("action", "")).lower().strip()
    if action not in ("run", "find_value", "final", "giveup"):
        action = "giveup"
    sql = _strip_sql(str(data.get("sql", "") or ""))
    value = str(data.get("value", "") or "").strip()
    logger.info("sql_agent_step action=%s sql=%s value=%s", action, sql[:140], value[:60])
    return {
        "action": action, "sql": sql, "value": value,
        "thought": str(data.get("thought", "")),
    }


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
    precomputed_context: str | None = None,
) -> AsyncIterator[str]:
    """Streaming counterpart of `generate_answer`. Builds the same prompt
    and yields token deltas as Ollama generates them.

    `precomputed_context` (tabular-SQL path): when provided, it is used as the
    context block verbatim instead of building one from `chunks` — so the model
    answers from the exact computed SQL result rather than retrieved fragments.
    """
    if precomputed_context is not None:
        context_block = precomputed_context
    else:
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
