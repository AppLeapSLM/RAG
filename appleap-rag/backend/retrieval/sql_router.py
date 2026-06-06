"""Tabular-SQL router — the Phase-4 orchestrator that turns an aggregation
question into an exact answer.

Flow (every step falls back to None → normal retrieval on any doubt/failure):
  1. retrieve_tables  — top-K candidate tables for the question (evidence).
  2. decide_sql_route — grounded decision: does this need a computation over one
     of those tables, and which? (LLM, structured output.)
  3. load_tables      — rebuild the chosen document's table(s) from row_data.
  4. SQL agent loop   — the model writes SQL, runs it read-only in in-memory
                        DuckDB, and on an error / wrong-looking result issues
                        diagnostic queries to inspect the data and repairs,
                        bounded to MAX_AGENT_STEPS, before a final answer.
  5. format           — render the exact result as a context block for the
                        generator to phrase the final answer around.

Returning None means "I did not produce a confident SQL answer" — the caller
runs normal RAG. This is the safe default: a missed aggregation degrades to
today's behavior, never to a crash.

SECURITY NOTE (known gap, must fix before multi-tenant): retrieve_tables scopes
to corpus + this conversation's attachments but does NOT yet apply per-document
ACL. The chunk-retrieval path does (acl_filter). For a single-tenant deployment
this matches today's reality; before multi-tenant, the catalog needs ACL.
"""

from __future__ import annotations

import asyncio
import logging
from typing import Any

from sqlalchemy.ext.asyncio import AsyncSession

from backend.generation.llm import decide_sql_route, sql_agent_step
from backend.retrieval.table_catalog import retrieve_tables
from backend.retrieval.tabular_sql import (
    build_connection,
    describe_tables_for_sql,
    load_tables,
    run_select,
)

logger = logging.getLogger(__name__)

# Cosine-distance ceiling for a catalog hit to be considered relevant enough to
# even run the (LLM) routing decision. Loose on purpose — the grounded decision
# is the real filter; this just avoids a decision call when nothing is close.
# Tune against the eval (distances are logged).
DEFAULT_MAX_DISTANCE = 0.6
CANDIDATE_TOP_K = 5
# Max LLM steps in the SQL agent loop (diagnostics + final). Bounds latency/cost
# while leaving room to inspect the data and repair one or two wrong attempts.
MAX_AGENT_STEPS = 4


async def try_sql_answer(
    session: AsyncSession,
    question: str,
    conversation_id: str | None = None,
    max_distance: float = DEFAULT_MAX_DISTANCE,
) -> dict[str, Any] | None:
    """Attempt an exact SQL answer for `question`. Returns a dict with
    {context, sources, sql, table_name, result} on success, or None to fall
    back to normal retrieval."""
    candidates = await retrieve_tables(
        session, question, conversation_id=conversation_id,
        top_k=CANDIDATE_TOP_K, max_distance=max_distance,
    )
    if candidates:
        logger.info(
            "sql_router: %d candidate table(s), nearest dist=%.3f",
            len(candidates), candidates[0]["distance"] or -1,
        )
    if not candidates:
        return None

    decision = await decide_sql_route(question, candidates)
    if decision["route"] != "sql":
        logger.info("sql_router: route=rag reason=%s", decision.get("reason"))
        return None

    chosen = candidates[decision["table"] - 1]
    tables = await load_tables(session, chosen["document_id"])
    if not tables:
        # Catalog had it but row_data is missing (pre-Phase-1 doc not re-ingested).
        logger.info("sql_router: chosen doc %s not rebuildable -> rag", chosen["document_id"])
        return None

    schema_text = describe_tables_for_sql(tables)
    try:
        conn = await asyncio.to_thread(build_connection, tables)
    except Exception as exc:
        logger.warning("sql_router: build_connection failed (%s) -> rag", exc)
        return None
    try:
        sql, result = await _run_sql_agent(conn, question, schema_text)
    finally:
        conn.close()

    if sql is None or not result or not result["rows"]:
        logger.info("sql_router: agent produced no usable answer -> rag")
        return None

    logger.info(
        "sql_router: SQL answered table=%s rows=%d", chosen["table_name"], result["row_count"]
    )
    return {
        "context": _format_context(chosen, sql, result),
        "sources": [{
            "chunk_id": None,
            "document_id": chosen["document_id"],
            "content_preview": (
                f'[Computed over table "{chosen["table_name"]}"] {chosen["description"]}'
            )[:200],
            "chunk_index": 0,
        }],
        "sql": sql,
        "table_name": chosen["table_name"],
        "result": result,
    }


async def _run_sql_agent(conn, question: str, schema_text: str):
    """Bounded ReAct loop. The model issues diagnostic "run" queries to inspect
    the data (e.g. find which column holds a value) and a "final" query whose
    result is the answer. Returns (sql, result) on success, (None, None) if it
    gives up or never finalizes.

    Every query goes through the same validated, read-only execution path; a
    query that errors becomes feedback the model repairs from on the next step.
    This generalises over wrong-column, wrong-value, and type-error failures
    without per-case heuristics — the model discovers the fix by querying."""
    transcript: list[dict] = []
    for _ in range(MAX_AGENT_STEPS):
        step = await sql_agent_step(question, schema_text, transcript)
        action, sql = step["action"], step["sql"]
        if action == "giveup" or not sql:
            return None, None
        try:
            res = await asyncio.to_thread(run_select, conn, sql)
        except Exception as exc:
            transcript.append({"sql": sql, "error": str(exc)[:300]})
            continue
        if action == "final":
            return sql, res
        transcript.append({"sql": sql, "result_preview": _preview(res)})
    return None, None


def _preview(result: dict, max_rows: int = 15, max_cell: int = 60) -> str:
    """Compact rendering of a query result for the agent transcript (bounded so
    a diagnostic over a big column can't blow up the prompt)."""
    cols = result["columns"]
    rows = result["rows"][:max_rows]
    lines = [" | ".join(str(c) for c in cols)]
    for r in rows:
        lines.append(
            " | ".join((str(v)[:max_cell] if v is not None else "") for v in r)
        )
    extra = result["row_count"] - len(rows)
    if extra > 0:
        lines.append(f"... (+{extra} more rows)")
    return "\n".join(lines)


def _format_context(chosen: dict, sql: str, result: dict) -> str:
    """Render the exact SQL result as the context block the generator answers
    from. The framing tells the model this is a complete, computed result over
    the whole table (not a sample) so it stops presenting partial sums."""
    cols = result["columns"]
    rows = result["rows"]
    header = " | ".join(str(c) for c in cols)
    body = "\n".join(
        " | ".join("" if v is None else str(v) for v in r) for r in rows
    )
    truncated = "\n(results truncated to the row cap)" if result.get("truncated") else ""
    return (
        f'The following is the EXACT result of a database query computed over the '
        f'COMPLETE table "{chosen["table_name"]}" ({chosen["description"]}). '
        f"It is a full computation over every row, not a sample.\n\n"
        f"SQL executed:\n{sql}\n\n"
        f"Result ({result['row_count']} row(s)):\n{header}\n{body}{truncated}"
    )
