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

ACL: when user_email is passed, this path enforces the SAME per-chunk ACL as
normal retrieval — retrieve_tables only offers a table the user can see a chunk
of, and load_tables rebuilds the table from ONLY the user's visible rows, so a
SQL aggregate can never expose data the user couldn't retrieve normally. A user
with no visible rows gets no SQL answer (falls back to RAG). Remaining nuance:
when a user can see only SOME rows of a table, the aggregate is over their
visible subset — correct (no leak), but a "total" is a total of what they can
see; answer framing for partial visibility is a future refinement.
"""

from __future__ import annotations

import asyncio
import logging
from typing import Any

from sqlalchemy.ext.asyncio import AsyncSession

from backend.acl import resolve_user_groups
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
MAX_AGENT_STEPS = 5


async def try_sql_answer(
    session: AsyncSession,
    question: str,
    conversation_id: str | None = None,
    max_distance: float = DEFAULT_MAX_DISTANCE,
    user_email: str | None = None,
) -> dict[str, Any] | None:
    """Attempt an exact SQL answer for `question`. Returns a dict with
    {context, sources, sql, table_name, result} on success, or None to fall
    back to normal retrieval.

    `user_email` enforces ACL: candidate tables are gated and the rebuilt table
    is filtered to the user's visible rows, so a SQL aggregate can never expose
    data the user couldn't retrieve normally."""
    user_groups = await resolve_user_groups(session, user_email) if user_email else []
    candidates = await retrieve_tables(
        session, question, conversation_id=conversation_id,
        top_k=CANDIDATE_TOP_K, max_distance=max_distance,
        user_email=user_email, user_groups=user_groups,
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
    tables = await load_tables(
        session, chosen["document_id"],
        user_email=user_email, user_groups=user_groups,
    )
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
        sql, result = await _run_sql_agent(conn, tables, question, schema_text)
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


async def _run_sql_agent(conn, tables: list[dict], question: str, schema_text: str):
    """Bounded ReAct tool-use loop. Each step the model picks a tool:
      - run        : execute a diagnostic SELECT,
      - find_value : locate which column(s) hold a literal (one scan, any width),
      - final      : the answer query,
      - giveup     : the tables can't answer it.
    Returns (sql, result) on success, (None, None) otherwise.

    Tools are dispatched below; adding a new one (column stats, row sampling…)
    is a new branch + a transcript entry, not a loop rewrite. Every query still
    goes through the validated read-only path; an error becomes feedback the
    model repairs from. This generalises over wrong-column/value and type-error
    failures without per-case heuristics — the model discovers the fix."""
    transcript: list[dict] = []
    challenged = False
    for _ in range(MAX_AGENT_STEPS):
        step = await sql_agent_step(question, schema_text, transcript)
        action = step["action"]

        if action == "giveup":
            return None, None

        if action == "find_value":
            value = step["value"]
            if not value:
                return None, None
            transcript.append({"value": value, "observation": _find_value(tables, value)})
            continue

        sql = step["sql"]
        if not sql:
            return None, None
        try:
            res = await asyncio.to_thread(run_select, conn, sql)
        except Exception as exc:
            transcript.append({"sql": sql, "error": str(exc)[:300]})
            continue

        if action == "final":
            # A 0 / NULL / empty result is the ambiguous case: it may be correct,
            # or the query may have filtered the wrong column/value. Don't accept
            # it blind. Challenge it once — force a verification round — and if it
            # is STILL suspicious afterwards, fall back to RAG rather than present
            # a likely-wrong "0". (Cost: a genuinely-zero answer also degrades to
            # RAG, which is safe; we never assert a 0 we couldn't verify.)
            if _is_suspicious(res):
                if challenged:
                    logger.info("sql_router: final still 0/empty after verify -> rag")
                    return None, None
                challenged = True
                transcript.append({
                    "sql": sql,
                    "result_preview": _preview(res) + (
                        "\n<- This is 0/empty. Before finalizing, VERIFY you used "
                        "the right column AND value: run a diagnostic query (e.g. "
                        'SELECT DISTINCT on the column) to confirm where the value '
                        "actually appears. If it is genuinely absent, finalize anyway."
                    ),
                })
                continue
            return sql, res

        transcript.append({"sql": sql, "result_preview": _preview(res)})
    return None, None


def _is_suspicious(result: dict) -> bool:
    """A result that warrants a verification round: zero rows, or a single
    scalar that is NULL or 0 (the classic 'filtered the wrong column' signature
    for a count/sum)."""
    rows = result["rows"]
    if not rows:
        return True
    if len(rows) == 1 and len(rows[0]) == 1:
        v = rows[0][0]
        if v is None:
            return True
        if isinstance(v, (int, float)) and not isinstance(v, bool) and v == 0:
            return True
    return False


def _find_value(tables: list[dict], value: str, max_cols: int = 8) -> str:
    """Locate which column(s) contain `value`, scanning every column of every
    table in ONE pass — cost is independent of table width, so it scales where
    column-by-column probing does not. Exact (case-insensitive) match preferred;
    falls back to partial (contains). General over any value shape — the agent
    supplies the value it reasoned about; no per-shape extraction heuristic."""
    needle = value.strip().lower()
    if not needle:
        return "no value given."
    exact: list[str] = []
    partial: list[str] = []
    for t in tables:
        for col in t["columns"]:
            name = col["name"]
            vals = [
                str(r.get(name)).strip().lower()
                for r in t["rows"]
                if r.get(name) not in (None, "")
            ]
            if any(v == needle for v in vals):
                exact.append(f'table "{t["name"]}" column "{name}"')
            elif any(needle in v for v in vals):
                partial.append(f'table "{t["name"]}" column "{name}"')
    if exact:
        return f'"{value}" is an exact value in: ' + "; ".join(exact[:max_cols])
    if partial:
        return (
            f'"{value}" is not an exact value, but appears within: '
            + "; ".join(partial[:max_cols])
        )
    return f'"{value}" was not found in any column.'


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
