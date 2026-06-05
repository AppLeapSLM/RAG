"""Tabular SQL engine — exact answers for aggregation questions over tabular
data (csv/tsv/xlsx/xls), the part RAG fundamentally cannot do.

RAG retrieves relevant row *fragments*; it does not *enumerate*. "Total profit"
over a 575-row sheet is unanswerable by top-k retrieval. This module rebuilds
the table the user is asking about into an in-memory DuckDB instance and runs a
SELECT against it, so SUM/COUNT/AVG/GROUP BY return exact values.

Source of truth: the structured `row_data` persisted on every tabular chunk and
the `table_schema` persisted on the document (Phase 1, ingest). The original
file is NOT retained after ingest, and reverse-parsing the pipe-format chunk
text is ambiguous (cell values can contain '|' or ':'), so we rebuild from the
structured copy.

Safety model (this is a fully on-prem, air-gapped product — generated SQL must
never reach the filesystem or network):
  - DuckDB opened in-memory with `enable_external_access=false` → blocks
    read_csv/COPY/ATTACH/INSTALL/LOAD/httpfs and every other file/network path.
  - Only a single SELECT / WITH statement is accepted (validated).
  - Statement watchdog (conn.interrupt) bounds runaway queries.
  - Results are row-capped.
  - The connection is in-memory and discarded after the query — zero persistence.

This module has NO query-path wiring yet; Phases 3-4 add routing + SQL
generation and call build_connection()/run_select(). load_tables() is the only
DB-coupled function; build_connection/run_select/validate are pure and unit-
testable without a database.
"""

from __future__ import annotations

import logging
import re
import threading
from typing import Any

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from backend.db.models import Chunk, Document

logger = logging.getLogger(__name__)

# Inferred-type (Phase 1 schema) → DuckDB column type.
_DUCKDB_TYPE = {
    "integer": "BIGINT",
    "double": "DOUBLE",
    "date": "DATE",
    "timestamp": "TIMESTAMP",
    "text": "VARCHAR",
}

DEFAULT_ROW_CAP = 1000
DEFAULT_TIMEOUT_S = 10.0

# External-access / config-changing keywords that must never appear even though
# enable_external_access=false already neutralises them. Scanned as whole words
# against a copy with string literals stripped (so a quoted value can't trip it).
_FORBIDDEN = re.compile(
    r"\b(attach|detach|copy|install|load|pragma|export|import|set|call|"
    r"insert|update|delete|drop|alter|create|replace)\b",
    re.IGNORECASE,
)
_STRING_LITERAL = re.compile(r"'(?:[^']|'')*'")


# ── Table spec (pure data; see load_tables for the DB-backed builder) ────
# A "table" is one CSV/TSV file or one Excel sheet:
#   {"name": str, "columns": [{"name": str, "type": str}], "rows": [dict]}


def describe_tables(tables: list[dict[str, Any]]) -> str:
    """Compact schema text for the SQL-generation prompt (Phase 4). One line
    per table: name + quoted columns with their types."""
    lines: list[str] = []
    for t in tables:
        cols = ", ".join(
            f'"{c["name"]}" {_DUCKDB_TYPE.get(c.get("type", "text"), "VARCHAR")}'
            for c in t["columns"]
        )
        lines.append(f'Table "{t["name"]}" ({t.get("row_count", len(t.get("rows", [])))} rows): {cols}')
    return "\n".join(lines)


def build_connection(tables: list[dict[str, Any]]):
    """Build an in-memory DuckDB connection holding every table, correctly
    typed. Values that fail their column's cast become NULL (TRY_CAST) rather
    than aborting the load — real customer data is messy. Returns the open
    connection (caller closes it). Raises ValueError if there's nothing to build.
    """
    import duckdb  # lazy — keeps the app importable if duckdb isn't installed

    if not tables:
        raise ValueError("no tables to build")

    conn = duckdb.connect(database=":memory:")
    # Lock down BEFORE loading any data. Once false, external access cannot be
    # re-enabled in this connection — a one-way latch.
    conn.execute("SET enable_external_access=false")

    for t in tables:
        cols = t["columns"]
        if not cols:
            continue
        col_names = [c["name"] for c in cols]
        staging = f'{t["name"]}__staging'

        staging_cols = ", ".join(f"{_q(c)} VARCHAR" for c in col_names)
        conn.execute(f"CREATE TABLE {_q(staging)} ({staging_cols})")

        placeholders = ", ".join("?" for _ in col_names)
        data = [
            tuple(_as_str(row.get(c)) for c in col_names) for row in t["rows"]
        ]
        if data:
            conn.executemany(
                f"INSERT INTO {_q(staging)} VALUES ({placeholders})", data
            )

        typed_cols = ", ".join(
            f'TRY_CAST({_q(c["name"])} AS {_DUCKDB_TYPE.get(c.get("type", "text"), "VARCHAR")}) AS {_q(c["name"])}'
            for c in cols
        )
        conn.execute(
            f"CREATE TABLE {_q(t['name'])} AS SELECT {typed_cols} FROM {_q(staging)}"
        )
        conn.execute(f"DROP TABLE {_q(staging)}")

    return conn


def run_select(
    conn,
    sql: str,
    row_cap: int = DEFAULT_ROW_CAP,
    timeout_s: float = DEFAULT_TIMEOUT_S,
) -> dict[str, Any]:
    """Validate + execute a single read-only SELECT. Returns
    {columns, rows, row_count, truncated}. Raises ValueError on an unsafe/
    malformed statement, TimeoutError if the watchdog fires."""
    import duckdb

    clean = validate_select(sql)
    interrupt_exc = getattr(duckdb, "InterruptException", None)

    # Watchdog: interrupt the query if it overruns. DuckDB raises on interrupt.
    timer = threading.Timer(timeout_s, conn.interrupt)
    timer.start()
    try:
        cur = conn.execute(clean)
        fetched = cur.fetchmany(row_cap + 1)
        columns = [d[0] for d in cur.description] if cur.description else []
    except Exception as exc:
        if interrupt_exc is not None and isinstance(exc, interrupt_exc):
            raise TimeoutError(
                f"SQL exceeded {timeout_s}s and was interrupted"
            ) from exc
        raise
    finally:
        timer.cancel()

    truncated = len(fetched) > row_cap
    rows = [list(r) for r in fetched[:row_cap]]
    return {
        "columns": columns,
        "rows": rows,
        "row_count": len(rows),
        "truncated": truncated,
    }


def validate_select(sql: str) -> str:
    """Return a cleaned single-statement SELECT/WITH, or raise ValueError.

    Primary gate is structural (single statement, starts with SELECT/WITH) and
    the connection's disabled external access; the keyword scan is defence in
    depth against config/DDL/DML smuggled past a leading CTE."""
    clean = (sql or "").strip().rstrip(";").strip()
    if not clean:
        raise ValueError("empty SQL")
    if ";" in clean:
        raise ValueError("multiple statements are not allowed")

    head = clean.lstrip("(").lstrip().upper()
    if not (head.startswith("SELECT") or head.startswith("WITH")):
        raise ValueError("only SELECT / WITH queries are allowed")

    scan_target = _STRING_LITERAL.sub("''", clean)
    if head.startswith("WITH"):
        # A leading CTE can legally precede DML in some dialects; reject any
        # mutating/external keyword outside string literals.
        bad = _FORBIDDEN.search(scan_target)
        if bad:
            raise ValueError(f"disallowed keyword in query: {bad.group(0)}")
    else:
        # Plain SELECT: only the external-access verbs are a real concern.
        for kw in ("attach", "copy", "install", "load", "pragma", "export", "import"):
            if re.search(rf"\b{kw}\b", scan_target, re.IGNORECASE):
                raise ValueError(f"disallowed keyword in query: {kw}")
    return clean


# ── DB-backed table loader (Phase 4 entry point) ─────────────────────────


async def load_tables(
    session: AsyncSession, document_id: str
) -> list[dict[str, Any]]:
    """Rebuild a document's table(s) from its persisted chunks. Returns a list
    of table specs (one per Excel sheet, or one for a CSV/TSV). Empty list if
    the document isn't tabular or predates Phase 1 (no row_data) — the caller
    then falls back to normal retrieval."""
    doc = await session.get(Document, document_id)
    if doc is None:
        return []

    schema = (doc.metadata_ or {}).get("table_schema") or {}
    schema_entries = schema.get("tables") or []

    res = await session.execute(
        select(Chunk)
        .where(Chunk.document_id == document_id)
        .order_by(Chunk.chunk_index)
    )
    rows_by_sheet: dict[Any, list[dict]] = {}
    for ch in res.scalars().all():
        md = ch.metadata_ or {}
        row_data = md.get("row_data")
        if not row_data:
            continue
        rows_by_sheet.setdefault(md.get("sheet_name"), []).append(row_data)

    if not rows_by_sheet:
        return []

    tables: list[dict[str, Any]] = []
    used: set[str] = set()
    for entry in schema_entries:
        sheet = entry.get("sheet_name")
        rows = rows_by_sheet.get(sheet)
        if not rows:
            continue
        tables.append({
            "name": _table_name(sheet, used),
            "columns": entry.get("columns") or _text_columns(rows),
            "rows": rows,
            "row_count": len(rows),
            "sheet_name": sheet,
        })

    # Defensive: chunks have row_data but the schema didn't line up (shouldn't
    # happen for Phase-1 ingests) — synthesise an all-text schema so the data is
    # still queryable rather than silently dropped.
    if not tables:
        for sheet, rows in rows_by_sheet.items():
            tables.append({
                "name": _table_name(sheet, used),
                "columns": _text_columns(rows),
                "rows": rows,
                "row_count": len(rows),
                "sheet_name": sheet,
            })

    return tables


# ── Helpers ──────────────────────────────────────────────────────────────


def _q(ident: str) -> str:
    """Quote a SQL identifier, escaping embedded double-quotes."""
    return '"' + str(ident).replace('"', '""') + '"'


def _as_str(value: Any) -> Any:
    """row_data values are already strings; coerce defensively, keep None."""
    return None if value is None else str(value)


def _table_name(sheet_name: str | None, used: set[str]) -> str:
    """A readable, unique table name. CSV/TSV (no sheet) → 'data'; Excel → the
    sanitised sheet name. Deduped against names already issued."""
    base = _sanitize_ident(sheet_name) if sheet_name else "data"
    name = base or "data"
    i = 2
    while name in used:
        name = f"{base}_{i}"
        i += 1
    used.add(name)
    return name


def _sanitize_ident(raw: str) -> str:
    """Lower-case, non-alnum → underscore, collapse repeats, trim. Leading digit
    gets a 't_' prefix. (Identifiers are always quoted anyway; this is purely so
    the SQL-generation prompt sees a clean, referenceable name.)"""
    s = re.sub(r"[^0-9a-zA-Z]+", "_", str(raw).strip().lower()).strip("_")
    if not s:
        return ""
    if s[0].isdigit():
        s = f"t_{s}"
    return s


def _text_columns(rows: list[dict]) -> list[dict[str, str]]:
    """Union of keys across rows (first-appearance order), all typed text."""
    order: list[str] = []
    seen: set[str] = set()
    for row in rows:
        for k in row:
            if k not in seen:
                seen.add(k)
                order.append(k)
    return [{"name": c, "type": "text"} for c in order]
