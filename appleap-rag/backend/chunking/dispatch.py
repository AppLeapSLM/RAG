"""Single entry point for ingesting a file.

Routes each file by its inferred Format:

    PROSE       → Unstructured.io + recursive character splitter (existing path)
    CSV         → one chunk per row, pipe-formatted key: value pairs (no tree-sitter)
    everything  → tree-sitter (language-specific grammar) + deterministic NL header

Semantic doc_type is attached to every chunk's metadata for later
retrieval-time filtering and boosting.
"""

from __future__ import annotations

import asyncio
import csv
import io
import logging
from pathlib import Path
from typing import Any

from backend.chunking.chunker import chunk_parsed_document_async
from backend.chunking.code_chunker import chunk_code
from backend.chunking.doc_type import DocType, Format, classify

logger = logging.getLogger(__name__)


async def process_file(
    file_path: str | Path,
    extra_metadata: dict[str, Any] | None = None,
    display_name: str | None = None,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Parse + chunk a file. Returns (chunks, doc_level_metadata)."""
    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")

    logical_path = Path(display_name) if display_name else path

    meta = dict(extra_metadata or {})
    hints = [
        meta.get("source", ""),
        meta.get("category", ""),
        meta.get("folder_path", ""),
    ]
    classification = classify(logical_path, hints=hints)
    meta["format"] = classification.format.value
    meta["doc_type"] = classification.doc_type.value

    logger.info(
        "Processing %s → format=%s doc_type=%s",
        logical_path.name, classification.format.value, classification.doc_type.value,
    )

    if classification.format == Format.PROSE:
        return await _process_prose(path, meta, display_name=logical_path.name)

    if classification.format == Format.CSV:
        return await asyncio.to_thread(_process_csv, path, meta, logical_path.name)

    return await asyncio.to_thread(
        _process_structured, path, classification.format, meta, logical_path.name
    )


# ── Prose path (existing Unstructured + recursive char splitter) ──────


async def _process_prose(
    path: Path,
    meta: dict[str, Any],
    display_name: str,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    from backend.parsing.parser import parse_file
    parsed_doc = await asyncio.to_thread(parse_file, str(path), meta)
    parsed_doc.filename = display_name
    chunks = await chunk_parsed_document_async(parsed_doc)
    doc_meta = {
        "filetype": parsed_doc.filetype,
        "original_filename": display_name,
        "num_elements": len(parsed_doc.elements),
        "format": Format.PROSE.value,
        "doc_type": meta.get("doc_type", DocType.GENERIC.value),
    }
    return chunks, doc_meta


# ── Structured path (tree-sitter) ─────────────────────────────────────


def _process_structured(
    path: Path,
    fmt: Format,
    meta: dict[str, Any],
    display_name: str,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    source = path.read_text(encoding="utf-8", errors="replace")
    chunks = chunk_code(
        source=source,
        fmt=fmt,
        filename=display_name,
        metadata=meta,
    )
    doc_meta = {
        "filetype": _mime_for_format(fmt),
        "original_filename": display_name,
        "num_elements": len(chunks),
        "format": fmt.value,
        "doc_type": meta.get("doc_type", DocType.GENERIC.value),
    }
    return chunks, doc_meta


# ── CSV path (direct pipe-format, no tree-sitter) ─────────────────────


def _process_csv(
    path: Path,
    meta: dict[str, Any],
    display_name: str,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Parse a CSV into one chunk per row, each row rendered as
    "col1: value1 | col2: value2 | ...".

    Rationale: JSON-formatted CSV row chunks were poorly aligned with Nomic's
    NL training distribution (keys and braces drowned out the semantic
    content), which hurt CMDB retrieval recall. Pipe-formatted `key: value`
    pairs read more like natural language and embed closer to NL queries.
    Tree-sitter is no longer involved in the CSV path.

    Defensive steps for real-world CSVs:
      - Strip UTF-8 BOM if present.
      - Use csv.DictReader which handles quoted fields + embedded newlines.
      - Sanitize `\\n` and `\\r` out of every cell value so each row stays a
        single logical line of text.
      - Drop empty cells per row to keep semantic density.
    """
    raw = path.read_text(encoding="utf-8", errors="replace")
    if raw.startswith("\ufeff"):  # strip BOM
        raw = raw[1:]

    rows = _csv_to_row_objects(raw)
    if not rows:
        return [], _csv_doc_meta(display_name, meta, num_elements=0)

    chunks: list[dict[str, Any]] = []
    bracket = _csv_bracket_header(display_name, meta)
    for i, row in enumerate(rows):
        pipe_content = _row_to_pipe(row)
        if not pipe_content:
            continue
        chunks.append({
            "text": f"{bracket}{pipe_content}",
            "metadata": {
                **meta,
                "csv_row_index": i,
                "total_rows": len(rows),
            },
            "element_types": ["csv_row"],
        })

    return chunks, _csv_doc_meta(display_name, meta, num_elements=len(chunks))


def _csv_to_row_objects(raw: str) -> list[dict[str, str]]:
    """Parse CSV into a list of dicts. Sanitizes embedded newlines in cell
    values and drops empty cells. Preserves column order from DictReader."""
    reader = csv.DictReader(io.StringIO(raw))
    if reader.fieldnames is None:
        return []
    out: list[dict[str, str]] = []
    for row in reader:
        cleaned: dict[str, str] = {}
        for k, v in row.items():
            if k is None or v is None:
                continue
            key = _sanitize_cell(str(k))
            val = _sanitize_cell(str(v))
            if key == "" or val == "":
                continue
            cleaned[key] = val
        if cleaned:
            out.append(cleaned)
    return out


def _sanitize_cell(value: str) -> str:
    """Force a single CSV cell value to a single logical line of text.

    Removes embedded newlines and carriage returns (which otherwise break
    the one-chunk-per-row invariant for the pipe format). Strips surrounding
    whitespace. Does NOT drop pipe characters — if a customer's cell value
    legitimately contains `|`, we accept minor ambiguity in the output string
    rather than destroying the content.
    """
    return value.replace("\n", " ").replace("\r", " ").strip()


def _row_to_pipe(row: dict[str, str]) -> str:
    """Render a row dict as 'col1: val1 | col2: val2 | ...'. Order preserved
    from DictReader insertion order (Python ≥3.7 dicts are ordered)."""
    return " | ".join(f"{k}: {v}" for k, v in row.items())


def _csv_bracket_header(display_name: str, meta: dict[str, Any]) -> str:
    """Citation-metadata bracket for a CSV row chunk. No 'Row X of Y' — per
    production review, the row index carries no semantic signal for the LLM
    and wastes tokens."""
    parts = [f"File: {display_name}"]
    source = meta.get("source", "")
    if source and source not in ("manual", "upload", ""):
        parts.append(f"Source: {source}")
    folder = meta.get("folder_path", "")
    if folder and folder != "/":
        parts.append(f"Path: {folder}")
    return "[" + " | ".join(parts) + "]\n"


def _csv_doc_meta(
    display_name: str,
    meta: dict[str, Any],
    num_elements: int,
) -> dict[str, Any]:
    return {
        "filetype": "text/csv",
        "original_filename": display_name,
        "num_elements": num_elements,
        "format": Format.CSV.value,
        "doc_type": meta.get("doc_type", DocType.GENERIC.value),
    }


# ── Helpers ───────────────────────────────────────────────────────────


def _mime_for_format(fmt: Format) -> str:
    return {
        Format.TERRAFORM: "text/x-terraform",
        Format.YAML: "application/x-yaml",
        Format.JSON: "application/json",
        Format.PUPPET: "text/x-puppet",
        Format.PYTHON: "text/x-python",
        Format.GO: "text/x-go",
        Format.RUBY: "text/x-ruby",
        Format.JAVASCRIPT: "application/javascript",
        Format.TYPESCRIPT: "application/typescript",
        Format.BASH: "application/x-sh",
        Format.DOCKERFILE: "text/x-dockerfile",
        Format.CSV: "text/csv",
        Format.PROSE: "text/plain",
    }.get(fmt, "application/octet-stream")
