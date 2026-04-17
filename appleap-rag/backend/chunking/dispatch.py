"""Single entry point for ingesting a file.

Routes each file by its inferred Format:

    PROSE       → Unstructured.io + recursive character splitter (existing path)
    CSV         → rows → JSON array → tree-sitter (json grammar)
    everything  → tree-sitter (language-specific grammar)

Semantic doc_type is attached to every chunk's metadata for later
retrieval-time filtering and boosting.
"""

from __future__ import annotations

import asyncio
import csv
import io
import json
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
    """Parse + chunk a file. Returns (chunks, doc_level_metadata).

    `display_name` is the logical filename used for classification and header
    breadcrumbs; if omitted, `file_path.name` is used. Callers with tmp files
    (uploads) or connector-downloaded files should pass the original name so
    doc_type inference sees the real path hints.

    `doc_level_metadata` is the fields that should go on the Document row
    (filetype, num_elements, original_filename, format, doc_type).
    """
    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")

    # Use display_name for classification so tmp paths don't confuse path-hint rules.
    logical_path = Path(display_name) if display_name else path

    meta = dict(extra_metadata or {})
    # Gather hint strings so classify() can fall back on source/category when
    # the on-disk path itself lacks folder context.
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
    # Lazy import — Unstructured is a heavy optional dep, only needed for prose.
    from backend.parsing.parser import parse_file
    parsed_doc = await asyncio.to_thread(parse_file, str(path), meta)
    # Override the parsed filename with the logical display name (e.g. the
    # original upload name rather than a tmp suffix).
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


# ── CSV path (csv → json → tree-sitter json) ──────────────────────────


def _process_csv(
    path: Path,
    meta: dict[str, Any],
    display_name: str,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Convert CSV rows to a JSON array of row-objects, drop empty cells,
    then hand the JSON to the tree-sitter json chunker.

    This keeps the two-tool architecture (Unstructured + tree-sitter) while
    emitting one chunk per row, which is the right unit for CMDB-style data.
    """
    raw = path.read_text(encoding="utf-8", errors="replace")
    rows = _csv_to_row_objects(raw)
    if not rows:
        return [], {
            "filetype": "text/csv",
            "original_filename": display_name,
            "num_elements": 0,
            "format": Format.CSV.value,
            "doc_type": meta.get("doc_type", DocType.GENERIC.value),
        }

    json_source = json.dumps(rows, indent=2, ensure_ascii=False)
    chunks = chunk_code(
        source=json_source,
        fmt=Format.JSON,
        filename=display_name,
        metadata={**meta, "format": Format.CSV.value},  # preserve original format label
    )

    doc_meta = {
        "filetype": "text/csv",
        "original_filename": display_name,
        "num_elements": len(rows),
        "format": Format.CSV.value,
        "doc_type": meta.get("doc_type", DocType.GENERIC.value),
    }
    return chunks, doc_meta


def _csv_to_row_objects(raw: str) -> list[dict[str, str]]:
    """Parse CSV into a list of dicts. Drops empty-string values so each row's
    JSON stays semantically dense."""
    reader = csv.DictReader(io.StringIO(raw))
    if reader.fieldnames is None:
        return []
    out: list[dict[str, str]] = []
    for row in reader:
        filtered = {
            k: v for k, v in row.items()
            if k is not None and v is not None and str(v).strip() != ""
        }
        if filtered:
            out.append(filtered)
    return out


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
