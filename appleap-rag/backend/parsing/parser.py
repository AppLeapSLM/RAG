from __future__ import annotations

import logging
import re
import subprocess
from pathlib import Path
from typing import Any

from unstructured.partition.auto import partition

from backend.config import settings
from backend.parsing.base import ElementType, ParsedDocument, ParsedElement

logger = logging.getLogger(__name__)


def _pdftotext_fallback(path: Path) -> ParsedDocument | None:
    """When Unstructured can't extract any usable text from a PDF (image-heavy
    slide decks, scanned-but-not-OCR'd pages, anything Unstructured's heuristics
    bail on), try Poppler's `pdftotext` as a last resort.

    Returns a ParsedDocument with the extracted text as a single NarrativeText
    element, or None if pdftotext is missing, times out, exits non-zero, or
    produces no text. Callers fall through to raising the original "no content"
    error in the None case.
    """
    if path.suffix.lower() != ".pdf":
        return None
    try:
        result = subprocess.run(
            ["pdftotext", "-layout", "-q", str(path), "-"],
            capture_output=True, text=True, timeout=120, check=False,
        )
    except (FileNotFoundError, subprocess.TimeoutExpired) as exc:
        logger.warning("pdftotext fallback unavailable/slow on %s: %s", path.name, exc)
        return None
    if result.returncode != 0:
        logger.warning(
            "pdftotext exit=%d on %s: %s",
            result.returncode, path.name, (result.stderr or "")[:200],
        )
        return None
    text = result.stdout.strip()
    if not text:
        return None
    logger.info("pdftotext fallback extracted %d chars from %s", len(text), path.name)
    return ParsedDocument(
        filename=path.name,
        filetype="application/pdf",
        elements=[
            ParsedElement(
                text=text,
                element_type=ElementType.NARRATIVE_TEXT,
                metadata={},
            ),
        ],
        metadata={},
    )


def _html_table_to_markdown(html: str) -> str:
    """Convert an HTML table to Markdown table format.

    Produces clean Markdown that LLMs can read:
      | CPU    | 8 cores |
      | Memory | 32GB    |
    """
    # Extract rows
    rows: list[list[str]] = []
    for tr_match in re.finditer(r"<tr[^>]*>(.*?)</tr>", html, re.DOTALL | re.IGNORECASE):
        row_html = tr_match.group(1)
        cells = re.findall(
            r"<(?:td|th)[^>]*>(.*?)</(?:td|th)>", row_html, re.DOTALL | re.IGNORECASE
        )
        # Strip HTML tags within cells
        cleaned = [re.sub(r"<[^>]+>", "", c).strip() for c in cells]
        if any(cleaned):
            rows.append(cleaned)

    if not rows:
        return ""

    # Normalize column count
    max_cols = max(len(r) for r in rows)
    for row in rows:
        while len(row) < max_cols:
            row.append("")

    # Build Markdown table
    lines: list[str] = []
    for i, row in enumerate(rows):
        lines.append("| " + " | ".join(row) + " |")
        if i == 0:
            lines.append("| " + " | ".join("---" for _ in row) + " |")

    return "\n".join(lines)

_TYPE_MAP: dict[str, ElementType] = {
    "Title": ElementType.TITLE,
    "NarrativeText": ElementType.NARRATIVE_TEXT,
    "Table": ElementType.TABLE,
    "ListItem": ElementType.LIST_ITEM,
    "Image": ElementType.IMAGE,
    "Header": ElementType.HEADER,
    "Footer": ElementType.FOOTER,
    "PageBreak": ElementType.PAGE_BREAK,
    "Formula": ElementType.FORMULA,
}


def parse_file(
    file_path: str | Path,
    extra_metadata: dict[str, Any] | None = None,
) -> ParsedDocument:
    """Parse a file using Unstructured.io's partition().

    Synchronous — callers in async endpoints must use asyncio.to_thread().
    """
    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")
    if path.stat().st_size == 0:
        raise ValueError(f"Empty file: {path}")

    elements = partition(
        filename=str(path),
        strategy=settings.parsing_strategy,
    )

    parsed_elements: list[ParsedElement] = []
    for el in elements or []:
        el_type = _TYPE_MAP.get(type(el).__name__, ElementType.UNCATEGORIZED)

        el_meta: dict[str, Any] = {}
        if hasattr(el, "metadata"):
            meta = el.metadata
            for attr in (
                "page_number",
                "section",
                "filename",
                "filetype",
                "parent_id",
                "text_as_html",
            ):
                val = getattr(meta, attr, None)
                if val is not None:
                    el_meta[attr] = val

        # For Table elements: convert HTML table to Markdown so LLMs can read it.
        # Unstructured's el.text flattens tables into unreadable strings.
        if el_type == ElementType.TABLE and el_meta.get("text_as_html"):
            text = _html_table_to_markdown(el_meta["text_as_html"])
            if not text:
                # Fallback to flat text if HTML conversion fails
                text = el.text.strip() if el.text else ""
        else:
            text = el.text.strip() if el.text else ""

        if not text:
            continue

        parsed_elements.append(
            ParsedElement(text=text, element_type=el_type, metadata=el_meta)
        )

    # Unstructured produced nothing usable. For PDFs, try Poppler's pdftotext —
    # rescues slide decks and other graphics-heavy PDFs where Unstructured's
    # heuristics bail or its element list comes back empty. Only raise the
    # original error if even the fallback finds no text.
    if not parsed_elements:
        fallback = _pdftotext_fallback(path)
        if fallback is not None:
            fallback.metadata.update(extra_metadata or {})
            return fallback
        raise ValueError(f"No content extracted from: {path}")

    filetype = ""
    if elements and hasattr(elements[0], "metadata"):
        filetype = getattr(elements[0].metadata, "filetype", "") or ""

    return ParsedDocument(
        filename=path.name,
        filetype=filetype,
        elements=parsed_elements,
        metadata=extra_metadata or {},
    )


def parse_text(
    text: str,
    filename: str = "manual_input.txt",
    extra_metadata: dict[str, Any] | None = None,
) -> ParsedDocument:
    """Wrap raw text into a ParsedDocument for backward compat with POST /ingest."""
    if not text or not text.strip():
        raise ValueError("Empty text content")

    return ParsedDocument(
        filename=filename,
        filetype="text/plain",
        elements=[
            ParsedElement(
                text=text.strip(),
                element_type=ElementType.NARRATIVE_TEXT,
                metadata={},
            )
        ],
        metadata=extra_metadata or {},
    )
