from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from unstructured.partition.auto import partition

from backend.config import settings
from backend.parsing.base import ElementType, ParsedDocument, ParsedElement

logger = logging.getLogger(__name__)

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

    if not elements:
        raise ValueError(f"No content extracted from: {path}")

    parsed_elements: list[ParsedElement] = []
    for el in elements:
        text = el.text.strip() if el.text else ""
        if not text:
            continue

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

        parsed_elements.append(
            ParsedElement(text=text, element_type=el_type, metadata=el_meta)
        )

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
