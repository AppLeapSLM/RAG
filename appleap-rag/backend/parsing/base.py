from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class ElementType(str, Enum):
    """Content types that Unstructured.io can detect."""

    TITLE = "Title"
    NARRATIVE_TEXT = "NarrativeText"
    TABLE = "Table"
    LIST_ITEM = "ListItem"
    IMAGE = "Image"
    CODE_SNIPPET = "CodeSnippet"
    HEADER = "Header"
    FOOTER = "Footer"
    PAGE_BREAK = "PageBreak"
    FORMULA = "Formula"
    UNCATEGORIZED = "UncategorizedText"


@dataclass
class ParsedElement:
    """Single element extracted from a document by Unstructured.io."""

    text: str
    element_type: ElementType
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class ParsedDocument:
    """Full parsed output for a single file."""

    filename: str
    filetype: str  # e.g. "application/pdf", "text/markdown"
    elements: list[ParsedElement]
    metadata: dict[str, Any] = field(default_factory=dict)
