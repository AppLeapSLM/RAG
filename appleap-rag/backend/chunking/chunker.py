from __future__ import annotations

import logging
import math
from typing import Any

from unstructured.chunking.title import chunk_by_title
from unstructured.documents.elements import (
    ListItem,
    NarrativeText,
    Table,
    Title,
)

from backend.config import settings
from backend.parsing.base import ElementType, ParsedDocument, ParsedElement

logger = logging.getLogger(__name__)


# ── Public API ────────────────────────────────────────────────────────


def chunk_parsed_document(
    doc: ParsedDocument,
    strategy: str | None = None,
) -> list[dict[str, Any]]:
    """Route a parsed document to the appropriate chunking strategy (sync).

    Raises RuntimeError if by_similarity is selected — use the async version.
    """
    if not doc.elements:
        return []

    effective = strategy or _detect_strategy(doc)
    logger.info(
        "Chunking %s with strategy=%s (%d elements)",
        doc.filename,
        effective,
        len(doc.elements),
    )

    if effective == "by_title":
        raw = _chunk_by_title(doc)
    elif effective == "by_similarity":
        raise RuntimeError(
            "by_similarity requires async (calls embed_batch). "
            "Use chunk_parsed_document_async() instead."
        )
    elif effective == "naive":
        raw = _chunk_naive(doc)
    else:
        logger.warning("Unknown strategy %s, falling back to by_title", effective)
        raw = _chunk_by_title(doc)

    return _apply_context_headers(raw, doc)


async def chunk_parsed_document_async(
    doc: ParsedDocument,
    strategy: str | None = None,
) -> list[dict[str, Any]]:
    """Async version — supports all strategies including by_similarity."""
    import asyncio

    if not doc.elements:
        return []

    effective = strategy or _detect_strategy(doc)

    if effective == "by_similarity":
        raw = await _chunk_by_similarity(doc)
        return _apply_context_headers(raw, doc)
    else:
        return await asyncio.to_thread(chunk_parsed_document, doc, effective)


# ── Contextual chunk headers ─────────────────────────────────────────


def _apply_context_headers(
    chunks: list[dict[str, Any]],
    doc: ParsedDocument,
) -> list[dict[str, Any]]:
    """Prepend a contextual header to every chunk and enrich metadata.

    The header becomes part of the text that gets embedded, so the embedding
    captures document context (title, section, source, author, path).
    """
    total = len(chunks)
    title_texts = frozenset(
        el.text for el in doc.elements if el.element_type == ElementType.TITLE
    )

    for i, chunk in enumerate(chunks):
        section = _extract_section(chunk["text"], title_texts)
        header = _build_header(doc, section, i, total)
        chunk["text"] = header + chunk["text"]
        chunk["metadata"]["section"] = section
        chunk["metadata"]["chunk_position"] = i + 1
        chunk["metadata"]["total_chunks"] = total

    return chunks


def _build_header(
    doc: ParsedDocument,
    section: str,
    chunk_index: int,
    total_chunks: int,
) -> str:
    """Build a bracketed header line from available metadata."""
    parts: list[str] = []
    meta = doc.metadata

    # Document title — use explicit title from metadata, fall back to filename
    title = meta.get("title") or doc.filename
    if title:
        parts.append(f"Document: {title}")

    # Section hierarchy (e.g. "Kubernetes Guide > Prerequisites")
    if section:
        parts.append(f"Section: {section}")

    # Source system (skip generic sources)
    source = meta.get("source", "")
    if source and source not in ("manual", "upload", ""):
        parts.append(f"Source: {source}")

    # Folder / project path (from connectors)
    folder = meta.get("folder_path", "")
    if folder and folder != "/":
        parts.append(f"Path: {folder}")

    # Author / owner
    author = meta.get("owner_email", "")
    if author:
        parts.append(f"Author: {author}")

    # Last modified — truncate to date if ISO timestamp
    modified = meta.get("last_modified", "")
    if modified:
        parts.append(f"Modified: {modified[:10]}")

    # Chunk position
    parts.append(f"Part {chunk_index + 1} of {total_chunks}")

    return "[" + " | ".join(parts) + "]\n"


def _extract_section(chunk_text: str, title_texts: frozenset[str]) -> str:
    """Extract section title hierarchy from the start of a chunk.

    Walks the leading lines of the chunk and collects consecutive titles,
    producing a path like "Kubernetes Guide > Prerequisites".
    """
    if not title_texts:
        return ""

    sections: list[str] = []
    for line in chunk_text.split("\n"):
        line = line.strip()
        if not line:
            continue
        if line in title_texts:
            sections.append(line)
        else:
            break  # first non-title line — stop

    return " > ".join(sections)


# ── Strategy auto-detection ───────────────────────────────────────────


def _detect_strategy(doc: ParsedDocument) -> str:
    """Pick strategy based on document content.

    - Titles present → by_title (structure exists)
    - No titles → by_similarity (unstructured prose)
    - Single element → naive (nothing to split)
    """
    if settings.chunking_strategy != "auto":
        return settings.chunking_strategy

    if len(doc.elements) <= 1:
        return "naive"

    has_titles = any(el.element_type == ElementType.TITLE for el in doc.elements)
    return "by_title" if has_titles else "by_similarity"


# ── by_title (Unstructured native) ───────────────────────────────────


def _chunk_by_title(doc: ParsedDocument) -> list[dict[str, Any]]:
    """Use Unstructured's chunk_by_title on parsed elements."""
    # Convert ParsedElements → Unstructured Element objects
    unstructured_elements = []
    for el in doc.elements:
        if el.element_type == ElementType.TITLE:
            unstructured_elements.append(Title(text=el.text))
        elif el.element_type == ElementType.TABLE:
            unstructured_elements.append(Table(text=el.text))
        elif el.element_type == ElementType.LIST_ITEM:
            unstructured_elements.append(ListItem(text=el.text))
        else:
            unstructured_elements.append(NarrativeText(text=el.text))

    chunks = chunk_by_title(
        unstructured_elements,
        max_characters=settings.chunk_max_characters,
        new_after_n_chars=settings.chunk_new_after_n_chars,
        overlap=settings.chunk_overlap,
        combine_text_under_n_chars=settings.chunk_combine_under_n_chars,
    )

    result: list[dict[str, Any]] = []
    for chunk in chunks:
        text = chunk.text.strip()
        if not text:
            continue
        result.append(
            {
                "text": text,
                "metadata": {**doc.metadata},
                "element_types": [type(chunk).__name__],
            }
        )
    return result


# ── by_similarity (custom, async) ────────────────────────────────────


async def _chunk_by_similarity(doc: ParsedDocument) -> list[dict[str, Any]]:
    """Custom semantic chunking using Nomic embeddings via Ollama.

    1. Embed all elements in one batch
    2. Compute cosine similarity between consecutive embeddings
    3. Cut where similarity < threshold
    4. Group into chunks, split if exceeding max_characters
    """
    from backend.embedding.embedder import embed_batch

    element_texts = [el.text for el in doc.elements]
    embeddings = await embed_batch(element_texts)

    # Consecutive cosine similarities
    similarities: list[float] = []
    for i in range(len(embeddings) - 1):
        similarities.append(_cosine_similarity(embeddings[i], embeddings[i + 1]))

    # Cut points where similarity drops below threshold
    cut_indices: list[int] = []
    for i, sim in enumerate(similarities):
        if sim < settings.similarity_threshold:
            cut_indices.append(i + 1)

    # Group elements into segments with overlap.
    # Carry trailing elements from each segment into the next so boundary
    # content appears in both chunks' embeddings.
    overlap_chars = settings.chunk_overlap
    segments: list[list[int]] = []
    start = 0
    for cut in cut_indices:
        segments.append(list(range(start, cut)))
        # Walk backwards from the cut to find overlap elements
        overlap_start = cut
        char_count = 0
        for j in range(cut - 1, start - 1, -1):
            char_count += len(doc.elements[j].text)
            if char_count >= overlap_chars:
                overlap_start = j
                break
        start = min(overlap_start, cut)  # overlap elements join the next segment
    segments.append(list(range(start, len(doc.elements))))

    # Convert segments to chunks
    result: list[dict[str, Any]] = []
    for segment_indices in segments:
        if not segment_indices:
            continue

        segment_elements = [doc.elements[i] for i in segment_indices]
        segment_text = "\n\n".join(el.text for el in segment_elements)
        segment_types = list({el.element_type.value for el in segment_elements})

        if len(segment_text) <= settings.similarity_max_characters:
            result.append(
                {
                    "text": segment_text,
                    "metadata": {**doc.metadata},
                    "element_types": segment_types,
                }
            )
        else:
            result.extend(
                _split_segment_by_size(
                    segment_elements,
                    doc.metadata,
                    settings.similarity_max_characters,
                )
            )

    return result


# ── naive (fallback) ─────────────────────────────────────────────────


def _chunk_naive(doc: ParsedDocument) -> list[dict[str, Any]]:
    """Simple character-based splitting for single-element docs."""
    full_text = "\n\n".join(el.text for el in doc.elements)
    size = settings.chunk_max_characters
    overlap = settings.chunk_overlap

    paragraphs = [p.strip() for p in full_text.split("\n\n") if p.strip()]
    chunks: list[str] = []
    current = ""

    for para in paragraphs:
        if current and len(current) + len(para) + 2 > size:
            chunks.append(current)
            current = current[-overlap:] + "\n\n" + para if overlap else para
        else:
            current = current + "\n\n" + para if current else para
    if current:
        chunks.append(current)

    all_types = list({el.element_type.value for el in doc.elements})
    return [
        {"text": c, "metadata": {**doc.metadata}, "element_types": all_types}
        for c in chunks
        if c.strip()
    ]


# ── Helpers ───────────────────────────────────────────────────────────


def _cosine_similarity(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(x * x for x in b))
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot / (norm_a * norm_b)


def _split_segment_by_size(
    elements: list[ParsedElement],
    doc_metadata: dict[str, Any],
    max_chars: int,
) -> list[dict[str, Any]]:
    """Split a list of elements into chunks that fit within max_chars."""
    result: list[dict[str, Any]] = []
    current_texts: list[str] = []
    current_types: set[str] = set()
    current_len = 0

    for el in elements:
        addition = len(el.text) + (2 if current_texts else 0)
        if current_len + addition > max_chars and current_texts:
            result.append(
                {
                    "text": "\n\n".join(current_texts),
                    "metadata": {**doc_metadata},
                    "element_types": list(current_types),
                }
            )
            current_texts = []
            current_types = set()
            current_len = 0

        current_texts.append(el.text)
        current_types.add(el.element_type.value)
        current_len += addition

    if current_texts:
        result.append(
            {
                "text": "\n\n".join(current_texts),
                "metadata": {**doc_metadata},
                "element_types": list(current_types),
            }
        )
    return result


# ── Legacy API (backward compat) ─────────────────────────────────────


def chunk_text(
    text: str,
    chunk_size: int = settings.chunk_size,
    overlap: int = settings.chunk_overlap,
) -> list[str]:
    """DEPRECATED: Use chunk_parsed_document() or chunk_parsed_document_async()."""
    from backend.parsing.parser import parse_text

    doc = parse_text(text)
    chunks = _chunk_naive(doc)
    return [c["text"] for c in chunks]
