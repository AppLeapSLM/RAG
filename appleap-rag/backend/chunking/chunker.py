from __future__ import annotations

import logging
from typing import Any

from backend.config import settings
from backend.parsing.base import ElementType, ParsedDocument

logger = logging.getLogger(__name__)

# Default separator hierarchy: paragraph → line → word → character
DEFAULT_SEPARATORS = ["\n\n", "\n", " ", ""]


# ── Public API ────────────────────────────────────────────────────────


async def chunk_parsed_document_async(
    doc: ParsedDocument,
) -> list[dict[str, Any]]:
    """Convert a parsed document into chunks ready for embedding and storage.

    1. Join all parsed elements into a single clean text string
    2. Split using the recursive character splitter
    3. Apply contextual headers to each chunk
    """
    if not doc.elements:
        return []

    # Combine all parsed elements into one continuous string.
    # Unstructured is used only as a parser — chunking is ours.
    full_text = "\n\n".join(el.text for el in doc.elements if el.text.strip())

    if not full_text.strip():
        return []

    # Split into chunks
    chunk_texts = recursive_character_split(
        text=full_text,
        chunk_size=settings.chunk_size,
        chunk_overlap=settings.chunk_overlap,
        separators=DEFAULT_SEPARATORS,
    )

    logger.info(
        "Chunked %s into %d chunks (avg %d chars)",
        doc.filename,
        len(chunk_texts),
        sum(len(c) for c in chunk_texts) // max(len(chunk_texts), 1),
    )

    # Build chunk dicts with metadata
    all_types = list({el.element_type.value for el in doc.elements})
    raw_chunks = [
        {
            "text": text,
            "metadata": {**doc.metadata},
            "element_types": all_types,
        }
        for text in chunk_texts
        if text.strip()
    ]

    return _apply_context_headers(raw_chunks, doc)


# ── Recursive Character Splitter ─────────────────────────────────────


def recursive_character_split(
    text: str,
    chunk_size: int = 3000,
    chunk_overlap: int = 200,
    separators: list[str] | None = None,
) -> list[str]:
    """Split text into chunks using a hierarchy of separators.

    Tries the highest-level separator first (paragraph breaks), falling back
    to lower levels (line breaks, spaces, characters) when needed.

    The recursive fallback handles edge cases like massive unbroken text
    blocks (e.g., minified JSON, long URLs): if a single piece exceeds
    chunk_size after splitting by the current separator, the function
    recurses with the next separator in the hierarchy until it fits.

    Args:
        text: The input text to split.
        chunk_size: Hard maximum characters per chunk (default 3000).
        chunk_overlap: Characters of overlap between consecutive chunks (default 200).
        separators: Ordered list of separators to try. Defaults to
                    ["\\n\\n", "\\n", " ", ""] (paragraph → line → word → char).

    Returns:
        List of text chunks, each ≤ chunk_size characters.
    """
    if separators is None:
        separators = DEFAULT_SEPARATORS

    # Base case: text fits in one chunk
    if len(text) <= chunk_size:
        stripped = text.strip()
        return [stripped] if stripped else []

    # Find the first separator that exists in the text
    active_sep = ""
    remaining_seps = []
    for i, sep in enumerate(separators):
        if sep == "":
            # Character-level split — always works as last resort
            active_sep = ""
            remaining_seps = []
            break
        if sep in text:
            active_sep = sep
            remaining_seps = separators[i + 1 :]
            break

    # Split using the active separator
    if active_sep:
        pieces = text.split(active_sep)
    else:
        # Character-by-character — split into individual chars
        pieces = list(text)

    # Accumulate pieces into chunks with overlap
    final_chunks: list[str] = []
    current_chunk = ""

    for piece in pieces:
        # Calculate what the chunk would be if we add this piece
        if current_chunk:
            candidate = current_chunk + active_sep + piece
        else:
            candidate = piece

        if len(candidate) <= chunk_size:
            # Fits — accumulate
            current_chunk = candidate
        else:
            # Doesn't fit — finalize current chunk
            if current_chunk:
                final_chunks.append(current_chunk.strip())

                # Calculate overlap: grab last (overlap + 50) chars, snap to word boundary
                overlap_text = _get_overlap(current_chunk, chunk_overlap)
            else:
                overlap_text = ""

            # Check if this single piece is itself oversized
            if len(piece) > chunk_size:
                # Recursive fallback: split this piece with the next separator level
                sub_chunks = recursive_character_split(
                    text=piece,
                    chunk_size=chunk_size,
                    chunk_overlap=chunk_overlap,
                    separators=remaining_seps if remaining_seps else [""],
                )
                # Prepend overlap to the first sub-chunk if it fits
                if sub_chunks and overlap_text:
                    merged = overlap_text + active_sep + sub_chunks[0]
                    if len(merged) <= chunk_size:
                        sub_chunks[0] = merged
                    else:
                        # Overlap doesn't fit — just add it as context
                        pass
                final_chunks.extend(sub_chunks)
                current_chunk = ""
            else:
                # Start new chunk with overlap + current piece
                if overlap_text:
                    current_chunk = overlap_text + active_sep + piece
                    # If overlap + piece exceeds chunk_size, drop the overlap
                    if len(current_chunk) > chunk_size:
                        current_chunk = piece
                else:
                    current_chunk = piece

    # Don't forget the last chunk
    if current_chunk and current_chunk.strip():
        final_chunks.append(current_chunk.strip())

    return [c for c in final_chunks if c]


def _get_overlap(text: str, overlap_size: int) -> str:
    """Get the last ~overlap_size characters, snapped to a word boundary.

    Grabs overlap_size + 50 characters, then finds the first space to
    avoid cutting mid-word. Falls back to exact overlap if no space found.
    """
    if len(text) <= overlap_size:
        return text

    # Grab extra chars so we can snap to a word boundary
    grab_size = min(overlap_size + 50, len(text))
    raw = text[-grab_size:]

    # Find the first space to snap to a complete word
    space_idx = raw.find(" ")
    if space_idx != -1 and space_idx < 50:
        return raw[space_idx + 1 :]

    # No space found in the buffer — fall back to exact overlap
    return text[-overlap_size:]


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

    title = meta.get("title") or doc.filename
    if title:
        parts.append(f"Document: {title}")

    if section:
        parts.append(f"Section: {section}")

    source = meta.get("source", "")
    if source and source not in ("manual", "upload", ""):
        parts.append(f"Source: {source}")

    folder = meta.get("folder_path", "")
    if folder and folder != "/":
        parts.append(f"Path: {folder}")

    author = meta.get("owner_email", "")
    if author:
        parts.append(f"Author: {author}")

    modified = meta.get("last_modified", "")
    if modified:
        parts.append(f"Modified: {modified[:10]}")

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
            break

    return " > ".join(sections)
