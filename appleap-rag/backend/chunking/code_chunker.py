"""Generic tree-sitter-based chunker.

One function, every language. Per-language knowledge lives in `queries.py` as
declarative S-expression strings — this module parses, runs the query, walks
captures, and emits chunk dicts.

Pipeline per file:
  1. Parse source → AST
  2. Run the language's query → list of (node, capture_name) tuples
  3. For each @chunk capture, emit a chunk:
       - Size-guard: if node text > chunk_size, recurse into child nodes; if
         no useful split exists, fall back to the recursive character splitter.
       - Build breadcrumbs from @name captures + parent AST nodes.
  4. If the query returned nothing (unusual file / malformed grammar), fall
     back to the recursive character splitter so we never drop the file.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from backend.chunking.chunker import recursive_character_split
from backend.chunking.doc_type import Format
from backend.chunking.queries import LANGUAGE_SPECS, LanguageSpec, get_terraform_spec
from backend.config import settings

logger = logging.getLogger(__name__)


# ── Parser/query cache ────────────────────────────────────────────────
# tree-sitter parsers and compiled queries are expensive to build; cache per grammar.

_parser_cache: dict[str, Any] = {}
_language_cache: dict[str, Any] = {}


def _get_parser_and_language(grammar_name: str):
    """Lazy import + cache for tree-sitter parsers. Raises ImportError if the
    tree-sitter extras aren't installed."""
    if grammar_name in _parser_cache:
        return _parser_cache[grammar_name], _language_cache[grammar_name]

    from tree_sitter_language_pack import get_language, get_parser  # type: ignore

    parser = get_parser(grammar_name)
    language = get_language(grammar_name)
    _parser_cache[grammar_name] = parser
    _language_cache[grammar_name] = language
    return parser, language


def _run_matches(language, query_str: str, root_node) -> list[dict[str, list]]:
    """Compile + run a query, returning a list of per-match dicts:
      [{"chunk": [node], "name": [node], "kind": [node]}, ...]

    Using matches() (not captures()) keeps @name / @kind captures scoped to
    their @chunk match — critical for recursive patterns like (pair) in JSON,
    where flattened captures leak labels from nested matches.
    """
    try:
        from tree_sitter import Query, QueryCursor  # type: ignore
        query = Query(language, query_str)
        cursor = QueryCursor(query)
        # matches() returns list of (pattern_index, {capture_name: [nodes]})
        raw = cursor.matches(root_node)
        return [m for _, m in raw]
    except ImportError:
        query = language.query(query_str)  # type: ignore[attr-defined]
        raw = query.matches(root_node)  # type: ignore[attr-defined]
        return [m for _, m in raw]


# ── Entry point ───────────────────────────────────────────────────────


def chunk_code(
    source: str,
    fmt: Format,
    filename: str,
    metadata: dict[str, Any],
) -> list[dict[str, Any]]:
    """Chunk a structured-data source file using tree-sitter.

    Returns chunk dicts compatible with the existing ingestion path:
      {"text": ..., "metadata": {...}, "element_types": [...]}
    """
    if not source.strip():
        return []

    spec = _resolve_spec(fmt, filename)
    if spec is None:
        # No grammar available — fall back to char splitter so we never drop data.
        return _fallback_chunks(source, metadata, reason=f"no grammar for {fmt.value}")

    try:
        parser, language = _get_parser_and_language(spec.grammar)
    except Exception as e:
        logger.warning("Tree-sitter unavailable for %s: %s — falling back", spec.grammar, e)
        return _fallback_chunks(source, metadata, reason="tree-sitter unavailable")

    source_bytes = source.encode("utf-8")
    tree = parser.parse(source_bytes)
    matches = _run_matches(language, spec.query, tree.root_node)

    chunks = _extract_chunks(matches, source_bytes, spec.grammar)

    if not chunks:
        logger.info("Query returned no chunks for %s — falling back", filename)
        return _fallback_chunks(source, metadata, reason="query empty")

    # Build chunk dicts with breadcrumbs + metadata.
    total = len(chunks)
    results: list[dict[str, Any]] = []
    for i, c in enumerate(chunks):
        header = _build_header(
            filename=filename,
            kind=c.get("kind"),
            name=c.get("name"),
            chunk_index=i,
            total=total,
            extra_meta=metadata,
        )
        chunk_metadata = {
            **metadata,
            "chunk_position": i + 1,
            "total_chunks": total,
            "section": _section_label(c.get("kind"), c.get("name")),
            "ast_kind": c.get("kind", ""),
            "ast_name": c.get("name", ""),
        }
        results.append({
            "text": header + c["text"],
            "metadata": chunk_metadata,
            "element_types": [spec.grammar],
        })

    return results


# ── Chunk extraction ──────────────────────────────────────────────────


def _extract_chunks(
    matches: list[dict[str, list]],
    source_bytes: bytes,
    grammar: str,
) -> list[dict[str, Any]]:
    """Convert per-match capture dicts into chunk entries with labels scoped
    to their own match (so nested-pattern captures don't leak across chunks)."""
    chunk_entries: list[tuple[Any, str | None, str | None]] = []  # (node, kind, name)

    for m in matches:
        chunk_nodes = m.get("chunk", [])
        if not chunk_nodes:
            continue
        chunk_node = chunk_nodes[0]
        kind = _node_text(m.get("kind", []), source_bytes)
        name = _node_text(m.get("name", []), source_bytes)
        chunk_entries.append((chunk_node, kind, name))

    # Deduplicate nested chunks: if a chunk contains another chunk entirely,
    # prefer the outer one unless outer exceeds chunk_size.
    chunk_entries = _dedup_nested(chunk_entries)

    # Materialize each chunk, applying the size-guard cascade.
    out: list[dict[str, Any]] = []
    size_limit = settings.chunk_size

    for node, kind, name in chunk_entries:
        text = source_bytes[node.start_byte:node.end_byte].decode("utf-8", errors="replace")
        if not text.strip():
            continue

        if len(text) <= size_limit:
            out.append({"text": text, "kind": kind, "name": name})
            continue

        # Oversize — try to split by immediate named children first.
        sub_chunks = _split_oversize_node(node, source_bytes, size_limit)
        if sub_chunks:
            for i, sub_text in enumerate(sub_chunks):
                out.append({
                    "text": sub_text,
                    "kind": kind,
                    "name": f"{name} [part {i + 1}/{len(sub_chunks)}]" if name else None,
                })
        else:
            # Last resort: recursive character splitter on this node's text.
            for sub_text in recursive_character_split(
                text=text,
                chunk_size=size_limit,
                chunk_overlap=settings.chunk_overlap,
            ):
                out.append({"text": sub_text, "kind": kind, "name": name})

    return out


def _node_text(nodes: list, source_bytes: bytes) -> str | None:
    """Decode the text of the first node in a capture list; None if empty."""
    if not nodes:
        return None
    n = nodes[0]
    raw = source_bytes[n.start_byte:n.end_byte].decode("utf-8", errors="replace")
    return raw.strip().strip('"').strip("'") or None


def _is_descendant(candidate, ancestor) -> bool:
    """True if candidate's byte range is within ancestor's byte range."""
    return (
        candidate.start_byte >= ancestor.start_byte
        and candidate.end_byte <= ancestor.end_byte
        and candidate != ancestor
    )


def _dedup_nested(entries: list[tuple[Any, str | None, str | None]]) -> list[tuple[Any, str | None, str | None]]:
    """Drop chunks that are strict descendants of another chunk in the list.

    Keeps the outermost. (Oversize handling is separate — outer chunks that
    exceed chunk_size get split by _split_oversize_node.)
    """
    kept = []
    nodes = [e[0] for e in entries]
    for i, entry in enumerate(entries):
        node = entry[0]
        if any(
            _is_descendant(node, other)
            for j, other in enumerate(nodes)
            if j != i
        ):
            continue
        kept.append(entry)
    return kept


def _split_oversize_node(node, source_bytes: bytes, size_limit: int) -> list[str]:
    """Split an oversized AST node along its immediate named child boundaries.

    Returns list of text chunks, each ≤ size_limit. Returns [] if splitting
    by children doesn't help (no useful children, or children are also too big).
    """
    named_children = [c for c in node.children if c.is_named]
    # Single-child wrappers (document → object, pair → object, etc.) — recurse
    # into the wrapper so the split actually happens at the meaningful layer.
    if len(named_children) == 1:
        return _split_oversize_node(named_children[0], source_bytes, size_limit)
    if len(named_children) < 2:
        return []

    out: list[str] = []
    buf = ""
    prev_end: int | None = None

    for child in named_children:
        # Include any gap text (whitespace/punctuation between children) to
        # preserve formatting.
        if prev_end is not None:
            gap = source_bytes[prev_end:child.start_byte].decode("utf-8", errors="replace")
        else:
            gap = ""
        child_text = source_bytes[child.start_byte:child.end_byte].decode("utf-8", errors="replace")
        candidate = buf + gap + child_text if buf else child_text

        if len(candidate) <= size_limit:
            buf = candidate
        else:
            if buf:
                out.append(buf)
            # If the child alone is too big, recurse; else start new buffer.
            if len(child_text) > size_limit:
                deeper = _split_oversize_node(child, source_bytes, size_limit)
                if deeper:
                    out.extend(deeper)
                    buf = ""
                else:
                    # Give up on splitting this child — keep it as a single
                    # oversized chunk; caller will fall through to recursive
                    # char splitter.
                    return []
            else:
                buf = child_text
        prev_end = child.end_byte

    if buf:
        out.append(buf)
    return out


# ── Header / breadcrumbs ──────────────────────────────────────────────


def _build_header(
    filename: str,
    kind: str | None,
    name: str | None,
    chunk_index: int,
    total: int,
    extra_meta: dict[str, Any],
) -> str:
    """Build a contextual header line for the chunk.

    Example: [File: main.tf | Type: resource | Name: aws_iam_role.app_role | Part 1 of 3]
    """
    parts: list[str] = [f"File: {filename}"]

    if kind:
        parts.append(f"Type: {kind}")
    if name:
        parts.append(f"Name: {name}")

    source = extra_meta.get("source", "")
    if source and source not in ("manual", "upload", ""):
        parts.append(f"Source: {source}")

    folder = extra_meta.get("folder_path", "")
    if folder and folder != "/":
        parts.append(f"Path: {folder}")

    parts.append(f"Part {chunk_index + 1} of {total}")
    return "[" + " | ".join(parts) + "]\n"


def _section_label(kind: str | None, name: str | None) -> str:
    if kind and name:
        return f"{kind} {name}"
    return name or kind or ""


# ── Grammar resolution + fallback ─────────────────────────────────────


def _resolve_spec(fmt: Format, filename: str) -> LanguageSpec | None:
    """Pick the right LanguageSpec for the format, handling Terraform's HCL/JSON split."""
    if fmt == Format.TERRAFORM:
        is_json = filename.lower().endswith(".tf.json")
        return get_terraform_spec(is_json=is_json)
    if fmt == Format.CSV:
        # CSV is converted to JSON upstream (see dispatch.py); shouldn't hit here.
        return None
    return LANGUAGE_SPECS.get(fmt)


def _fallback_chunks(
    source: str,
    metadata: dict[str, Any],
    reason: str,
) -> list[dict[str, Any]]:
    """Character-split fallback when tree-sitter isn't available or query empties out."""
    texts = recursive_character_split(
        text=source,
        chunk_size=settings.chunk_size,
        chunk_overlap=settings.chunk_overlap,
    )
    total = len(texts)
    return [
        {
            "text": t,
            "metadata": {
                **metadata,
                "chunk_position": i + 1,
                "total_chunks": total,
                "fallback_reason": reason,
            },
            "element_types": ["fallback_char_split"],
        }
        for i, t in enumerate(texts)
        if t.strip()
    ]


# ── Filename helpers ──────────────────────────────────────────────────


def filename_from_path(path: str) -> str:
    return Path(path).name
