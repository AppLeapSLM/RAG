"""Generic tree-sitter-based chunker.

One function, every language. Per-language knowledge lives in `queries.py` as
declarative S-expression strings — this module parses, runs the query, walks
captures, emits chunk dicts, and attaches a deterministic NL header to each
chunk for retrieval alignment.

Pipeline per file:
  1. Parse source → AST.
  2. Run the language's query → list of (node, capture_name) tuples.
  3. For each @chunk capture, emit a chunk:
       - Size-guard: if node text > chunk_size, recurse into child nodes; if
         no useful split exists, fall back to the recursive character splitter.
       - For YAML specifically, oversize splits use _split_yaml_document which
         preserves parent-level scalar pairs via `---` injection so every
         sub-chunk remains parseable with yaml.safe_load_all().
       - Build breadcrumbs from @name captures + parent AST nodes.
       - Extract AST-derived features for capture-based NL headers (HCL / Puppet).
  4. Build the final chunk text:
         [bracket metadata]\\n<NL header>\\n\\n<raw content>
     The NL header is a deterministic English description (see nl_headers.py).
  5. If the query returned nothing (unusual file / malformed grammar), fall
     back to the recursive character splitter so we never drop the file.
"""

from __future__ import annotations

import json
import logging
import textwrap
from pathlib import Path
from typing import Any

from backend.chunking.chunker import recursive_character_split
from backend.chunking.doc_type import Format
from backend.chunking.nl_headers import (
    build_hcl_header,
    build_nl_header,
    build_puppet_header,
)
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
    """Chunk a structured-data source file using tree-sitter + NL headers.

    Returns chunk dicts compatible with the existing ingestion path:
      {"text": ..., "metadata": {...}, "element_types": [...]}
    """
    if not source.strip():
        return []

    spec = _resolve_spec(fmt, filename)
    if spec is None:
        return _fallback_chunks(
            source, metadata, filename=filename, fmt=fmt,
            reason=f"no grammar for {fmt.value}",
        )

    try:
        parser, language = _get_parser_and_language(spec.grammar)
    except Exception as e:
        logger.warning("Tree-sitter unavailable for %s: %s — falling back", spec.grammar, e)
        return _fallback_chunks(
            source, metadata, filename=filename, fmt=fmt,
            reason="tree-sitter unavailable",
        )

    source_bytes = source.encode("utf-8")
    tree = parser.parse(source_bytes)
    matches = _run_matches(language, spec.query, tree.root_node)

    chunks = _extract_chunks(matches, source_bytes, spec.grammar)

    if not chunks:
        logger.info("Query returned no chunks for %s — falling back", filename)
        return _fallback_chunks(
            source, metadata, filename=filename, fmt=fmt,
            reason="query empty",
        )

    total = len(chunks)
    results: list[dict[str, Any]] = []
    for i, c in enumerate(chunks):
        bracket = _build_header(
            filename=filename,
            kind=c.get("kind"),
            name=c.get("name"),
            chunk_index=i,
            total=total,
            extra_meta=metadata,
        )
        nl_header = _compute_nl_header(fmt, spec.grammar, c, metadata)
        chunk_metadata = {
            **metadata,
            "chunk_position": i + 1,
            "total_chunks": total,
            "section": _section_label(c.get("kind"), c.get("name")),
            "ast_kind": c.get("kind", ""),
            "ast_name": c.get("name", ""),
        }
        # Final layout: [bracket]\n<NL header>\n\n<raw content>
        combined_text = f"{bracket}{nl_header}\n\n{c['text']}"
        results.append({
            "text": combined_text,
            "metadata": chunk_metadata,
            "element_types": [spec.grammar],
        })

    return results


# ── NL header dispatch ───────────────────────────────────────────────


def _compute_nl_header(
    fmt: Format,
    grammar: str,
    chunk: dict[str, Any],
    metadata: dict[str, Any],
) -> str:
    """Build the NL header line for a single chunk.

    HCL and Puppet use AST captures (extracted in _extract_features) because
    there's no stdlib parser worth the dependency. YAML/JSON/TF-JSON reparse
    the chunk text with stdlib safe_load/json.loads and traverse the dict.
    """
    try:
        if grammar == "hcl":
            return build_hcl_header(
                kind=chunk.get("kind"),
                name=chunk.get("name"),
                attribute_names=chunk.get("features", {}).get("hcl_attrs", []),
            )
        if grammar == "puppet":
            feat = chunk.get("features", {})
            return build_puppet_header(
                kind=chunk.get("kind"),
                name=chunk.get("name"),
                parameters=feat.get("puppet_params", []),
                resource_types=feat.get("puppet_resource_types", []),
                has_ordering=feat.get("puppet_has_ordering", False),
                includes=feat.get("puppet_includes", []),
            )
        return build_nl_header(fmt, chunk["text"], metadata)
    except Exception as e:  # noqa: BLE001 — must never crash ingestion
        logger.warning("NL header computation failed (%s/%s): %s", fmt.value, grammar, e)
        return f"{fmt.value.capitalize()} configuration."


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

    chunk_entries = _dedup_nested(chunk_entries)

    out: list[dict[str, Any]] = []
    size_limit = settings.chunk_size

    for node, kind, name in chunk_entries:
        text = source_bytes[node.start_byte:node.end_byte].decode("utf-8", errors="replace")
        if not text.strip():
            continue

        # Grammar-specific AST feature extraction (for capture-based NL headers).
        features = _extract_features(node, source_bytes, grammar)

        if len(text) <= size_limit:
            out.append({"text": text, "kind": kind, "name": name, "features": features})
            continue

        # Oversize cascade — YAML and JSON use parse-and-reserialize to
        # guarantee every sub-chunk is a valid document in its own format.
        # Byte slicing alone produces orphan fragments (inherited YAML indent,
        # bare JSON scalars / unbraced key-value strings) that fail stdlib
        # parsers and force the NL header into its weakest fallback path.
        if grammar == "yaml":
            sub_chunks = _split_yaml_document(node, source_bytes, size_limit)
        elif grammar == "json":
            sub_chunks = _split_json_document(node, source_bytes, size_limit)
        else:
            sub_chunks = _split_oversize_node(node, source_bytes, size_limit)

        if sub_chunks:
            for i, sub_text in enumerate(sub_chunks):
                out.append({
                    "text": sub_text,
                    "kind": kind,
                    "name": f"{name} [part {i + 1}/{len(sub_chunks)}]" if name else None,
                    "features": features,
                })
        else:
            for sub_text in recursive_character_split(
                text=text,
                chunk_size=size_limit,
                chunk_overlap=settings.chunk_overlap,
            ):
                out.append({"text": sub_text, "kind": kind, "name": name, "features": features})

    return out


def _node_text(nodes: list, source_bytes: bytes) -> str | None:
    """Decode the text of the first node in a capture list; None if empty."""
    if not nodes:
        return None
    n = nodes[0]
    raw = source_bytes[n.start_byte:n.end_byte].decode("utf-8", errors="replace")
    return raw.strip().strip('"').strip("'") or None


def _is_descendant(candidate, ancestor) -> bool:
    return (
        candidate.start_byte >= ancestor.start_byte
        and candidate.end_byte <= ancestor.end_byte
        and candidate != ancestor
    )


def _dedup_nested(entries: list[tuple[Any, str | None, str | None]]) -> list[tuple[Any, str | None, str | None]]:
    """Drop chunks that are strict descendants of another chunk in the list.
    Keeps the outermost. Oversize handling is separate.
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


# ── Size-guard cascade ────────────────────────────────────────────────


def _split_oversize_node(node, source_bytes: bytes, size_limit: int) -> list[str]:
    """Split an oversized AST node along its immediate named child boundaries.

    Format-agnostic. For YAML, prefer _split_yaml_document which additionally
    preserves parent-level scalar pairs across sub-chunks.
    """
    named_children = [c for c in node.children if c.is_named]
    if len(named_children) == 1:
        return _split_oversize_node(named_children[0], source_bytes, size_limit)
    if len(named_children) < 2:
        return []

    out: list[str] = []
    buf = ""
    prev_end: int | None = None

    for child in named_children:
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
            if len(child_text) > size_limit:
                deeper = _split_oversize_node(child, source_bytes, size_limit)
                if deeper:
                    out.extend(deeper)
                    buf = ""
                else:
                    return []
            else:
                buf = child_text
        prev_end = child.end_byte

    if buf:
        out.append(buf)
    return out


def _split_yaml_document(doc_node, source_bytes: bytes, size_limit: int) -> list[str]:
    """YAML-specific oversize split using tree-sitter boundaries + byte
    slicing + textwrap.dedent — deliberately avoids yaml.safe_dump.

    Why not PyYAML round-trip: `yaml.safe_dump` permanently strips inline
    comments, YAML anchors/aliases, custom quoting, and original formatting.
    In enterprise IT, YAML comments routinely carry load-bearing context
    (incident ticket references, compliance notes, TODO markers, regulatory
    flags). Sterilizing them during split is unacceptable data loss.

    Approach:
      1. Walk tree-sitter AST to identify top-level pair nodes.
      2. Classify by raw byte size: <= CONTEXT_BYTES is a context pair,
         the rest are content pairs requiring their own chunk.
      3. Extract each pair's text via byte slicing from the original source —
         comments + formatting + anchors all preserved verbatim.
      4. `textwrap.dedent` sub-chunks so the sliced text parses standalone
         at document root (fixes the inherited-indent issue).
      5. Assemble with `---\\n` separator so NL header's safe_load_all sees
         context and body as separate docs.

    Trade-off acknowledged: sub-chunks of a single very-oversized pair lose
    the pair's key label in parts 2..N. NL header falls back to listing
    whatever keys survive in the sub-chunk. In exchange, we keep every
    comment the operator wrote.

    Falls back to the generic byte-slice splitter if the doc has fewer than
    2 top-level pairs (nothing to classify).
    """
    CONTEXT_BYTES = 500

    # Walk through wrapper nodes (document → block_node → block_mapping) to
    # reach the mapping whose direct children are the scalar/structural pairs.
    current = doc_node
    for _ in range(6):  # bounded; weird grammars shouldn't infinitely chain
        named = [c for c in current.children if c.is_named]
        if len(named) != 1:
            break
        current = named[0]

    pairs = [c for c in current.children if c.is_named]
    if len(pairs) < 2:
        return _split_oversize_node(doc_node, source_bytes, size_limit)

    context_pair_texts: list[str] = []
    large_pairs: list[Any] = []
    for pair in pairs:
        pair_bytes = pair.end_byte - pair.start_byte
        pair_text = source_bytes[pair.start_byte:pair.end_byte].decode("utf-8", errors="replace")
        if pair_bytes <= CONTEXT_BYTES:
            context_pair_texts.append(pair_text)
        else:
            large_pairs.append(pair)

    if not large_pairs:
        # All pairs are small — doc shouldn't have been flagged oversized in
        # the first place. Fall through to the generic splitter.
        return _split_oversize_node(doc_node, source_bytes, size_limit)

    # Dedent context — if pairs came from a nested mapping they may share
    # leading indentation; dedent normalizes them to column 0.
    context_joined = "\n".join(t.rstrip() for t in context_pair_texts)
    context_text = textwrap.dedent(context_joined).strip()
    sep = "\n---\n"
    overhead = len(context_text) + len(sep) if context_text else 0

    out: list[str] = []
    for lp in large_pairs:
        lp_text = source_bytes[lp.start_byte:lp.end_byte].decode("utf-8", errors="replace")
        lp_dedented = textwrap.dedent(lp_text)

        if len(lp_dedented) + overhead <= size_limit:
            out.append(_combine_yaml(context_text, sep, lp_dedented))
            continue

        # Oversized pair — descend into its value mapping and emit sibling
        # groups via a contiguous byte slice that includes leading indent.
        # Using the generic _split_oversize_node here produces invalid YAML
        # because its source-gap concatenation makes sibling 1 flush-left
        # while siblings 2+ inherit source indent from the gap bytes.
        effective_limit = max(size_limit - overhead, 500)
        sub_texts = _split_yaml_mapping_children(lp, source_bytes, effective_limit)
        if not sub_texts:
            out.append(_combine_yaml(context_text, sep, lp_dedented))
            continue

        for st in sub_texts:
            out.append(_combine_yaml(context_text, sep, st))

    return out


def _split_yaml_mapping_children(
    pair_node, source_bytes: bytes, size_limit: int,
) -> list[str]:
    """Split a large YAML mapping_pair by re-emitting its value's children
    as contiguous byte slices (expanded backward to the start of the first
    child's line, then dedented).

    Why not a `\\n` separator between siblings: inter-sibling gap bytes in
    YAML are where inline comments live (tree-sitter doesn't model YAML
    comments as AST nodes — they ride in whitespace between named nodes).
    Replacing the gap with a plain newline would silently delete every
    `# INC-1029` / `# TODO(DATA-342)` / compliance note that an operator
    wrote between entries.

    The correct approach: take a contiguous slice from (first child's line
    start) to (last child's end), preserving all inter-sibling bytes
    including comments, then run textwrap.dedent to normalize indent.

    Known limitation: comments living in the gap between the last child of
    one buffer and the first child of the next buffer fall outside both
    contiguous slices and are dropped. Rare in practice for per-service
    configs; acceptable trade vs. the alternative (deleting every comment).
    """
    value_node = _yaml_pair_value(pair_node)
    if value_node is None:
        return []

    # Walk through wrappers (block_node → block_mapping, etc.) to the
    # container whose named children are the inner pairs.
    inner = value_node
    for _ in range(6):
        named = [c for c in inner.children if c.is_named]
        if len(named) != 1:
            break
        inner = named[0]

    children = [c for c in inner.children if c.is_named]
    if len(children) < 2:
        return []

    def slice_group(group: list) -> str:
        first, last = group[0], group[-1]
        # Back up to the start of first child's line so its leading indent
        # is captured, matching the indent siblings 2+ get from the gap.
        col = _node_column(first)
        start = max(0, first.start_byte - col)
        raw = source_bytes[start:last.end_byte].decode("utf-8", errors="replace")
        return textwrap.dedent(raw).rstrip()

    out: list[str] = []
    buf: list = []
    for child in children:
        candidate = buf + [child]
        if len(slice_group(candidate)) <= size_limit or not buf:
            buf = candidate
        else:
            out.append(slice_group(buf))
            buf = [child]
    if buf:
        out.append(slice_group(buf))
    return out


def _node_column(node) -> int:
    """Column (0-indexed) of a tree-sitter node's start. Defensive against
    binding variants that may lack start_point."""
    sp = getattr(node, "start_point", None)
    if sp is None:
        return 0
    try:
        return sp[1]
    except (TypeError, IndexError):
        return 0


def _combine_yaml(context_text: str, sep: str, body: str) -> str:
    return (context_text + sep + body) if context_text else body


def _yaml_pair_value(pair_node):
    """Return the value child of a YAML block_mapping_pair.

    tree-sitter-yaml marks the `value` field on mapping pairs; we use field
    access where possible and fall back to "last named child" (grammar
    convention: key precedes value) for robustness against grammar variants.
    """
    try:
        v = pair_node.child_by_field_name("value")
        if v is not None:
            return v
    except Exception:  # noqa: BLE001 — defensive, method may not exist on all bindings
        pass
    named = [c for c in pair_node.children if c.is_named]
    return named[-1] if len(named) >= 2 else None


def _split_json_document(node, source_bytes: bytes, size_limit: int) -> list[str]:
    """JSON-specific oversize split using json.loads + json.dumps.

    Byte-slicing a JSON pair node produces `"key": value` text that isn't
    valid JSON at document root — the fragment trap. Same fix as YAML:
    parse the whole node, partition structurally, re-serialize with
    json.dumps so every chunk is valid JSON by construction.

    For `.tf.json` files (structure: {"resource": {type: {name: {...}}}}),
    the recursive partitioner drills through single-root-key wrappers so
    each emitted chunk stays shaped like {"resource": {type: {...}}} — the
    Terraform NL header builder depends on that outer wrapper being present.

    Falls back to the generic byte-slice splitter if the node doesn't parse
    as JSON (corrupt input, grammar emitting odd ranges).
    """
    full_text = source_bytes[node.start_byte:node.end_byte].decode("utf-8", errors="replace")
    stripped = full_text.strip().rstrip(",")

    # The Terraform .tf.json query captures (pair) nodes — text like
    # `"resource": {...}` which is NOT a valid JSON document on its own.
    # Wrap in braces so pair chunks become single-key dicts that json.loads
    # can handle uniformly with generic (object) / (array) captures.
    try:
        data = json.loads(stripped)
    except json.JSONDecodeError:
        try:
            data = json.loads("{" + stripped + "}")
        except json.JSONDecodeError as e:
            logger.info("JSON parse failed during split (%s) — falling back", e)
            return _split_oversize_node(node, source_bytes, size_limit)

    partitions = _partition_json_value(data, size_limit)
    if not partitions:
        return _split_oversize_node(node, source_bytes, size_limit)

    return [json.dumps(p, indent=2) for p in partitions]


def _partition_json_value(data: Any, budget: int) -> list[Any]:
    """Recursively partition a JSON-serializable value so each partition
    serializes to at most `budget` bytes while preserving nesting structure.

    Structure preservation matters for two reasons:
      1. Downstream NL header builders (especially the Terraform one) rely
         on the wrapper key shape `{"resource": {...}}` to recognize the
         block type. Stripping it mid-split breaks that.
      2. The chunk content stays semantically navigable — a reader sees the
         path from root, not an orphan subtree.
    """
    if len(json.dumps(data, indent=2)) <= budget:
        return [data]

    if isinstance(data, dict):
        keys = list(data.keys())
        if not keys:
            return [data]
        if len(keys) == 1:
            # Single-root-key wrapper (typical .tf.json). Drill into the value
            # and rewrap each sub-partition under the same key so the chunk
            # shape remains `{key: <sub>}`.
            k = keys[0]
            wrapper_cost = len(k) + 12  # key + quotes + colon + braces + whitespace
            sub_parts = _partition_json_value(data[k], max(budget - wrapper_cost, 200))
            return [{k: sp} for sp in sub_parts]
        return _group_json_dict(data, budget)

    if isinstance(data, list):
        return _group_json_list(data, budget)

    # Scalar too big to serialize — can't split, emit whole.
    return [data]


def _group_json_dict(data: dict, budget: int) -> list[dict]:
    """Group a dict's entries into serializable sub-dicts, each ≤ budget.

    Every entry is checked *before* bundling. If a single entry alone exceeds
    budget, we recurse into its value rather than swallowing it whole — this
    is the fix for the CMDB regression where `{records: [85 items]}` was
    being emitted as one oversized chunk because the previous implementation
    only recursed on entries encountered after the buffer was non-empty.
    """
    groups: list[dict] = []
    current: dict = {}

    for k, v in data.items():
        single_text = json.dumps({k: v}, indent=2)

        # Entry alone is too big — finalize current, recurse into the value.
        if len(single_text) > budget:
            if current:
                groups.append(current)
                current = {}
            wrapper_cost = len(k) + 12
            sub_parts = _partition_json_value(v, max(budget - wrapper_cost, 200))
            groups.extend({k: sp} for sp in sub_parts)
            continue

        candidate = {**current, k: v}
        if len(json.dumps(candidate, indent=2)) <= budget:
            current = candidate
            continue

        # Bundling would exceed budget — finalize current, start fresh.
        groups.append(current)
        current = {k: v}

    if current:
        groups.append(current)
    return groups


def _group_json_list(data: list, budget: int) -> list[list]:
    """Group a list's items into serializable sub-lists, each ≤ budget.

    Arrays of objects (dicts or lists) get one chunk per item — each item is
    a semantic unit (a CMDB record, a resource, a service) and bundling
    dilutes its embedding for retrieval. Arrays of scalars do get bundled
    up to budget, since a dense list of scalars isn't independently
    retrievable and would otherwise fragment badly.

    Previously these were all bundled greedily which collapsed the 85-chunk
    CMDB ingestion (one per record) into a single oversized array chunk.
    """
    items_are_objects = any(isinstance(item, (dict, list)) for item in data)

    if items_are_objects:
        groups: list[list] = []
        for item in data:
            single_text = json.dumps([item], indent=2)
            if len(single_text) > budget:
                # Individual item too big — recurse into its structure.
                sub_parts = _partition_json_value(item, budget)
                groups.extend([sp] for sp in sub_parts)
            else:
                groups.append([item])
        return groups

    # Scalar array — bundle to budget to avoid over-fragmentation.
    groups = []
    current: list = []
    for item in data:
        candidate = current + [item]
        if len(json.dumps(candidate, indent=2)) <= budget or not current:
            current = candidate
            continue
        groups.append(current)
        current = [item]
    if current:
        groups.append(current)
    return groups


# ── AST feature extraction (for HCL / Puppet NL headers) ─────────────


def _extract_features(node, source_bytes: bytes, grammar: str) -> dict[str, Any]:
    """Extract AST-derived features for NL header builders that use captures.

    No-op (returns {}) for grammars whose NL headers parse raw text instead
    (YAML, JSON). Errors are swallowed — NL header build must never crash
    ingestion.
    """
    try:
        if grammar == "hcl":
            return {"hcl_attrs": _extract_hcl_attrs(node, source_bytes)}
        if grammar == "puppet":
            return _extract_puppet_features(node, source_bytes)
    except Exception as e:  # noqa: BLE001
        logger.debug("feature extraction failed for %s: %s", grammar, e)
    return {}


def _extract_hcl_attrs(block_node, source_bytes: bytes) -> list[str]:
    """Top-level attribute identifiers of an HCL block.

    Walks the block's `body` child and picks identifier names from direct
    `attribute` and `block` children. Skips deeper nesting so we don't
    pollute the header with sub-block internals.
    """
    attrs: list[str] = []
    body = None
    for child in block_node.children:
        if child.type == "body":
            body = child
            break
    if body is None:
        return attrs

    for child in body.children:
        if child.type == "attribute":
            for sub in child.children:
                if sub.type == "identifier":
                    name = source_bytes[sub.start_byte:sub.end_byte].decode("utf-8", errors="replace")
                    if name:
                        attrs.append(name)
                    break
        elif child.type == "block":
            for sub in child.children:
                if sub.type == "identifier":
                    name = source_bytes[sub.start_byte:sub.end_byte].decode("utf-8", errors="replace")
                    if name:
                        attrs.append(f"{name} block")
                    break
    return attrs


def _extract_puppet_features(block_node, source_bytes: bytes) -> dict[str, Any]:
    """Features for Puppet NL headers: parameter names, resource types used,
    presence of dependency chaining operators.
    """
    features: dict[str, Any] = {
        "puppet_params": [],
        "puppet_resource_types": [],
        "puppet_has_ordering": False,
        "puppet_includes": [],
    }

    def walk(n):
        t = n.type
        if t == "parameter":
            for c in n.children:
                if c.type in ("identifier", "variable"):
                    name = source_bytes[c.start_byte:c.end_byte].decode("utf-8", errors="replace")
                    name = name.lstrip("$")
                    if name and name not in features["puppet_params"]:
                        features["puppet_params"].append(name)
                    break
        elif t == "resource_declaration":
            for c in n.children:
                if c.type == "identifier":
                    rtype = source_bytes[c.start_byte:c.end_byte].decode("utf-8", errors="replace")
                    if rtype:
                        features["puppet_resource_types"].append(rtype)
                    break
        elif t in (
            "chain_operator",
            "chaining_arrow",
            "before_arrow",
            "notify_arrow",
            "->",
            "~>",
        ):
            features["puppet_has_ordering"] = True

        for child in n.children:
            walk(child)

    walk(block_node)
    return features


# ── Header / breadcrumbs ──────────────────────────────────────────────


def _build_header(
    filename: str,
    kind: str | None,
    name: str | None,
    chunk_index: int,
    total: int,
    extra_meta: dict[str, Any],
) -> str:
    """Bracket metadata line — citation/debug. Example:
    [File: main.tf | Type: resource | Name: aws_iam_role.app_role | Part 1 of 3]
    Always ends in a newline so the NL header starts on its own line.
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
        # CSV no longer routes here — handled directly by dispatch._process_csv
        # via pipe-format row chunks. Kept as a defensive no-op.
        return None
    return LANGUAGE_SPECS.get(fmt)


def _fallback_chunks(
    source: str,
    metadata: dict[str, Any],
    reason: str,
    filename: str = "unknown",
    fmt: Format | None = None,
) -> list[dict[str, Any]]:
    """Character-split fallback when tree-sitter can't structurally parse the
    file (grammar missing, tree-sitter unavailable, or query returned empty).

    Still attaches a bracket metadata header plus a generic NL description
    so the chunk participates meaningfully in retrieval even without AST
    context. Never drops a file.
    """
    texts = recursive_character_split(
        text=source,
        chunk_size=settings.chunk_size,
        chunk_overlap=settings.chunk_overlap,
    )
    total = len(texts)
    fmt_label = fmt.value.capitalize() if fmt else "Configuration"
    nl = f"{fmt_label} configuration fragment (structural parse unavailable)."

    results: list[dict[str, Any]] = []
    for i, t in enumerate(texts):
        if not t.strip():
            continue
        bracket = _build_header(
            filename=filename,
            kind=None,
            name=None,
            chunk_index=i,
            total=total,
            extra_meta=metadata,
        )
        combined_text = f"{bracket}{nl}\n\n{t}"
        results.append({
            "text": combined_text,
            "metadata": {
                **metadata,
                "chunk_position": i + 1,
                "total_chunks": total,
                "fallback_reason": reason,
            },
            "element_types": ["fallback_char_split"],
        })
    return results


# ── Filename helpers ──────────────────────────────────────────────────


def filename_from_path(path: str) -> str:
    return Path(path).name
