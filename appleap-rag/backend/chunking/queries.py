"""Tree-sitter queries for structural chunking.

Each entry maps a Format to:
  - grammar: the tree-sitter-language-pack grammar name
  - query:   an S-expression capturing "top-level chunks" for that language

Queries use two capture names:
  - @chunk : the node to emit as a chunk
  - @name  : optional — an identifier to surface as a breadcrumb label

Adding a language = one dict entry. No Python code changes.
"""

from __future__ import annotations

from dataclasses import dataclass

from backend.chunking.doc_type import Format


@dataclass(frozen=True)
class LanguageSpec:
    grammar: str  # tree-sitter-language-pack grammar name
    query: str    # S-expression query string


# ── Per-language specs ────────────────────────────────────────────────

_PYTHON = LanguageSpec(
    grammar="python",
    query="""
        (function_definition name: (identifier) @name) @chunk
        (class_definition    name: (identifier) @name) @chunk
        (decorated_definition) @chunk
    """,
)

_GO = LanguageSpec(
    grammar="go",
    query="""
        (function_declaration name: (identifier) @name) @chunk
        (method_declaration   name: (field_identifier) @name) @chunk
        (type_declaration)    @chunk
    """,
)

_RUBY = LanguageSpec(
    grammar="ruby",
    query="""
        (method   name: (_) @name) @chunk
        (class    name: (_) @name) @chunk
        (module   name: (_) @name) @chunk
    """,
)

_JAVASCRIPT = LanguageSpec(
    grammar="javascript",
    query="""
        (function_declaration name: (identifier) @name) @chunk
        (class_declaration    name: (identifier) @name) @chunk
        (method_definition    name: (property_identifier) @name) @chunk
        (lexical_declaration) @chunk
    """,
)

_TYPESCRIPT = LanguageSpec(
    grammar="typescript",
    query="""
        (function_declaration  name: (identifier) @name) @chunk
        (class_declaration     name: (type_identifier) @name) @chunk
        (method_definition     name: (property_identifier) @name) @chunk
        (interface_declaration name: (type_identifier) @name) @chunk
        (type_alias_declaration name: (type_identifier) @name) @chunk
    """,
)

_BASH = LanguageSpec(
    grammar="bash",
    query="""
        (function_definition name: (word) @name) @chunk
    """,
)

_DOCKERFILE = LanguageSpec(
    grammar="dockerfile",
    # Dockerfiles are flat — chunk per top-level instruction is too granular.
    # Group by stage_instruction (FROM ... blocks).
    query="""
        (from_instruction) @chunk
        (stage) @chunk
    """,
)

# Terraform HCL — resource/module/variable/output/data/provider blocks.
_TERRAFORM_HCL = LanguageSpec(
    grammar="hcl",
    query="""
        (block
            (identifier) @kind
            (string_lit)? @name
        ) @chunk
    """,
)

# Terraform JSON — .tf.json files. Top-level keys are "resource", "module", etc.,
# each mapping to objects. We chunk at the innermost named-resource level.
_TERRAFORM_JSON = LanguageSpec(
    grammar="json",
    query="""
        (pair key: (string) @name) @chunk
    """,
)

_YAML = LanguageSpec(
    grammar="yaml",
    # Each YAML document (separated by ---) becomes one chunk.
    # For single-doc YAML, fall back to top-level block_mapping_pair nodes.
    query="""
        (document) @chunk
    """,
)

_JSON = LanguageSpec(
    grammar="json",
    # Generic JSON: capture objects and arrays. Outermost-wins dedup means the
    # root container becomes one chunk; size-guard splits by children. For an
    # array-of-objects (CMDB-style), that yields one chunk per array element;
    # for a single-object root, one chunk per top-level pair.
    query="""
        (object) @chunk
        (array) @chunk
    """,
)

_PUPPET = LanguageSpec(
    grammar="puppet",
    # Real tree-sitter-puppet node types (verified against grammar):
    # - class_definition         → Puppet class
    # - node_definition          → node 'hostname' { ... }
    # - resource_declaration     → e.g. package { 'nginx': ensure => installed }
    query="""
        (class_definition)     @chunk
        (node_definition)      @chunk
        (resource_declaration) @chunk
    """,
)


# ── Dispatch table ────────────────────────────────────────────────────

LANGUAGE_SPECS: dict[Format, LanguageSpec] = {
    Format.PYTHON: _PYTHON,
    Format.GO: _GO,
    Format.RUBY: _RUBY,
    Format.JAVASCRIPT: _JAVASCRIPT,
    Format.TYPESCRIPT: _TYPESCRIPT,
    Format.BASH: _BASH,
    Format.DOCKERFILE: _DOCKERFILE,
    Format.YAML: _YAML,
    Format.JSON: _JSON,
    Format.PUPPET: _PUPPET,
    # Terraform resolved at runtime (HCL vs JSON) — see code_chunker.
}


def get_terraform_spec(is_json: bool) -> LanguageSpec:
    """Terraform has two on-disk formats: HCL (.tf) and JSON (.tf.json)."""
    return _TERRAFORM_JSON if is_json else _TERRAFORM_HCL
