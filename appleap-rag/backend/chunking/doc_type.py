"""Classify incoming files into a format (drives chunker choice) and
a semantic doc_type (drives retrieval filtering/boosting).

Two independent axes:
  - Format: how the file is parsed and chunked (terraform, yaml, prose, csv, ...)
  - DocType: what the file semantically *is* (runbook, incident, adr, cmdb, ...)

Path hints (e.g. `runbooks/`, `incidents/`) are the primary signal for doc_type.
Extension is the primary signal for format. Both fall through to sensible defaults.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from pathlib import Path


class Format(str, Enum):
    """Chunking strategy dispatch."""

    PROSE = "prose"            # Unstructured.io + recursive char splitter
    TERRAFORM = "terraform"    # tree-sitter (hcl or json)
    YAML = "yaml"              # tree-sitter yaml
    JSON = "json"              # tree-sitter json
    PUPPET = "puppet"          # tree-sitter puppet
    PYTHON = "python"          # tree-sitter python
    GO = "go"                  # tree-sitter go
    RUBY = "ruby"              # tree-sitter ruby
    JAVASCRIPT = "javascript"  # tree-sitter javascript
    TYPESCRIPT = "typescript"  # tree-sitter typescript
    BASH = "bash"              # tree-sitter bash
    DOCKERFILE = "dockerfile"  # tree-sitter dockerfile
    CSV = "csv"                # csv -> json -> tree-sitter json


class DocType(str, Enum):
    """Semantic doc_type for retrieval filtering/boosting."""

    RUNBOOK = "runbook"
    INCIDENT = "incident"
    PLAYBOOK = "playbook"
    ADR = "adr"
    ARCHITECTURE = "architecture"
    CMDB = "cmdb"
    TERRAFORM_CONFIG = "terraform_config"
    KUBERNETES_MANIFEST = "kubernetes_manifest"
    HELM_VALUES = "helm_values"
    DOCKER_COMPOSE = "docker_compose"
    PUPPET_CONFIG = "puppet_config"
    CODE = "code"
    DOCUMENTATION = "documentation"
    GENERIC = "generic"


@dataclass(frozen=True)
class FileClassification:
    format: Format
    doc_type: DocType


# Format inference by extension / filename.
# Tree-sitter grammar name (used by code_chunker) is derived from Format.value.
_EXT_TO_FORMAT: dict[str, Format] = {
    # IaC / config
    ".tf": Format.TERRAFORM,
    ".tfvars": Format.TERRAFORM,
    ".hcl": Format.TERRAFORM,
    ".yaml": Format.YAML,
    ".yml": Format.YAML,
    ".json": Format.JSON,
    ".pp": Format.PUPPET,
    # Code
    ".py": Format.PYTHON,
    ".pyi": Format.PYTHON,
    ".go": Format.GO,
    ".rb": Format.RUBY,
    ".js": Format.JAVASCRIPT,
    ".mjs": Format.JAVASCRIPT,
    ".cjs": Format.JAVASCRIPT,
    ".jsx": Format.JAVASCRIPT,
    ".ts": Format.TYPESCRIPT,
    ".tsx": Format.TYPESCRIPT,
    ".sh": Format.BASH,
    ".bash": Format.BASH,
    # Tabular
    ".csv": Format.CSV,
    # Prose
    ".md": Format.PROSE,
    ".markdown": Format.PROSE,
    ".txt": Format.PROSE,
    ".rst": Format.PROSE,
    ".pdf": Format.PROSE,
    ".docx": Format.PROSE,
    ".doc": Format.PROSE,
    ".pptx": Format.PROSE,
    ".ppt": Format.PROSE,
    ".html": Format.PROSE,
    ".htm": Format.PROSE,
    ".eml": Format.PROSE,
    ".rtf": Format.PROSE,
}

# Filenames (no extension, or conventional) mapped directly.
_FILENAME_TO_FORMAT: dict[str, Format] = {
    "Dockerfile": Format.DOCKERFILE,
    "Containerfile": Format.DOCKERFILE,
    "Puppetfile": Format.RUBY,  # Puppetfile uses Ruby DSL
}


def _infer_format(path: Path) -> Format:
    """Pick a Format from filename + extension. Defaults to PROSE for unknowns."""
    # `.tf.json` — Terraform emitted as JSON, not plain JSON
    lower = path.name.lower()
    if lower.endswith(".tf.json"):
        return Format.TERRAFORM

    if path.name in _FILENAME_TO_FORMAT:
        return _FILENAME_TO_FORMAT[path.name]

    ext = path.suffix.lower()
    return _EXT_TO_FORMAT.get(ext, Format.PROSE)


# Semantic doc_type: inferred from path components (NovaCrest-style folder conventions).
_PATH_HINT_TO_DOCTYPE: dict[str, DocType] = {
    "runbooks": DocType.RUNBOOK,
    "incidents": DocType.INCIDENT,
    "playbooks": DocType.PLAYBOOK,
    "documentation": DocType.DOCUMENTATION,
    "cmdb": DocType.CMDB,
    "terraform": DocType.TERRAFORM_CONFIG,
    "kubernetes": DocType.KUBERNETES_MANIFEST,
    "k8s": DocType.KUBERNETES_MANIFEST,
    "helm": DocType.HELM_VALUES,
    "docker-compose": DocType.DOCKER_COMPOSE,
    "compose": DocType.DOCKER_COMPOSE,
    "puppet": DocType.PUPPET_CONFIG,
    "topology": DocType.ARCHITECTURE,
    "architecture": DocType.ARCHITECTURE,
}

# ADR files often live under documentation/ but are identifiable by filename prefix.
_FILENAME_PREFIX_TO_DOCTYPE: dict[str, DocType] = {
    "adr-": DocType.ADR,
    "INC-": DocType.INCIDENT,
}


def _infer_doc_type(path: Path, fmt: Format, hints: list[str]) -> DocType:
    """Pick a DocType from filename prefixes, path hints, extra hints, then format fallback.

    `hints` is an ordered list of free-form strings (e.g. source='novacrest/runbooks',
    category='runbooks') that are searched for a doc_type keyword if the path
    itself doesn't contain one.
    """
    # Filename prefix wins — ADRs under documentation/ are still ADRs.
    name = path.name
    for prefix, dt in _FILENAME_PREFIX_TO_DOCTYPE.items():
        if name.startswith(prefix):
            return dt

    # Path components: any parent folder matching a hint.
    for part in path.parts:
        key = part.lower()
        if key in _PATH_HINT_TO_DOCTYPE:
            return _PATH_HINT_TO_DOCTYPE[key]

    # Extra hints (source strings, category tags, etc.) — check for any known
    # hint word as a substring/component.
    for hint in hints:
        if not hint:
            continue
        lower = hint.lower().replace("\\", "/")
        for piece in lower.split("/"):
            if piece in _PATH_HINT_TO_DOCTYPE:
                return _PATH_HINT_TO_DOCTYPE[piece]

    # Fallback by format — code files default to CODE; everything else GENERIC.
    code_formats = {
        Format.PYTHON, Format.GO, Format.RUBY, Format.JAVASCRIPT,
        Format.TYPESCRIPT, Format.BASH, Format.DOCKERFILE,
    }
    if fmt in code_formats:
        return DocType.CODE
    return DocType.GENERIC


def classify(
    path: str | Path,
    hints: list[str] | None = None,
) -> FileClassification:
    """Return (format, doc_type) for a given file path.

    Extra `hints` (source string, category name, folder override) help doc_type
    inference when the path itself lacks folder context (e.g. temp files).
    """
    p = Path(path)
    fmt = _infer_format(p)
    doc_type = _infer_doc_type(p, fmt, hints or [])
    return FileClassification(format=fmt, doc_type=doc_type)
