"""Deterministic natural-language headers for structured chunks.

Every chunk's embedded text starts with:

    [File: <name> | Type: ... | Name: ... | Part i of N]   ← bracket metadata (citation / debug)
    <NL header>                                            ← this module emits this line
    ---
    <raw chunk content>

The NL header is a single English sentence (or two) that describes the chunk
in words matching how a user would ask about it. This aligns Nomic's
embedding of structural data (YAML / HCL / JSON) much closer to the
embedding of a natural-language query.

### Design constraints

- **Zero LLM involvement.** Every string produced here is a template filled
  in from deterministic input — the parsed AST or dict. No summarization,
  no hallucination risk. (Rejecting LLM-generated summaries was the V8
  reason for rejecting NL headers entirely; this module does not violate
  that constraint.)

- **Never crash, never drop a chunk.** All structured parsers (`yaml.safe_load`,
  `json.loads`) are wrapped in try/except. On any parse failure we emit a
  regex-derived fallback like `"YAML configuration fragment with keys: ..."`.

- **Cap verbatim enumeration at MAX_ATTRS.** When we list attribute names
  (e.g., Terraform `.tf.json` resource attributes, Helm top-level keys),
  we stop at MAX_ATTRS with an overflow message to protect the 3000-char
  chunk budget from cascading splits.

- **For YAML + JSON:** parse the raw chunk text with the stdlib parser and
  traverse the resulting dict with a defensive `_dig()` helper. Tree-sitter
  is used *only* to define chunk boundaries; it is not used to extract
  specific keys (that would require brittle S-expression queries into
  YAML's generic `block_mapping_pair` nodes).

- **For Terraform HCL + Puppet:** no viable stdlib parser, so these are
  served by builder functions the chunker calls with captures it already
  has (kind, name, attribute-identifier names).
"""

from __future__ import annotations

import json
import logging
import re
from typing import Any

import yaml

from backend.chunking.doc_type import Format

logger = logging.getLogger(__name__)

# Cap on verbatim enumeration of attribute names / keys. Past this, we emit
# "and N other attributes". Keeps header length bounded so the NL prefix
# doesn't push chunks over the chunk_size cascade threshold.
MAX_ATTRS = 15


# ── Public entrypoints ───────────────────────────────────────────────


def build_nl_header(
    fmt: Format,
    raw_text: str,
    metadata: dict[str, Any] | None = None,
) -> str:
    """Top-level dispatch for YAML / JSON headers (stdlib parse path).

    `metadata` is checked for a `doc_type` hint (from classify()) — we trust
    that over content-based heuristics to disambiguate cases like Helm vs
    Docker Compose, which can look similar structurally.

    HCL and Puppet use `build_hcl_header` / `build_puppet_header` directly
    from the chunker because they rely on tree-sitter captures rather than
    stdlib parsing.
    """
    metadata = metadata or {}
    doc_type = metadata.get("doc_type")
    try:
        if fmt == Format.YAML:
            return _yaml_header(raw_text, doc_type=doc_type)
        if fmt == Format.JSON:
            return _json_header(raw_text, is_terraform=False)
        if fmt == Format.TERRAFORM:
            # Terraform JSON (.tf.json) routes through here — HCL does not.
            return _json_header(raw_text, is_terraform=True)
    except Exception as exc:  # noqa: BLE001 — defensive by design
        logger.warning(
            "NL header build failed for %s: %s — falling back", fmt.value, exc
        )
    return _fallback_header(raw_text, fmt)


def build_hcl_header(
    kind: str | None,
    name: str | None,
    attribute_names: list[str] | None = None,
) -> str:
    """Terraform HCL header from tree-sitter captures.

    `kind` is the block type (`resource`, `module`, `data`, `variable`,
    `provider`, `output`); `name` is the first string label (e.g.
    `aws_eks_cluster`); `attribute_names` enumerates top-level attribute
    identifiers inside the block body (capped at MAX_ATTRS).
    """
    parts: list[str] = ["Terraform"]
    if kind:
        parts.append(kind)
    if name:
        parts.append(f'"{name}"')
    header = " ".join(parts) + "."
    if attribute_names:
        header += _enumerate_attributes(attribute_names)
    return header


def build_puppet_header(
    kind: str | None,
    name: str | None,
    parameters: list[str] | None = None,
    resource_types: list[str] | None = None,
    has_ordering: bool = False,
    includes: list[str] | None = None,
) -> str:
    """Puppet header from tree-sitter captures.

    `kind` is `class` / `node` / `resource`; `name` is the declaration's
    identifier; the rest are optional feature flags derived from the AST.
    """
    parts = [f'Puppet {kind or "block"}']
    if name:
        parts.append(f'"{name}"')
    header = " ".join(parts)

    extras: list[str] = []
    if parameters:
        shown = parameters[:MAX_ATTRS]
        more = len(parameters) - len(shown)
        phrase = "parameters: " + ", ".join(shown)
        if more > 0:
            phrase += f" and {more} others"
        extras.append(phrase)
    if resource_types:
        unique = sorted(set(resource_types))[:MAX_ATTRS]
        extras.append("declares resources: " + ", ".join(unique))
    if includes:
        shown = includes[:MAX_ATTRS]
        extras.append("includes: " + ", ".join(shown))
    if has_ordering:
        extras.append("uses dependency chain ordering")

    if extras:
        header += " with " + "; ".join(extras)
    return header + "."


# ── Defensive dict traversal ─────────────────────────────────────────


def _dig(data: Any, *keys: str, default: Any = None) -> Any:
    """Walk a nested dict safely. Returns `default` on any broken path
    (missing key, non-dict intermediate, None partway down).
    """
    cur = data
    for key in keys:
        if not isinstance(cur, dict):
            return default
        cur = cur.get(key)
        if cur is None:
            return default
    return cur


# ── YAML ─────────────────────────────────────────────────────────────


def _yaml_header(raw_text: str, doc_type: str | None = None) -> str:
    """Parse YAML (possibly multi-doc) and build a header for the main doc.

    `safe_load_all` handles both single-doc chunks and parent-scalar-injected
    multi-doc chunks (see code_chunker._split_yaml_document). For multi-doc
    chunks we merge keys from the smaller doc(s) into a context dict so the
    NL description can reference the full picture.

    `doc_type` from classify() is the tiebreaker when multiple heuristics
    could fire — Helm values with a nested `services:` key would otherwise
    be misidentified as a Docker Compose file.
    """
    docs = [d for d in yaml.safe_load_all(raw_text) if d is not None]

    if not docs:
        return _fallback_header(raw_text, Format.YAML)

    main = max(docs, key=lambda d: len(str(d)))
    merged_context: dict[str, Any] = {}
    for d in docs:
        if d is main or not isinstance(d, dict):
            continue
        for k, v in d.items():
            merged_context.setdefault(k, v)

    if not isinstance(main, dict):
        return _fallback_header(raw_text, Format.YAML)

    # Explicit doc_type hint wins over content-based heuristics.
    if doc_type == "kubernetes_manifest":
        return _k8s_header(main, merged_context)
    if doc_type == "docker_compose":
        return _compose_header(main, merged_context)
    if doc_type == "helm_values":
        return _helm_header(main, merged_context)

    kind = main.get("kind") or merged_context.get("kind")
    has_api_version = "apiVersion" in main or "apiVersion" in merged_context
    if kind and has_api_version:
        return _k8s_header(main, merged_context)

    # Content-based compose detection requires both a `services:` dict AND
    # service bodies that look compose-shaped (image/build), AND not
    # Helm-shaped (replicas/replicaCount) — this is the fallback path when
    # no doc_type was provided.
    if _looks_like_compose(main):
        return _compose_header(main, merged_context)

    return _yaml_generic_header(main)


def _looks_like_compose(data: dict) -> bool:
    """Heuristic: presence of a `services:` mapping whose values have
    Compose-specific fields (`image`/`build`) but not Helm-specific ones
    (`replicaCount`/`replicas`).
    """
    services = data.get("services")
    if not isinstance(services, dict) or not services:
        return False

    compose_signal = False
    for svc in services.values():
        if not isinstance(svc, dict):
            continue
        if "replicaCount" in svc or "replicas" in svc:
            return False  # Helm / K8s shape
        if "image" in svc or "build" in svc:
            compose_signal = True
    return compose_signal


def _k8s_header(data: dict, parent_ctx: dict) -> str:
    kind = data.get("kind") or parent_ctx.get("kind", "resource")
    metadata = data.get("metadata")
    if not isinstance(metadata, dict):
        metadata = parent_ctx.get("metadata") if isinstance(parent_ctx.get("metadata"), dict) else {}

    name = metadata.get("name", "<unnamed>")
    namespace = metadata.get("namespace")

    parts = [f'Kubernetes {kind} "{name}"']
    if namespace:
        parts.append(f'in namespace "{namespace}"')
    header = " ".join(parts) + "."

    features = _k8s_features_for_kind(kind, data)
    if features:
        header += " Configures: " + ", ".join(features) + "."
    return header


def _k8s_features_for_kind(kind: str, data: dict) -> list[str]:
    kind_lower = (kind or "").lower()
    features: list[str] = []

    if kind_lower in ("deployment", "statefulset", "daemonset"):
        if _dig(data, "spec", "replicas") is not None:
            features.append("replica count")
        if _dig(data, "spec", "strategy") is not None:
            features.append("rollout strategy")

        containers = _dig(data, "spec", "template", "spec", "containers", default=[])
        if isinstance(containers, list):
            features.extend(_container_features(containers))

        if _dig(data, "spec", "template", "spec", "volumes") is not None:
            features.append("mounted volumes")
        if _dig(data, "spec", "template", "spec", "affinity") is not None:
            features.append("pod affinity rules")
        if _dig(data, "spec", "template", "spec", "serviceAccountName") is not None:
            features.append("service account")

    elif kind_lower == "horizontalpodautoscaler":
        if _dig(data, "spec", "minReplicas") is not None:
            features.append("minimum replica count")
        if _dig(data, "spec", "maxReplicas") is not None:
            features.append("maximum replica count")

        metrics = _dig(data, "spec", "metrics", default=[])
        if isinstance(metrics, list) and metrics:
            metric_types: set[str] = set()
            for m in metrics:
                rname = _dig(m, "resource", "name")
                if rname:
                    metric_types.add(str(rname))
            if metric_types:
                features.append(
                    f"autoscaling on {', '.join(sorted(metric_types))} metrics"
                )
            else:
                features.append("autoscaling metrics")

    elif kind_lower == "service":
        svc_type = _dig(data, "spec", "type", default="ClusterIP")
        features.append(f"type {svc_type}")
        ports = _dig(data, "spec", "ports", default=[])
        if isinstance(ports, list) and ports:
            features.append(f"{len(ports)} exposed port(s)")

    elif kind_lower in ("configmap", "secret"):
        data_keys = data.get("data") if isinstance(data.get("data"), dict) else {}
        if data_keys:
            key_names = list(data_keys.keys())
            shown = key_names[:MAX_ATTRS]
            more = len(key_names) - len(shown)
            phrase = "keys: " + ", ".join(shown)
            if more > 0:
                phrase += f" and {more} others"
            features.append(phrase)

    elif kind_lower == "ingress":
        rules = _dig(data, "spec", "rules", default=[])
        if isinstance(rules, list):
            hosts = [
                r.get("host") for r in rules
                if isinstance(r, dict) and r.get("host")
            ]
            if hosts:
                features.append(f"hostnames {', '.join(hosts[:3])}")

    elif kind_lower in ("cronjob", "job"):
        schedule = _dig(data, "spec", "schedule")
        if schedule:
            features.append(f"schedule {schedule}")

    return features


def _container_features(containers: list) -> list[str]:
    """Aggregate container-level features across a pod spec's containers."""
    has: dict[str, bool] = {
        "resources": False,
        "liveness": False,
        "readiness": False,
        "startup": False,
        "env": False,
        "envFrom": False,
    }
    for c in containers:
        if not isinstance(c, dict):
            continue
        if c.get("resources"):
            has["resources"] = True
        if c.get("livenessProbe"):
            has["liveness"] = True
        if c.get("readinessProbe"):
            has["readiness"] = True
        if c.get("startupProbe"):
            has["startup"] = True
        if c.get("env"):
            has["env"] = True
        if c.get("envFrom"):
            has["envFrom"] = True

    out: list[str] = []
    if has["resources"]:
        out.append("container CPU and memory requests and limits")
    if has["liveness"]:
        out.append("liveness probe")
    if has["readiness"]:
        out.append("readiness probe")
    if has["startup"]:
        out.append("startup probe")
    if has["env"]:
        out.append("environment variables")
    if has["envFrom"]:
        out.append("config map references")
    return out


def _compose_header(data: dict, ctx: dict | None = None) -> str:
    """Docker Compose header, tolerant of oversize-split chunks.

    After _split_yaml_document splits a large `services:` section, the
    service entries appear at the root of the body doc — `services:` is
    no longer present as a wrapper key. We recover service names via:
      1. An explicit `services:` dict in main, if present (pristine chunk)
      2. Otherwise, treat top-level dict entries as services, filtering out
         compose-reserved root keywords (version / volumes / networks / etc)
      3. Plus any `services:` dict preserved in the context half of a
         split chunk

    Approach (2) is a blacklist rather than a positive signature (matching
    on `image`/`build`) because we're already dispatching based on
    doc_type=docker_compose — we trust the classification and don't need
    to second-guess the content.
    """
    ctx = ctx or {}

    # Reserved top-level Compose spec keys — anything else at root is
    # presumed to be a service entry after a split.
    COMPOSE_RESERVED = {
        "version", "volumes", "networks", "configs", "secrets",
        "x-", "services",  # services itself is handled explicitly above
    }

    service_dicts: list[tuple[str, dict]] = []
    seen: set[str] = set()

    def _add(k: str, v: Any):
        if k in seen or not isinstance(v, dict):
            return
        service_dicts.append((k, v))
        seen.add(k)

    # Source 1: explicit services: mapping in main
    svc_map = data.get("services") if isinstance(data.get("services"), dict) else None
    if svc_map:
        for k, v in svc_map.items():
            _add(k, v)
    else:
        # Source 2: top-level keys in main minus compose-reserved keywords.
        for k, v in data.items():
            key_str = str(k)
            if key_str in COMPOSE_RESERVED or key_str.startswith("x-"):
                continue
            _add(key_str, v)

    # Source 3: services dict in context (in case split moved services out)
    ctx_svc_map = ctx.get("services") if isinstance(ctx.get("services"), dict) else None
    if ctx_svc_map:
        for k, v in ctx_svc_map.items():
            _add(k, v)

    svc_names = [k for k, _ in service_dicts]
    shown = svc_names[:MAX_ATTRS]
    more = len(svc_names) - len(shown)

    header = "Docker Compose configuration"
    if shown:
        header += " with services: " + ", ".join(shown)
        if more > 0:
            header += f" and {more} others"
    else:
        return "Docker Compose configuration fragment."

    feats: list[str] = []
    svc_values = [v for _, v in service_dicts]
    if any(s.get("ports") for s in svc_values):
        feats.append("port mappings")
    if any(s.get("environment") for s in svc_values):
        feats.append("environment variables")
    if any(s.get("volumes") for s in svc_values):
        feats.append("mounted volumes")
    if any(s.get("depends_on") for s in svc_values):
        feats.append("service dependencies")

    if feats:
        header += ". Defines " + ", ".join(feats)
    return header + "."


def _yaml_generic_header(data: dict) -> str:
    """Catch-all YAML header — just lists top-level keys.

    Intentionally minimal. Used when no more specific shape is recognized
    (neither K8s nor compose nor Helm).
    """
    keys = list(data.keys())
    shown = keys[:MAX_ATTRS]
    more = len(keys) - len(shown)
    if not shown:
        return "YAML configuration with no top-level keys."
    header = "YAML configuration with top-level keys: " + ", ".join(shown)
    if more > 0:
        header += f" and {more} others"
    return header + "."


def _helm_header(data: dict, ctx: dict) -> str:
    """Helm values-file header.

    Helm charts typically nest microservice configs under a `services:` key
    or put them at the top level. Either way, the NL header should name the
    actual service components so queries like "does the api-gateway have
    resource limits" embed near the right chunk.
    """
    services = data.get("services") if isinstance(data.get("services"), dict) else None
    component_names: list[str] = list(services.keys()) if services else []
    # Merge in top-level keys that aren't the `services:` wrapper itself.
    for k in data.keys():
        if k == "services":
            continue
        if k not in component_names:
            component_names.append(k)
    # Also pull context-level keys (e.g. global / ingress preserved as small
    # pairs during oversize splits) so every split chunk reports the whole
    # file's component shape.
    for k in ctx.keys():
        if k not in component_names:
            component_names.append(k)

    shown = component_names[:MAX_ATTRS]
    more = len(component_names) - len(shown)
    if not shown:
        return "Helm values configuration."
    header = "Helm values for components: " + ", ".join(shown)
    if more > 0:
        header += f" and {more} others"
    return header + "."


# ── JSON ─────────────────────────────────────────────────────────────


def _json_header(raw_text: str, *, is_terraform: bool) -> str:
    data = json.loads(raw_text)

    if is_terraform and isinstance(data, dict):
        for block_type in ("resource", "module", "data", "variable", "provider", "output"):
            if block_type in data:
                return _tf_json_header(block_type, data[block_type])
        # `.tf.json` fragment that doesn't have a known block key — fall through
        # to generic description.

    if isinstance(data, dict):
        keys = list(data.keys())
        shown = keys[:MAX_ATTRS]
        more = len(keys) - len(shown)
        header = "JSON object with keys: " + ", ".join(shown)
        if more > 0:
            header += f" and {more} others"
        return header + "."

    if isinstance(data, list):
        return f"JSON array with {len(data)} element(s)."

    return "JSON scalar value."


def _tf_json_header(block_type: str, body: Any) -> str:
    """Describe one Terraform JSON block, e.g. `{"aws_eks_cluster": {"main": {...}}}`.

    Structure: `{block_type: {resource_type: {resource_name: {...attrs}}}}`.
    We drill two levels to find the resource_type and resource_name, then
    enumerate the attribute keys.
    """
    if not isinstance(body, dict):
        return f"Terraform {block_type} block."

    resource_type: str | None = None
    resource_name: str | None = None
    attrs: list[str] = []

    for rtype, rbody in body.items():
        resource_type = rtype
        if isinstance(rbody, dict):
            for rname, rattrs in rbody.items():
                resource_name = rname
                if isinstance(rattrs, dict):
                    attrs = list(rattrs.keys())
                break
        break

    parts = [f"Terraform {block_type}"]
    if resource_type:
        parts.append(resource_type)
    if resource_name:
        parts.append(f'"{resource_name}"')
    header = " ".join(parts) + "."

    if attrs:
        header += _enumerate_attributes(attrs)
    return header


# ── Helpers ──────────────────────────────────────────────────────────


def _enumerate_attributes(attrs: list[str]) -> str:
    """Format a list of attribute names for a header, capped at MAX_ATTRS."""
    if not attrs:
        return ""
    shown = attrs[:MAX_ATTRS]
    more = len(attrs) - len(shown)
    out = " Attributes: " + ", ".join(shown)
    if more > 0:
        out += f" and {more} other attributes"
    return out + "."


# Regex for fallback: a key-ish identifier at the start of a line followed by
# a colon (YAML) or a quoted key followed by a colon (JSON). Catches the
# obvious top-level keys without needing a full parse.
_TOP_KEY_RE = re.compile(
    r"^(?:\s*)(?:\"([^\"]+)\"|([A-Za-z_][A-Za-z0-9_-]*))\s*:",
    re.MULTILINE,
)


def _fallback_header(raw_text: str, fmt: Format) -> str:
    """Regex-based fallback for any structured format whose parser failed.

    Never empty — always returns a usable sentence with whatever keys we
    could eyeball from the raw text.
    """
    keys: list[str] = []
    seen: set[str] = set()
    for match in _TOP_KEY_RE.finditer(raw_text):
        key = match.group(1) or match.group(2)
        if key and key not in seen:
            seen.add(key)
            keys.append(key)
        if len(keys) >= MAX_ATTRS:
            break

    label = {
        Format.YAML: "YAML",
        Format.JSON: "JSON",
        Format.TERRAFORM: "Terraform",
        Format.PUPPET: "Puppet",
    }.get(fmt, fmt.value.capitalize())

    if keys:
        return f"{label} configuration fragment with keys: {', '.join(keys)}."
    return f"{label} configuration fragment."
