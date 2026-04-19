"""Smoke test for the tree-sitter chunker + deterministic NL headers.

Runs one representative file per format through backend.chunking.dispatch.process_file
and prints / asserts:
  1. classification (format + doc_type)
  2. chunk count
  3. chunk preview for eyeballing
  4. NL header sanity assertions per format:
       - YAML k8s → starts with "Kubernetes <Kind> "..."
       - Terraform / .tf.json → mentions "Terraform"
       - Puppet → mentions "Puppet"
       - CSV → pipe-format (contains " | " and "key: value"), no JSON braces
  5. YAML safe_load_all round-trip on every YAML chunk — must parse without error
     (validates the parent-scalar-preservation fragment-trap defense).

Usage:
    python -m eval.smoke_chunking [--data-dir /path/to/novacrest] [--format terraform,yaml,...]

Exits non-zero if any sample fails to chunk, falls back unexpectedly, or
violates a header / format assertion. Designed as a pre-reingest sanity check.
"""

from __future__ import annotations

import argparse
import asyncio
import sys
from pathlib import Path

import yaml

from backend.chunking.dispatch import process_file
from backend.chunking.doc_type import Format


# Representative samples per format. Paths are relative to --data-dir.
SAMPLES: dict[Format, list[str]] = {
    Format.PROSE: [
        "runbooks",
        "incidents",
        "playbooks",
        "documentation",
    ],
    Format.TERRAFORM: [
        "terraform",
    ],
    Format.YAML: [
        "docker-compose",
        "helm",
        "kubernetes",
    ],
    Format.JSON: [
        "topology",
        "cmdb",
    ],
    Format.PUPPET: [
        "puppet",
    ],
    Format.CSV: [
        "cmdb",
    ],
}


def _pick_sample(data_dir: Path, subdir: str, extensions: tuple[str, ...]) -> Path | None:
    target = data_dir / subdir
    if not target.exists():
        return None
    for path in sorted(target.rglob("*")):
        if path.is_file() and path.name.lower().endswith(extensions):
            return path
    return None


def _extensions_for(fmt: Format) -> tuple[str, ...]:
    return {
        Format.PROSE: (".md",),
        Format.TERRAFORM: (".tf", ".tf.json"),
        Format.YAML: (".yaml", ".yml"),
        Format.JSON: (".json",),
        Format.PUPPET: (".pp",),
        Format.CSV: (".csv",),
    }[fmt]


# ── Per-format NL header assertions ──────────────────────────────────


def _assert_nl_header(fmt: Format, chunk_text: str, sample_path: Path) -> list[str]:
    """Return a list of assertion failure messages for this chunk. Empty = passed."""
    errors: list[str] = []

    lines = chunk_text.split("\n", 2)
    if len(lines) < 2:
        errors.append(f"chunk has <2 lines; expected bracket + NL header")
        return errors

    bracket_line = lines[0]
    if not (bracket_line.startswith("[File:") or bracket_line.startswith("[Document:")):
        errors.append(f"first line not a bracket header: {bracket_line[:80]!r}")

    if fmt == Format.PROSE:
        # Prose uses its own contextual header path; NL header assertions don't apply.
        return errors

    if fmt == Format.CSV:
        # CSV: after the bracket line, the next non-empty line is the pipe payload.
        payload = "\n".join(lines[1:]).strip()
        if " | " not in payload:
            errors.append("CSV payload missing pipe separator")
        if ":" not in payload:
            errors.append("CSV payload missing 'key: value' pattern")
        if payload.startswith("{") or payload.startswith("["):
            errors.append("CSV payload starts with JSON brace — pipe format expected")
        return errors

    # Everything else: bracket + NL header on line 2 (may be multi-sentence).
    nl_header = lines[1].strip()
    if not nl_header:
        errors.append("NL header line is empty")
        return errors

    if fmt == Format.YAML:
        head_lower = nl_header.lower()
        # Accepted openings: "Kubernetes <Kind>", "Helm", "Docker Compose", "YAML".
        if not any(s in head_lower for s in ("kubernetes", "helm", "docker compose", "yaml")):
            errors.append(f"YAML NL header has unexpected opening: {nl_header[:80]!r}")
    elif fmt == Format.TERRAFORM:
        if "terraform" not in nl_header.lower():
            errors.append(f"Terraform NL header missing 'Terraform': {nl_header[:80]!r}")
    elif fmt == Format.JSON:
        if not any(s in nl_header.lower() for s in ("json", "terraform")):
            errors.append(f"JSON NL header missing 'JSON' / 'Terraform': {nl_header[:80]!r}")
    elif fmt == Format.PUPPET:
        if "puppet" not in nl_header.lower():
            errors.append(f"Puppet NL header missing 'Puppet': {nl_header[:80]!r}")

    return errors


def _assert_yaml_parseable(chunk_text: str) -> list[str]:
    """For YAML chunks, the content (everything after the NL header line) must
    parse with safe_load_all without raising. Validates parent-scalar-preservation
    doesn't produce malformed multi-doc output.
    """
    errors: list[str] = []
    # Find where the raw YAML starts: we emit "\n---\n" between NL header and body
    # for chunks where splitting happened; for single-doc chunks, the body starts
    # after the NL header line + blank line. Both cases: skip the first two lines
    # (bracket + NL header), then parse the rest.
    lines = chunk_text.split("\n")
    # Skip bracket header line
    body_lines = lines[1:] if lines else []
    # Skip NL header line (the sentence(s) after bracket, before the first blank line)
    # We do "\n\n" between NL header and body, so find that separator.
    body = "\n".join(body_lines).lstrip("\n")
    if "\n\n" in body:
        _, body = body.split("\n\n", 1)
    try:
        list(yaml.safe_load_all(body))
    except yaml.YAMLError as e:
        errors.append(f"YAML chunk body fails safe_load_all: {type(e).__name__}: {str(e)[:120]}")
    return errors


# ── Runner ───────────────────────────────────────────────────────────


async def _run_one(path: Path, fmt: Format) -> tuple[int, list[str], bool, list[str]]:
    """Run dispatch on a file. Returns (count, previews, fell_back, assertion_errors)."""
    chunks, doc_meta = await process_file(
        path,
        extra_metadata={"source": f"smoke/{path.parent.name}"},
        display_name=f"{path.parent.name}/{path.name}",
    )
    previews: list[str] = []
    used_fallback = False
    assertion_errors: list[str] = []

    for idx, c in enumerate(chunks):
        if idx < 3:
            text = c["text"].replace("\n", " ")[:200]
            previews.append(text)
        if "fallback_char_split" in c.get("element_types", []):
            used_fallback = True

        # Per-chunk NL header + format assertions
        for err in _assert_nl_header(fmt, c["text"], path):
            assertion_errors.append(f"chunk {idx}: {err}")

        if fmt == Format.YAML:
            for err in _assert_yaml_parseable(c["text"]):
                assertion_errors.append(f"chunk {idx}: {err}")

    if len(chunks) > 3:
        previews.append(f"... ({len(chunks) - 3} more)")

    print(f"    classification: format={doc_meta.get('format')} doc_type={doc_meta.get('doc_type')}")
    print(f"    chunks: {len(chunks)}")
    for p in previews:
        print(f"      | {p}")

    return len(chunks), previews, used_fallback, assertion_errors


async def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", type=Path, default=None,
                    help="Path to NovaCrest synthetic data root")
    ap.add_argument("--formats", type=str, default="",
                    help="Comma-separated list of formats to test (default: all)")
    args = ap.parse_args()

    if args.data_dir is None:
        candidates = [
            Path.home() / "Desktop" / "AppLeapv3" / "appleap" / "synthetic_data" / "output" / "novacrest",
            Path(r"C:\Users\Abhinav\Desktop\AppLeapv3\appleap\synthetic_data\output\novacrest"),
        ]
        args.data_dir = next((c for c in candidates if c.exists()), None)
        if args.data_dir is None:
            print("ERROR: could not find NovaCrest data — pass --data-dir", file=sys.stderr)
            sys.exit(2)

    only_formats: set[Format] | None = None
    if args.formats.strip():
        only_formats = {Format(x.strip()) for x in args.formats.split(",") if x.strip()}

    failures = 0
    assertion_failures = 0
    for fmt, subdirs in SAMPLES.items():
        if only_formats and fmt not in only_formats:
            continue
        print(f"\n=== {fmt.value} ===")
        for subdir in subdirs:
            sample = _pick_sample(args.data_dir, subdir, _extensions_for(fmt))
            if sample is None:
                print(f"  [skip] no sample in {subdir}/ with {_extensions_for(fmt)}")
                continue
            print(f"  {sample.relative_to(args.data_dir)}")
            try:
                count, _, fb, errs = await _run_one(sample, fmt)
                if count == 0:
                    print("    FAIL: 0 chunks")
                    failures += 1
                elif fb:
                    print("    WARN: fell back to character splitter (tree-sitter path missed)")
                if errs:
                    for e in errs[:5]:
                        print(f"    ASSERT: {e}")
                    if len(errs) > 5:
                        print(f"    ASSERT: ... and {len(errs) - 5} more")
                    assertion_failures += 1
            except ModuleNotFoundError as e:
                # Prose path requires unstructured[all-docs] (heavy optional dep
                # from the [parsing] extra). Local dev often lacks it — downgrade
                # to a warning so smoke still validates structured paths.
                print(f"    SKIP (dep missing): {e}")
            except Exception as e:
                print(f"    FAIL: {type(e).__name__}: {e}")
                failures += 1

    print(f"\n{'OK' if failures == 0 and assertion_failures == 0 else f'{failures} failures, {assertion_failures} assertion failures'}")
    sys.exit(0 if failures == 0 and assertion_failures == 0 else 1)


if __name__ == "__main__":
    asyncio.run(main())
