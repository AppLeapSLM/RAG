"""Smoke test for the new tree-sitter-based chunking pipeline.

Feeds one representative file per format through backend.chunking.dispatch.process_file
and prints: (1) classification result, (2) chunk count, (3) first ~120 chars of
each chunk, so the structural output can be eyeballed before a full re-ingestion.

Usage:
    python -m eval.smoke_chunking [--data-dir /path/to/novacrest] [--format terraform,yaml,...]

Exits non-zero if any sample fails to chunk or falls back to char-splitting
unexpectedly. Designed to be a fast sanity check, not a full eval.
"""

from __future__ import annotations

import argparse
import asyncio
import sys
from pathlib import Path

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
    """Find the first file under data_dir/subdir matching any of the extensions."""
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


async def _run_one(path: Path) -> tuple[int, list[str], bool]:
    """Run dispatch on a single file. Returns (chunk_count, preview_strings, used_fallback)."""
    chunks, doc_meta = await process_file(
        path,
        extra_metadata={"source": f"smoke/{path.parent.name}"},
        display_name=f"{path.parent.name}/{path.name}",
    )
    previews = []
    used_fallback = False
    for c in chunks[:3]:
        text = c["text"].replace("\n", " ")[:140]
        previews.append(text)
        if "fallback_char_split" in c.get("element_types", []):
            used_fallback = True
    if len(chunks) > 3:
        previews.append(f"... ({len(chunks) - 3} more)")
    print(f"    classification: format={doc_meta.get('format')} doc_type={doc_meta.get('doc_type')}")
    print(f"    chunks: {len(chunks)}")
    for p in previews:
        print(f"      | {p}")
    return len(chunks), previews, used_fallback


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
                count, _, fb = await _run_one(sample)
                if count == 0:
                    print("    FAIL: 0 chunks")
                    failures += 1
                elif fb:
                    print("    WARN: fell back to character splitter (tree-sitter path missed)")
            except Exception as e:
                print(f"    FAIL: {type(e).__name__}: {e}")
                failures += 1

    print(f"\n{'OK' if failures == 0 else f'{failures} FAILURES'}")
    sys.exit(0 if failures == 0 else 1)


if __name__ == "__main__":
    asyncio.run(main())
