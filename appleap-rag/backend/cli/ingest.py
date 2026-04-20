"""Corpus ingestion CLI.

Walks a file or directory and POSTs each supported file to /ingest/file.
Sends `X-Admin-Token` from the APPLEAP_ADMIN_TOKEN env var so the server's
admin gate accepts the request.

Usage:
    export APPLEAP_ADMIN_TOKEN=<token>
    python -m backend.cli.ingest /path/to/corpus
    python -m backend.cli.ingest /path/to/runbooks --api-url http://localhost:8000

On-prem (docker-compose):
    docker compose exec appleap \\
        python -m backend.cli.ingest /mnt/data/runbooks

Only files with extensions the server knows how to parse are uploaded.
Unsupported files are skipped with a notice (no failure).
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

import httpx


SUPPORTED_EXTENSIONS = {
    # Prose / office
    ".pdf", ".docx", ".doc", ".pptx", ".ppt", ".xlsx", ".xls",
    ".md", ".markdown", ".txt", ".html", ".htm", ".rst", ".xml",
    ".eml", ".rtf",
    # Tabular
    ".csv",
    # IaC / config
    ".tf", ".tfvars", ".hcl", ".yaml", ".yml", ".json", ".pp",
    ".conf",
    # Code
    ".py", ".pyi", ".go", ".rb", ".js", ".mjs", ".cjs", ".jsx",
    ".ts", ".tsx", ".sh", ".bash",
}


def _classify_ext(name: str) -> str:
    lower = name.lower()
    if lower.endswith(".tf.json"):
        return ".tf.json"
    return Path(name).suffix.lower()


def _iter_files(path: Path):
    """Yield every file under `path` (recursively) or `path` itself if it's a file."""
    if path.is_file():
        yield path
        return
    if not path.is_dir():
        return
    for p in sorted(path.rglob("*")):
        if p.is_file():
            yield p


def _is_supported(path: Path) -> bool:
    ext = _classify_ext(path.name)
    return ext in SUPPORTED_EXTENSIONS or ext == ".tf.json"


def upload_file(
    client: httpx.Client,
    api_url: str,
    admin_token: str | None,
    path: Path,
    source: str,
) -> tuple[bool, str]:
    """POST one file to /ingest/file. Returns (ok, message)."""
    headers: dict[str, str] = {}
    if admin_token:
        headers["X-Admin-Token"] = admin_token

    try:
        with open(path, "rb") as fh:
            files = {"file": (path.name, fh, "application/octet-stream")}
            data = {"source": source}
            r = client.post(
                f"{api_url}/ingest/file",
                files=files,
                data=data,
                headers=headers,
                timeout=600.0,
            )
    except Exception as e:
        return False, f"request failed: {e}"

    if r.status_code == 200:
        body = r.json()
        return True, f"ok ({body.get('chunks_stored', '?')} chunks)"

    try:
        detail = r.json().get("detail", r.text)
    except Exception:
        detail = r.text
    return False, f"HTTP {r.status_code}: {detail}"


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Ingest files into the AppLeap RAG corpus",
    )
    parser.add_argument("path", type=Path, help="File or directory to ingest")
    parser.add_argument(
        "--api-url",
        default=os.environ.get("APPLEAP_API_URL", "http://localhost:8000"),
        help="Base URL of the AppLeap RAG API (default: http://localhost:8000)",
    )
    parser.add_argument(
        "--source",
        default="cli",
        help="Value to store in documents.source (default: 'cli')",
    )
    parser.add_argument(
        "--continue-on-error",
        action="store_true",
        help="Keep going after individual file failures",
    )
    args = parser.parse_args()

    if not args.path.exists():
        print(f"Path not found: {args.path}", file=sys.stderr)
        return 1

    admin_token = os.environ.get("APPLEAP_ADMIN_TOKEN")
    if not admin_token:
        print(
            "WARN: APPLEAP_ADMIN_TOKEN not set. "
            "If the server has admin_token configured, uploads will be rejected.",
            file=sys.stderr,
        )

    # Collect files up front so we can print a total
    files = [p for p in _iter_files(args.path) if _is_supported(p)]
    skipped = [p for p in _iter_files(args.path) if not _is_supported(p)]

    if skipped:
        print(f"Skipping {len(skipped)} unsupported file(s)", file=sys.stderr)

    if not files:
        print("No supported files found.", file=sys.stderr)
        return 1

    print(f"Uploading {len(files)} file(s) to {args.api_url}")
    ok_count = 0
    fail_count = 0
    started = time.time()

    with httpx.Client() as client:
        for i, path in enumerate(files, 1):
            ok, msg = upload_file(client, args.api_url, admin_token, path, args.source)
            status = "OK  " if ok else "FAIL"
            print(f"  [{i:>4}/{len(files)}] {status}  {path.name}  {msg}")
            if ok:
                ok_count += 1
            else:
                fail_count += 1
                if not args.continue_on_error:
                    print("Stopping. Pass --continue-on-error to keep going.", file=sys.stderr)
                    break

    elapsed = time.time() - started
    print(
        f"\nDone in {elapsed:.1f}s — {ok_count} ok, {fail_count} failed, "
        f"{len(skipped)} skipped"
    )
    return 0 if fail_count == 0 else 2


if __name__ == "__main__":
    sys.exit(main())
