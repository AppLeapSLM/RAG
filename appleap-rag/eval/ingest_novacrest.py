"""Bulk-ingest NovaCrest synthetic data into the AppLeap RAG system.

Usage:
    python -m eval.ingest_novacrest [--api-url http://localhost:8000] [--data-dir /path/to/novacrest]

Walks the NovaCrest synthetic data directory and ingests every supported file
via the /ingest/file or /ingest endpoint.
"""

import argparse
import sys
import time
from pathlib import Path

import httpx

SUPPORTED_EXTENSIONS = {
    ".md", ".txt", ".html", ".htm", ".csv", ".json", ".xml",
    ".yaml", ".yml", ".pp", ".tf", ".conf",
}

# .tf.json files need special handling (extension is .json but path contains .tf.json)
SKIP_DIRS = {"__pycache__", ".git", "node_modules"}


def find_files(data_dir: Path) -> list[Path]:
    """Walk the data directory and collect all ingestable files."""
    files = []
    for path in sorted(data_dir.rglob("*")):
        if path.is_dir():
            continue
        if any(skip in path.parts for skip in SKIP_DIRS):
            continue
        if path.suffix.lower() in SUPPORTED_EXTENSIONS:
            files.append(path)
        elif path.name.endswith(".tf.json"):
            files.append(path)
    return files


def ingest_file_via_api(client: httpx.Client, api_url: str, filepath: Path) -> dict:
    """Ingest a single file via POST /ingest/file."""
    # Determine the relative category from directory structure
    rel = filepath.relative_to(filepath.parents[1]) if len(filepath.parts) > 2 else filepath
    category = filepath.parent.name

    with open(filepath, "rb") as f:
        files = {"file": (filepath.name, f)}
        data = {
            "source": f"novacrest/{category}",
            "metadata_json": f'{{"category": "{category}", "dataset": "novacrest"}}',
        }
        response = client.post(f"{api_url}/ingest/file", files=files, data=data, timeout=120.0)

    if response.status_code == 200:
        return response.json()
    else:
        return {"error": response.status_code, "detail": response.text}


def ingest_text_via_api(client: httpx.Client, api_url: str, filepath: Path) -> dict:
    """Ingest a file as raw text via POST /ingest (fallback for unsupported extensions)."""
    content = filepath.read_text(encoding="utf-8", errors="replace")
    category = filepath.parent.name

    response = client.post(
        f"{api_url}/ingest",
        json={
            "title": filepath.name,
            "content": content,
            "source": f"novacrest/{category}",
            "metadata": {"category": category, "dataset": "novacrest"},
        },
        timeout=120.0,
    )

    if response.status_code == 200:
        return response.json()
    else:
        return {"error": response.status_code, "detail": response.text}


def main():
    parser = argparse.ArgumentParser(description="Ingest NovaCrest data into AppLeap RAG")
    parser.add_argument(
        "--api-url",
        default="http://localhost:8000",
        help="Base URL of the AppLeap RAG API (default: http://localhost:8000)",
    )
    parser.add_argument(
        "--data-dir",
        default=None,
        help="Path to the NovaCrest synthetic data directory",
    )
    args = parser.parse_args()

    # Resolve data directory
    if args.data_dir:
        data_dir = Path(args.data_dir)
    else:
        # Default: look for the NovaCrest data relative to common locations
        candidates = [
            Path(__file__).resolve().parent.parent.parent
            / "AppLeapv3" / "appleap" / "synthetic_data" / "output" / "novacrest",
            Path.home() / "Desktop" / "AppLeapv3" / "appleap"
            / "synthetic_data" / "output" / "novacrest",
        ]
        data_dir = None
        for c in candidates:
            if c.is_dir():
                data_dir = c
                break
        if not data_dir:
            print("ERROR: Could not find NovaCrest data directory.")
            print("Specify with --data-dir /path/to/novacrest")
            sys.exit(1)

    print(f"Data directory: {data_dir}")
    files = find_files(data_dir)
    print(f"Found {len(files)} files to ingest\n")

    if not files:
        print("No files found. Check the data directory path.")
        sys.exit(1)

    # Check API is reachable
    client = httpx.Client()
    try:
        r = client.get(f"{args.api_url}/health", timeout=5.0)
        r.raise_for_status()
    except Exception as e:
        print(f"ERROR: Cannot reach API at {args.api_url}: {e}")
        sys.exit(1)

    # File extensions supported by /ingest/file (via Unstructured)
    file_api_extensions = {".md", ".txt", ".html", ".htm", ".csv", ".json", ".xml"}

    ingested = 0
    failed = 0
    total_chunks = 0
    start = time.time()

    for i, filepath in enumerate(files, 1):
        ext = filepath.suffix.lower()
        category = filepath.parent.name
        print(f"[{i}/{len(files)}] {category}/{filepath.name}", end=" ... ", flush=True)

        try:
            if ext in file_api_extensions:
                result = ingest_file_via_api(client, args.api_url, filepath)
            else:
                result = ingest_text_via_api(client, args.api_url, filepath)

            if "error" in result:
                print(f"FAILED ({result.get('detail', 'unknown')[:80]})")
                failed += 1
            else:
                chunks = result.get("chunks_stored", 0)
                total_chunks += chunks
                print(f"OK ({chunks} chunks)")
                ingested += 1

        except Exception as e:
            print(f"ERROR: {e}")
            failed += 1

    elapsed = time.time() - start
    print(f"\n{'=' * 60}")
    print(f"Ingestion complete in {elapsed:.1f}s")
    print(f"  Ingested: {ingested}/{len(files)} files")
    print(f"  Failed:   {failed}/{len(files)} files")
    print(f"  Chunks:   {total_chunks} total")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
