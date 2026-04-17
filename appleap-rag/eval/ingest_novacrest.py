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
from urllib.parse import urlparse

import httpx

SUPPORTED_EXTENSIONS = {
    ".md", ".txt", ".html", ".htm", ".csv", ".json", ".xml",
    ".yaml", ".yml", ".pp", ".tf", ".conf",
}

# .tf.json files need special handling (extension is .json but path contains .tf.json)
SKIP_DIRS = {"__pycache__", ".git", "node_modules"}

MAX_RETRIES = 3
RETRY_BACKOFF = [5, 15, 30]  # seconds to wait before each retry


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


def unload_llm(ollama_url: str) -> None:
    """Unload the LLM (phi4) from Ollama to free VRAM for embedding.

    Uses the Ollama generate API with keep_alive=0 to immediately unload.
    The model will auto-reload on the next query request.
    """
    print("Unloading LLM (phi4) to free VRAM for embedding...")
    try:
        with httpx.Client() as client:
            response = client.post(
                f"{ollama_url}/api/generate",
                json={"model": "phi4", "keep_alive": 0},
                timeout=30.0,
            )
            if response.status_code == 200:
                print("  phi4 unloaded successfully.\n")
            else:
                print(f"  Warning: phi4 unload returned {response.status_code} (may not be loaded).\n")
    except Exception as e:
        print(f"  Warning: Could not unload phi4: {e} (continuing anyway).\n")


def ingest_file_via_api(client: httpx.Client, api_url: str, filepath: Path) -> dict:
    """Ingest a single file via POST /ingest/file."""
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


def ingest_one(client: httpx.Client, api_url: str, filepath: Path, file_api_extensions: set) -> dict:
    """Ingest a single file, choosing the right API based on extension."""
    ext = filepath.suffix.lower()
    if ext in file_api_extensions:
        return ingest_file_via_api(client, api_url, filepath)
    else:
        return ingest_text_via_api(client, api_url, filepath)


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
    parser.add_argument(
        "--ollama-url",
        default=None,
        help="Ollama API URL (default: derived from api-url host, port 11434)",
    )
    args = parser.parse_args()

    # Resolve data directory
    if args.data_dir:
        data_dir = Path(args.data_dir)
    else:
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

    # Derive Ollama URL from API URL (same host, port 11434)
    if args.ollama_url:
        ollama_url = args.ollama_url
    else:
        parsed = urlparse(args.api_url)
        ollama_url = f"{parsed.scheme}://{parsed.hostname}:11434"

    # Unload LLM to free VRAM for embedding
    unload_llm(ollama_url)

    # File extensions supported by /ingest/file (via Unstructured)
    # Route every supported file through /ingest/file — that endpoint uses
    # dispatch.process_file, which routes prose → Unstructured and structured
    # formats (YAML, Puppet, Terraform, JSON, CSV, ...) → tree-sitter.
    # The legacy /ingest raw-text endpoint bypasses dispatch, so sending
    # YAML/Puppet/HCL there would silently prose-chunk them.
    file_api_extensions = set(SUPPORTED_EXTENSIONS)

    ingested = 0
    failed_files = []  # (index, filepath) for retry
    total_chunks = 0
    start = time.time()

    # ── First pass ─────────────────────────────────────────────────
    for i, filepath in enumerate(files, 1):
        category = filepath.parent.name
        print(f"[{i}/{len(files)}] {category}/{filepath.name}", end=" ... ", flush=True)

        try:
            result = ingest_one(client, args.api_url, filepath, file_api_extensions)

            if "error" in result:
                print(f"FAILED ({result.get('detail', 'unknown')[:80]})")
                failed_files.append((i, filepath))
            else:
                chunks = result.get("chunks_stored", 0)
                total_chunks += chunks
                print(f"OK ({chunks} chunks)")
                ingested += 1

        except Exception as e:
            print(f"ERROR: {e}")
            failed_files.append((i, filepath))

    # ── Retry failed files ─────────────────────────────────────────
    if failed_files:
        for attempt in range(MAX_RETRIES):
            if not failed_files:
                break

            wait = RETRY_BACKOFF[attempt]
            print(f"\n--- Retry {attempt + 1}/{MAX_RETRIES}: {len(failed_files)} files, waiting {wait}s ---")
            time.sleep(wait)

            # Unload LLM again in case it got reloaded
            unload_llm(ollama_url)

            still_failed = []
            for orig_idx, filepath in failed_files:
                category = filepath.parent.name
                print(f"  [retry {attempt + 1}] {category}/{filepath.name}", end=" ... ", flush=True)

                try:
                    result = ingest_one(client, args.api_url, filepath, file_api_extensions)

                    if "error" in result:
                        print(f"FAILED ({result.get('detail', 'unknown')[:80]})")
                        still_failed.append((orig_idx, filepath))
                    else:
                        chunks = result.get("chunks_stored", 0)
                        total_chunks += chunks
                        print(f"OK ({chunks} chunks)")
                        ingested += 1

                except Exception as e:
                    print(f"ERROR: {e}")
                    still_failed.append((orig_idx, filepath))

            failed_files = still_failed

    elapsed = time.time() - start
    final_failed = len(failed_files)

    print(f"\n{'=' * 60}")
    print(f"Ingestion complete in {elapsed:.1f}s")
    print(f"  Ingested: {ingested}/{len(files)} files")
    print(f"  Failed:   {final_failed}/{len(files)} files (after {MAX_RETRIES} retries)")
    print(f"  Chunks:   {total_chunks} total")

    if failed_files:
        print(f"\n  Permanently failed files:")
        for _, fp in failed_files:
            print(f"    {fp.parent.name}/{fp.name}")

    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
