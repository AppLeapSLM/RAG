"""Evaluation framework for AppLeap RAG — retrieval quality + answer accuracy.

Usage:
    python -m eval.evaluate [--api-url http://localhost:8000] [--top-k 5] [--verbose]

Runs ground-truth Q&A pairs from questions.json against the system and
measures retrieval quality (Recall@K, MRR) and answer relevance.

Metrics:
    Recall@K: What fraction of expected source documents appeared in the top-K results?
    MRR (Mean Reciprocal Rank): Average of 1/rank for the first correct source per question.
    Hit Rate: Did at least one expected source appear in top-K?
"""

import argparse
import json
import sys
import time
from pathlib import Path

import httpx


def load_questions(path: Path) -> list[dict]:
    """Load Q&A pairs from questions.json."""
    with open(path) as f:
        return json.load(f)


def source_matches(expected_filename: str, actual_preview: str, actual_source: dict) -> bool:
    """Check if a retrieved chunk came from the expected source file.

    Every chunk's content starts with a bracketed header that includes the
    original filename — `[Document: init.pp | ...]` for prose chunks and
    `[File: init.pp | Type: class | ...]` for tree-sitter chunks — so a
    direct substring match on the filename is the most reliable signal.

    The previous token-based heuristic required ≥3 long tokens and
    structurally excluded filenames like init.pp, eks.tf.json,
    api-gateway.yaml, and cmdb-production.csv — every stuck-at-0% category
    in earlier eval runs.
    """
    return expected_filename.lower() in actual_preview.lower()


def evaluate_retrieval(
    client: httpx.Client,
    api_url: str,
    questions: list[dict],
    top_k: int,
    verbose: bool,
) -> dict:
    """Run all questions and compute retrieval metrics."""

    # First, fetch all documents to build a title -> filename mapping
    # We'll use this to match expected_sources to actual retrieved documents
    doc_title_map = {}  # document_id -> title

    results = {
        "total": 0,
        "recall_sum": 0.0,
        "mrr_sum": 0.0,
        "hits": 0,
        "by_category": {},
        "details": [],
    }

    for q in questions:
        qid = q["id"]
        question = q["question"]
        expected_sources = q["expected_sources"]
        category = q["category"]

        if category not in results["by_category"]:
            results["by_category"][category] = {
                "total": 0,
                "recall_sum": 0.0,
                "mrr_sum": 0.0,
                "hits": 0,
            }

        results["total"] += 1
        results["by_category"][category]["total"] += 1

        # Call the query endpoint
        try:
            response = client.post(
                f"{api_url}/query",
                json={"question": question, "top_k": top_k},
                timeout=300.0,
            )
            response.raise_for_status()
            data = response.json()
        except Exception as e:
            print(f"  [{qid}] ERROR: {e}")
            results["details"].append({
                "id": qid, "question": question, "error": str(e),
            })
            continue

        answer = data["answer"]
        sources = data["sources"]

        # For each retrieved source, fetch the document title
        retrieved_titles = []
        for src in sources:
            doc_id = src["document_id"]
            if doc_id not in doc_title_map:
                try:
                    # We need a way to get document titles — use content preview as fallback
                    doc_title_map[doc_id] = src.get("content_preview", "")
                except Exception:
                    pass
            retrieved_titles.append({
                "document_id": doc_id,
                "preview": src.get("content_preview", ""),
                "chunk_index": src.get("chunk_index", -1),
            })

        # Compute Recall@K: what fraction of expected sources were found?
        found_sources = []
        first_rank = None

        for expected in expected_sources:
            expected_lower = expected.lower()
            matched = False
            for rank, rt in enumerate(retrieved_titles, 1):
                # Match by checking if key terms from the expected filename
                # appear in the retrieved content preview
                if source_matches(expected, rt["preview"], rt):
                    if not matched:
                        found_sources.append(expected)
                        matched = True
                        if first_rank is None or rank < first_rank:
                            first_rank = rank
                    break

        recall = len(found_sources) / len(expected_sources) if expected_sources else 0
        mrr = 1.0 / first_rank if first_rank else 0.0
        hit = 1 if found_sources else 0

        results["recall_sum"] += recall
        results["mrr_sum"] += mrr
        results["hits"] += hit

        results["by_category"][category]["recall_sum"] += recall
        results["by_category"][category]["mrr_sum"] += mrr
        results["by_category"][category]["hits"] += hit

        detail = {
            "id": qid,
            "question": question,
            "category": category,
            "difficulty": q.get("difficulty", "unknown"),
            "recall": recall,
            "mrr": mrr,
            "hit": bool(hit),
            "found_sources": found_sources,
            "expected_sources": expected_sources,
            "answer_preview": answer[:200],
        }
        results["details"].append(detail)

        if verbose:
            status = "HIT" if hit else "MISS"
            print(f"  [{qid}] {status} | Recall={recall:.2f} | MRR={mrr:.2f} | {question[:60]}")
            if not hit:
                print(f"         Expected: {expected_sources}")
                print(f"         Got previews: {[t['preview'][:50] for t in retrieved_titles[:3]]}")

    return results


def print_report(results: dict, top_k: int):
    """Print a formatted evaluation report."""
    total = results["total"]
    if total == 0:
        print("No questions evaluated.")
        return

    recall_at_k = results["recall_sum"] / total
    mrr = results["mrr_sum"] / total
    hit_rate = results["hits"] / total

    print(f"\n{'=' * 70}")
    print(f"  EVALUATION REPORT (top_k={top_k}, {total} questions)")
    print(f"{'=' * 70}")
    print(f"\n  Overall Metrics:")
    print(f"    Recall@{top_k}:  {recall_at_k:.3f}  ({results['recall_sum']:.1f}/{total})")
    print(f"    MRR:         {mrr:.3f}")
    print(f"    Hit Rate:    {hit_rate:.3f}  ({results['hits']}/{total})")

    print(f"\n  By Category:")
    print(f"    {'Category':<20} {'Recall@K':>10} {'MRR':>10} {'Hit Rate':>10} {'Count':>8}")
    print(f"    {'-'*20} {'-'*10} {'-'*10} {'-'*10} {'-'*8}")

    for cat, cat_results in sorted(results["by_category"].items()):
        n = cat_results["total"]
        if n == 0:
            continue
        cat_recall = cat_results["recall_sum"] / n
        cat_mrr = cat_results["mrr_sum"] / n
        cat_hit = cat_results["hits"] / n
        print(f"    {cat:<20} {cat_recall:>10.3f} {cat_mrr:>10.3f} {cat_hit:>10.3f} {n:>8}")

    # Show misses
    misses = [d for d in results["details"] if not d.get("hit", False) and "error" not in d]
    if misses:
        print(f"\n  Missed Questions ({len(misses)}):")
        for m in misses:
            print(f"    [{m['id']}] {m['question'][:70]}")
            print(f"            Expected: {m['expected_sources']}")

    print(f"\n{'=' * 70}")


def run_followup_eval(
    client: httpx.Client,
    api_url: str,
    questions: list[dict],
    top_k: int,
    verbose: bool,
) -> dict:
    """Run conversational follow-up questions using conversation memory."""
    followup_questions = [q for q in questions if "followup" in q]

    if not followup_questions:
        return {"total": 0}

    print(f"\n  Conversational Follow-Up Evaluation ({len(followup_questions)} pairs)")
    print(f"  {'-' * 60}")

    results = {"total": 0, "recall_sum": 0.0, "mrr_sum": 0.0, "hits": 0}

    for q in followup_questions:
        # Step 1: Ask the initial question (creates a conversation)
        try:
            r1 = client.post(
                f"{api_url}/query",
                json={"question": q["question"], "top_k": top_k},
                timeout=300.0,
            )
            r1.raise_for_status()
            d1 = r1.json()
            conv_id = d1.get("conversation_id")
        except Exception as e:
            print(f"  [{q['id']}] Initial question ERROR: {e}")
            continue

        # Step 2: Ask the follow-up using the same conversation
        followup = q["followup"]
        try:
            r2 = client.post(
                f"{api_url}/query",
                json={
                    "question": followup["question"],
                    "top_k": top_k,
                    "conversation_id": conv_id,
                },
                timeout=300.0,
            )
            r2.raise_for_status()
            d2 = r2.json()
        except Exception as e:
            print(f"  [{q['id']}] Follow-up ERROR: {e}")
            continue

        # Check if the follow-up retrieved the right sources
        results["total"] += 1
        sources = d2.get("sources", [])
        expected = followup["expected_sources"]

        found = False
        for exp in expected:
            for src in sources:
                if source_matches(exp, src.get("content_preview", ""), src):
                    found = True
                    break
            if found:
                break

        if found:
            results["hits"] += 1
            results["recall_sum"] += 1.0
            results["mrr_sum"] += 1.0

        if verbose:
            status = "HIT" if found else "MISS"
            print(f"  [{q['id']}] {status} | Q1: {q['question'][:40]} -> Q2: {followup['question'][:40]}")

    return results


def main():
    parser = argparse.ArgumentParser(description="Evaluate AppLeap RAG retrieval quality")
    parser.add_argument("--api-url", default="http://localhost:8000")
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--verbose", "-v", action="store_true")
    parser.add_argument("--output", "-o", help="Save detailed results to JSON file")
    parser.add_argument("--skip-followups", action="store_true", help="Skip conversational tests")
    args = parser.parse_args()

    questions_path = Path(__file__).parent / "questions.json"
    if not questions_path.exists():
        print(f"ERROR: {questions_path} not found")
        sys.exit(1)

    questions = load_questions(questions_path)
    print(f"Loaded {len(questions)} evaluation questions")

    # Check API
    client = httpx.Client()
    try:
        r = client.get(f"{args.api_url}/health", timeout=5.0)
        r.raise_for_status()
    except Exception as e:
        print(f"ERROR: Cannot reach API at {args.api_url}: {e}")
        sys.exit(1)

    # Filter out follow-up-only questions for the main eval
    main_questions = [q for q in questions if q["category"] != "followup_only"]

    print(f"\nRunning retrieval evaluation (top_k={args.top_k})...")
    start = time.time()

    results = evaluate_retrieval(client, args.api_url, main_questions, args.top_k, args.verbose)
    elapsed = time.time() - start

    print_report(results, args.top_k)
    print(f"\n  Elapsed: {elapsed:.1f}s ({elapsed / max(results['total'], 1):.1f}s/question)")

    # Run follow-up evaluation if conversation memory is available
    if not args.skip_followups:
        followup_results = run_followup_eval(client, args.api_url, questions, args.top_k, args.verbose)
        if followup_results["total"] > 0:
            n = followup_results["total"]
            hit_rate = followup_results["hits"] / n
            print(f"\n  Conversational Follow-Up Results:")
            print(f"    Hit Rate: {hit_rate:.3f} ({followup_results['hits']}/{n})")

    # Save detailed results
    if args.output:
        output_path = Path(args.output)
        results["elapsed_seconds"] = elapsed
        results["top_k"] = args.top_k
        with open(output_path, "w") as f:
            json.dump(results, f, indent=2)
        print(f"\n  Detailed results saved to {output_path}")


if __name__ == "__main__":
    main()
