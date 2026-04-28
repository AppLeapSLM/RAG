"""Evaluation framework for AppLeap RAG — retrieval quality + answer accuracy.

Usage:
    python -m eval.evaluate [--api-url http://localhost:8000] [--top-k 5] [--verbose]

Loads two question sets:
    - questions.json: single-turn Q&A pairs (retrieval scoring)
    - conversations.json: multi-turn traces (conversation-memory scoring)

Categories:
    - normal categories (incident, runbook, ...): scored on retrieval
        Recall@K = fraction of expected_sources found in top-K
        MRR = 1/rank of first matching source
    - refusal: scored on whether the answer matches a refusal pattern
        ("could not find", "not in the documents", etc.) — verifies that the
        system declines to fabricate when no relevant content is in the corpus.
    - aggregation: scored like normal but expected_sources may include the
        same file multiple times implicitly (multi-row recall on a single CSV).

Conversational eval walks each trace turn-by-turn, threading conversation_id
so the system sees full history. Each turn is scored independently on retrieval.
"""

import argparse
import json
import re
import sys
import time
from pathlib import Path

import httpx


REFUSAL_PATTERNS = [
    r"could not find",
    r"i don'?t have",
    r"not in the (?:available |provided )?documents",
    r"no (?:relevant )?information",
    r"unable to (?:find|locate)",
    r"cannot (?:find|locate|answer)",
    r"don'?t have (?:that |this |any )?information",
    r"no (?:matching|specific) (?:information|documents|details)",
]
REFUSAL_RE = re.compile("|".join(REFUSAL_PATTERNS), re.IGNORECASE)


def load_questions(path: Path) -> list[dict]:
    """Load single-turn Q&A pairs from questions.json."""
    with open(path) as f:
        return json.load(f)


def load_conversations(path: Path) -> list[dict]:
    """Load multi-turn conversation traces from conversations.json."""
    with open(path) as f:
        return json.load(f)


def is_refusal_response(answer: str) -> bool:
    """True if the answer matches any known refusal pattern."""
    return bool(REFUSAL_RE.search(answer))


def source_matches(expected_filename: str, actual_preview: str) -> bool:
    """Substring match on the filename in the chunk's content preview.

    Every chunk's content starts with a bracketed header that includes the
    original filename — `[Document: init.pp | ...]` for prose chunks and
    `[File: init.pp | Type: class | ...]` for tree-sitter chunks.
    """
    return expected_filename.lower() in actual_preview.lower()


def score_retrieval(expected_sources: list[str], retrieved: list[dict]) -> tuple[float, float, int]:
    """Compute recall, MRR, and hit (0/1) for one question's retrieval.

    Returns: (recall, mrr, hit)
    """
    if not expected_sources:
        return 0.0, 0.0, 0

    found = 0
    first_rank = None
    for expected in expected_sources:
        for rank, src in enumerate(retrieved, 1):
            preview = src.get("content_preview", "")
            if source_matches(expected, preview):
                found += 1
                if first_rank is None or rank < first_rank:
                    first_rank = rank
                break

    recall = found / len(expected_sources)
    mrr = 1.0 / first_rank if first_rank else 0.0
    hit = 1 if found > 0 else 0
    return recall, mrr, hit


def evaluate_retrieval(
    client: httpx.Client,
    api_url: str,
    questions: list[dict],
    top_k: int,
    verbose: bool,
) -> dict:
    """Run all single-turn questions and compute per-category metrics."""
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
        expected_sources = q.get("expected_sources", [])
        category = q["category"]

        cat = results["by_category"].setdefault(category, {
            "total": 0, "recall_sum": 0.0, "mrr_sum": 0.0, "hits": 0,
        })

        results["total"] += 1
        cat["total"] += 1

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
            results["details"].append({"id": qid, "question": question, "error": str(e)})
            continue

        answer = data["answer"]
        sources = data["sources"]

        # Refusal questions are scored on the answer pattern, not retrieval
        if category == "refusal":
            refused = is_refusal_response(answer)
            recall, mrr, hit = (1.0, 1.0, 1) if refused else (0.0, 0.0, 0)
        else:
            recall, mrr, hit = score_retrieval(expected_sources, sources)

        results["recall_sum"] += recall
        results["mrr_sum"] += mrr
        results["hits"] += hit
        cat["recall_sum"] += recall
        cat["mrr_sum"] += mrr
        cat["hits"] += hit

        detail = {
            "id": qid,
            "question": question,
            "category": category,
            "difficulty": q.get("difficulty", "unknown"),
            "recall": recall,
            "mrr": mrr,
            "hit": bool(hit),
            "expected_sources": expected_sources,
            "answer_preview": answer[:200],
        }
        if category != "refusal":
            detail["found_sources"] = [
                e for e in expected_sources
                if any(source_matches(e, s.get("content_preview", "")) for s in sources)
            ]
        else:
            detail["refused"] = is_refusal_response(answer)
        results["details"].append(detail)

        if verbose:
            status = "HIT" if hit else "MISS"
            print(f"  [{qid}] {status} | Recall={recall:.2f} | MRR={mrr:.2f} | {question[:60]}")
            if not hit and category != "refusal":
                print(f"         Expected: {expected_sources}")
                print(f"         Got previews: {[s.get('content_preview', '')[:50] for s in sources[:3]]}")
            elif not hit and category == "refusal":
                print(f"         Expected refusal, got: {answer[:120]}")

    return results


def evaluate_conversations(
    client: httpx.Client,
    api_url: str,
    conversations: list[dict],
    top_k: int,
    verbose: bool,
) -> dict:
    """Walk each conversation turn-by-turn, threading conversation_id."""
    results = {
        "total_traces": 0,
        "total_turns": 0,
        "turn_recall_sum": 0.0,
        "turn_mrr_sum": 0.0,
        "turn_hits": 0,
        "trace_details": [],
    }

    for conv in conversations:
        cid = conv["id"]
        category = conv.get("category", "unknown")
        tests = conv.get("tests", "")
        turns = conv["turns"]

        results["total_traces"] += 1
        conversation_id = None
        trace_detail = {
            "id": cid,
            "category": category,
            "tests": tests,
            "turns": [],
            "all_hit": True,
        }

        if verbose:
            print(f"\n  [{cid}] {category} ({tests}) — {len(turns)} turns")

        for i, turn in enumerate(turns, 1):
            question = turn["question"]
            expected_sources = turn.get("expected_sources", [])

            payload = {"question": question, "top_k": top_k}
            if conversation_id is not None:
                payload["conversation_id"] = conversation_id

            try:
                response = client.post(f"{api_url}/query", json=payload, timeout=300.0)
                response.raise_for_status()
                data = response.json()
            except Exception as e:
                print(f"    [T{i}] ERROR: {e}")
                trace_detail["turns"].append({"turn": i, "error": str(e)})
                trace_detail["all_hit"] = False
                continue

            if conversation_id is None:
                conversation_id = data.get("conversation_id")

            sources = data.get("sources", [])
            recall, mrr, hit = score_retrieval(expected_sources, sources)

            results["total_turns"] += 1
            results["turn_recall_sum"] += recall
            results["turn_mrr_sum"] += mrr
            results["turn_hits"] += hit
            if not hit:
                trace_detail["all_hit"] = False

            trace_detail["turns"].append({
                "turn": i,
                "question": question,
                "expected_sources": expected_sources,
                "recall": recall,
                "mrr": mrr,
                "hit": bool(hit),
                "answer_preview": data.get("answer", "")[:140],
            })

            if verbose:
                status = "HIT" if hit else "MISS"
                print(f"    [T{i}] {status} | Recall={recall:.2f} | {question[:60]}")

        results["trace_details"].append(trace_detail)

    return results


def print_report(results: dict, top_k: int):
    """Print formatted single-turn evaluation report."""
    total = results["total"]
    if total == 0:
        print("No questions evaluated.")
        return

    recall_at_k = results["recall_sum"] / total
    mrr = results["mrr_sum"] / total
    hit_rate = results["hits"] / total

    print(f"\n{'=' * 70}")
    print(f"  SINGLE-TURN EVALUATION (top_k={top_k}, {total} questions)")
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

    misses = [d for d in results["details"] if not d.get("hit", False) and "error" not in d]
    if misses:
        print(f"\n  Missed Questions ({len(misses)}):")
        for m in misses:
            print(f"    [{m['id']}] {m['question'][:70]}")
            if m["category"] == "refusal":
                print(f"            Expected refusal; got non-refusal answer")
            else:
                print(f"            Expected: {m['expected_sources']}")

    print(f"\n{'=' * 70}")


def print_conversation_report(results: dict):
    """Print formatted multi-turn evaluation report."""
    total_traces = results["total_traces"]
    total_turns = results["total_turns"]
    if total_turns == 0:
        return

    turn_recall = results["turn_recall_sum"] / total_turns
    turn_mrr = results["turn_mrr_sum"] / total_turns
    turn_hit_rate = results["turn_hits"] / total_turns
    full_traces_passed = sum(1 for t in results["trace_details"] if t.get("all_hit", False))

    print(f"\n{'=' * 70}")
    print(f"  MULTI-TURN CONVERSATION EVALUATION ({total_traces} traces, {total_turns} turns)")
    print(f"{'=' * 70}")
    print(f"\n  Overall Per-Turn Metrics:")
    print(f"    Turn Recall:    {turn_recall:.3f}")
    print(f"    Turn MRR:       {turn_mrr:.3f}")
    print(f"    Turn Hit Rate:  {turn_hit_rate:.3f}  ({results['turn_hits']}/{total_turns})")
    print(f"\n  Full-Trace Pass Rate:  {full_traces_passed}/{total_traces}  (every turn HIT)")

    failed_traces = [t for t in results["trace_details"] if not t.get("all_hit", False)]
    if failed_traces:
        print(f"\n  Traces with at least one MISS ({len(failed_traces)}):")
        for t in failed_traces:
            misses = [tu for tu in t["turns"] if not tu.get("hit", False) and "error" not in tu]
            print(f"    [{t['id']}] {t['category']}: {len(misses)}/{len(t['turns'])} miss")

    print(f"\n{'=' * 70}")


def main():
    parser = argparse.ArgumentParser(description="Evaluate AppLeap RAG retrieval quality")
    parser.add_argument("--api-url", default="http://localhost:8000")
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--verbose", "-v", action="store_true")
    parser.add_argument("--output", "-o", help="Save detailed results to JSON file")
    parser.add_argument("--skip-conversations", action="store_true", help="Skip multi-turn eval")
    parser.add_argument("--skip-singleturn", action="store_true", help="Skip single-turn eval")
    args = parser.parse_args()

    eval_dir = Path(__file__).parent
    questions_path = eval_dir / "questions.json"
    conversations_path = eval_dir / "conversations.json"

    client = httpx.Client()
    try:
        r = client.get(f"{args.api_url}/health", timeout=5.0)
        r.raise_for_status()
    except Exception as e:
        print(f"ERROR: Cannot reach API at {args.api_url}: {e}")
        sys.exit(1)

    combined: dict = {"top_k": args.top_k}

    # Single-turn eval
    if not args.skip_singleturn:
        if not questions_path.exists():
            print(f"ERROR: {questions_path} not found")
            sys.exit(1)
        questions = load_questions(questions_path)
        print(f"Loaded {len(questions)} single-turn questions")
        print(f"\nRunning single-turn retrieval evaluation (top_k={args.top_k})...")
        start = time.time()
        results = evaluate_retrieval(client, args.api_url, questions, args.top_k, args.verbose)
        elapsed = time.time() - start
        print_report(results, args.top_k)
        print(f"\n  Elapsed: {elapsed:.1f}s ({elapsed / max(results['total'], 1):.1f}s/question)")
        results["elapsed_seconds"] = elapsed
        combined["single_turn"] = results

    # Multi-turn eval
    if not args.skip_conversations and conversations_path.exists():
        conversations = load_conversations(conversations_path)
        print(f"\nLoaded {len(conversations)} conversation traces")
        print(f"Running multi-turn evaluation...")
        start = time.time()
        conv_results = evaluate_conversations(
            client, args.api_url, conversations, args.top_k, args.verbose
        )
        elapsed = time.time() - start
        print_conversation_report(conv_results)
        print(f"\n  Elapsed: {elapsed:.1f}s ({elapsed / max(conv_results['total_turns'], 1):.1f}s/turn)")
        conv_results["elapsed_seconds"] = elapsed
        combined["conversations"] = conv_results

    if args.output:
        with open(args.output, "w") as f:
            json.dump(combined, f, indent=2)
        print(f"\n  Detailed results saved to {args.output}")


if __name__ == "__main__":
    main()
