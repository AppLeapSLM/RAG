"""Focused correctness eval for the tabular-SQL path.

Ground truth is COMPUTED from the source CSVs at run time (not hand-authored
answer keys), so it cannot drift from the data. For each case we check BOTH:
  - routing: did it go to SQL vs RAG as expected, and
  - correctness: does the SQL scalar equal the value computed from the CSV.

Includes negative cases (non-tabular questions that must NOT route to SQL) as a
false-positive guard. Runs try_sql_answer directly (the HTTP/SSE path is covered
by the separate E2E); user_email=None since the CMDB corpus is public.

Usage (on the VM, as the service user):
    python -m eval.sql_eval --data-dir /tmp        # or the novacrest cmdb dir
"""

import argparse
import asyncio
import csv
from pathlib import Path

from backend.db.connection import async_session
from backend.retrieval.sql_router import try_sql_answer

FILES = ["cmdb-production.csv", "cmdb-databases.csv", "cmdb-full-export.csv"]


def _load(data_dir: str) -> dict[str, list[dict]]:
    out = {}
    for f in FILES:
        for cand in (Path(data_dir) / f, Path(data_dir) / "cmdb" / f):
            if cand.exists():
                out[f] = list(csv.DictReader(open(cand, encoding="utf-8")))
                break
        else:
            out[f] = []
    return out


def _count(rows, pred) -> int:
    return sum(1 for r in rows if pred(r))


def _build_cases(D):
    prod, dbs, full = D["cmdb-production.csv"], D["cmdb-databases.csv"], D["cmdb-full-export.csv"]
    # (question, expected_route, expected_value)  — value computed from the CSV
    return [
        ("In cmdb-production.csv, how many CIs have status degraded or maintenance?",
         "sql", _count(prod, lambda r: r["status"] in ("degraded", "maintenance"))),
        ("In cmdb-production.csv, how many CIs are tier 1?",
         "sql", _count(prod, lambda r: r["tier"] == "1")),
        ("In cmdb-production.csv, how many CIs are of type cache?",
         "sql", _count(prod, lambda r: r["type"] == "cache")),
        ("In cmdb-production.csv, how many CIs have criticality critical?",
         "sql", _count(prod, lambda r: r["criticality"] == "critical")),
        ("In cmdb-databases.csv, how many databases are tier 1?",
         "sql", _count(dbs, lambda r: r["tier"] == "1")),
        ("In cmdb-databases.csv, how many databases have multi_az enabled?",
         "sql", _count(dbs, lambda r: str(r.get("multi_az", "")).lower() in ("true", "yes", "1"))),
        ("In cmdb-full-export.csv, how many CIs are in the staging environment?",
         "sql", _count(full, lambda r: r["environment"] == "staging")),
        ("In cmdb-full-export.csv, how many CIs have status operational?",
         "sql", _count(full, lambda r: r["status"] == "operational")),
        ("In cmdb-full-export.csv, how many CIs are tier 2?",
         "sql", _count(full, lambda r: r["tier"] == "2")),
        # Negative cases — must NOT route to SQL (prose / no relevant table):
        ("What is the auth-service JWT key rotation procedure?", "rag", None),
        ("What does ADR-013 decide about the Kafka platform?", "rag", None),
    ]


def _scalar(ans) -> float | None:
    if not ans or not ans["result"]["rows"]:
        return None
    v = ans["result"]["rows"][0][0]
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


async def main(data_dir: str):
    D = _load(data_dir)
    missing = [f for f in FILES if not D[f]]
    if missing:
        print(f"WARNING: could not load {missing} from {data_dir} (ground truth incomplete)")
    cases = _build_cases(D)
    passed = 0
    async with async_session() as s:
        for q, exp_route, exp_val in cases:
            ans = await try_sql_answer(s, q, conversation_id=None, user_email=None)
            route = "sql" if ans else "rag"
            if exp_route == "rag":
                ok = route == "rag"
                print(f"[{'PASS' if ok else 'FAIL'}] route={route} (want rag) | {q[:58]}")
            else:
                val = _scalar(ans)
                ok = route == "sql" and val is not None and val == float(exp_val)
                print(f"[{'PASS' if ok else 'FAIL'}] got={val} want={exp_val} route={route} | {q[:50]}")
            passed += ok
    print(f"\n{passed}/{len(cases)} passed")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--data-dir", default="/tmp")
    args = p.parse_args()
    asyncio.run(main(args.data_dir))
