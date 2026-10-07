#!/usr/bin/env python3
"""Gate report across models from the run registry (#8).

Prints the last N runs with per-gate pass/fail. Exits 1 if any listed run failed.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from utils.run_registry import DEFAULT_DB, RunRegistry  # noqa: E402

GATES = ("layer", "task", "parity", "performance")


def _cell(v) -> str:
    if v is None:
        return "-"
    return "PASS" if v >= 1.0 else "FAIL"


def build_report(runs) -> tuple[list[str], bool]:
    lines = [f"{'id':>4}  {'model':<36} {'status':<7} " + " ".join(f"{g:<11}" for g in GATES)]
    any_failed = False
    for r in runs:
        m = r.get("metrics", {})
        if r["status"] == "failed":
            any_failed = True
        lines.append(f"{r['id']:>4}  {r['model_id'][:36]:<36} {r['status']:<7} "
                     + " ".join(f"{_cell(m.get('gate_' + g)):<11}" for g in GATES))
    return lines, any_failed


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--db", type=Path, default=DEFAULT_DB)
    ap.add_argument("-n", "--last", type=int, default=20)
    ap.add_argument("--model", help="Filter to one model id")
    args = ap.parse_args(argv)
    runs = [r for r in RunRegistry(args.db).list_runs(limit=args.last, model_id=args.model)
            if r["status"] != "running"]
    if not runs:
        print("No completed runs in registry.")
        return 0
    lines, failed = build_report(runs)
    print("\n".join(lines))
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
