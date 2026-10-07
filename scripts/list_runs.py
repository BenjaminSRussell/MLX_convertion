#!/usr/bin/env python3
"""List conversion runs from the SQLite registry (#6)."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from utils.run_registry import DEFAULT_DB, RunRegistry  # noqa: E402


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--db", type=Path, default=DEFAULT_DB)
    ap.add_argument("-n", "--limit", type=int, default=20)
    ap.add_argument("--model")
    ap.add_argument("--status", choices=["running", "passed", "failed"])
    ap.add_argument("--json", action="store_true")
    args = ap.parse_args(argv)
    runs = RunRegistry(args.db).list_runs(limit=args.limit, model_id=args.model, status=args.status)
    if args.json:
        print(json.dumps(runs, indent=2))
        return 0
    print(f"{'id':>4}  {'status':<8} {'model':<40} {'bits':>4}  {'strategy':<14} created")
    for r in runs:
        print(f"{r['id']:>4}  {r['status']:<8} {r['model_id'][:40]:<40} {str(r['bits'] or ''):>4}  "
              f"{(r['strategy'] or '')[:14]:<14} {r['created_at']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
