#!/usr/bin/env python3
"""Tail conversion/test logs to monitor throughput and quality."""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Dict, List

PROJECT_ROOT = Path(__file__).resolve().parents[1]
CONVERSION_DIR = PROJECT_ROOT / "results" / "conversion_logs"
TEST_DIR = PROJECT_ROOT / "results" / "test_results"


def load_jsonl(path: Path) -> List[Dict]:
    rows: List[Dict] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return rows


def summarize_conversion() -> Dict:
    stats = {"jobs": 0, "failures": 0}
    for log in CONVERSION_DIR.glob("*.jsonl"):
        rows = load_jsonl(log)
        stats["jobs"] += len(rows)
        stats["failures"] += sum(1 for row in rows if row.get("returncode"))
    return stats


def summarize_tests() -> Dict:
    stats: Dict[str, Dict[str, float]] = {}
    for log in TEST_DIR.glob("*.jsonl"):
        rows = load_jsonl(log)
        for row in rows:
            dataset = row.get("dataset", "unknown")
            metrics = row.get("metrics", {})
            dataset_stats = stats.setdefault(dataset, {"accuracy_sum": 0.0, "runs": 0})
            if "accuracy" in metrics:
                dataset_stats["accuracy_sum"] += metrics["accuracy"]
            dataset_stats["runs"] += 1
    return stats


def render(conversion: Dict, tests: Dict) -> None:
    print("=== Conversion Jobs ===")
    print(f"Total: {conversion['jobs']} | Failures: {conversion['failures']}")
    print("=== Test Runs ===")
    if not tests:
        print("No test logs yet")
        return
    for dataset, stats in tests.items():
        runs = stats["runs"]
        avg = stats["accuracy_sum"] / runs if runs else 0.0
        print(f"{dataset}: runs={runs} avg_accuracy={avg:.4f}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Monitor conversion + test outputs")
    parser.add_argument("--interval", type=int, default=30, help="Polling interval seconds")
    parser.add_argument("--oneshot", action="store_true", help="Print once and exit")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    while True:
        conversion = summarize_conversion()
        tests = summarize_tests()
        render(conversion, tests)
        if args.oneshot:
            break
        time.sleep(args.interval)


if __name__ == "__main__":
    main()
