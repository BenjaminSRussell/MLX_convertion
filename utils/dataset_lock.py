"""Reproducible eval datasets (#9).

Each dataset in ``config/datasets.yaml`` may pin:
    revision: <HF git revision / commit sha>   # passed to load_dataset
    sha256:   <fingerprint of the eval split>  # verified after download

The fingerprint is a sha256 over a canonical JSONL rendering of the first
``fingerprint_rows`` rows (default 256) of the validation split, so it is
cheap to compute and independent of HF cache layout. A mismatch raises
``DatasetChecksumError`` and the download is treated as failed.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Iterable, Mapping, Optional

DEFAULT_FINGERPRINT_ROWS = 256


class DatasetChecksumError(RuntimeError):
    pass


def canonical_rows(rows: Iterable[Mapping[str, Any]]) -> bytes:
    lines = [json.dumps({k: (v if isinstance(v, (int, float, bool)) or v is None else str(v))
                         for k, v in sorted(dict(r).items())}, sort_keys=True, ensure_ascii=False)
             for r in rows]
    return ("\n".join(lines) + "\n").encode("utf-8")


def fingerprint_rows(rows: Iterable[Mapping[str, Any]]) -> str:
    return hashlib.sha256(canonical_rows(rows)).hexdigest()


def fingerprint_split(split, n: int = DEFAULT_FINGERPRINT_ROWS) -> str:
    """Fingerprint a datasets.Dataset (or any sequence of dict rows)."""
    if hasattr(split, "select"):
        rows = split.select(range(min(n, len(split))))
    else:
        rows = list(split)[:n]
    return fingerprint_rows(rows)


def verify(actual: str, expected: Optional[str], name: str) -> None:
    if expected and actual != expected:
        raise DatasetChecksumError(f"{name}: sha256 mismatch (expected {expected[:12]}…, got {actual[:12]}…)")


def fingerprint_jsonl(path: Path, n: int = DEFAULT_FINGERPRINT_ROWS) -> str:
    rows = []
    with Path(path).open(encoding="utf-8") as fh:
        for line in fh:
            if line.strip():
                rows.append(json.loads(line))
            if len(rows) >= n:
                break
    return fingerprint_rows(rows)


def verify_fixture(path: Path, expected: str, n: int = DEFAULT_FINGERPRINT_ROWS) -> str:
    actual = fingerprint_jsonl(path, n)
    verify(actual, expected, str(path))
    return actual
