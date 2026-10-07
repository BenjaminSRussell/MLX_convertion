"""Standard artifact layout + metadata.json (#11).

Layout:
    artifacts/{model_slug}/{bits}bit/
        metadata.json      # strategy, metrics, git sha, files (+sha256)
        <weights / config / tokenizer files>

``model_slug`` replaces "/" in HF ids with "__" so paths stay flat.
"""
from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Mapping, Optional

ARTIFACTS_ROOT = Path("artifacts")
METADATA_FILE = "metadata.json"
SCHEMA_VERSION = 1


def model_slug(model_id: str) -> str:
    return model_id.replace("/", "__")


def artifact_dir(model_id: str, bits: int, root: Path | str = ARTIFACTS_ROOT) -> Path:
    return Path(root) / model_slug(model_id) / f"{int(bits)}bit"


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def build_metadata(model_id: str, bits: int, directory: Path, strategy: Optional[str] = None,
                   metrics: Optional[Mapping[str, Any]] = None, git_sha: Optional[str] = None,
                   run_id: Optional[int] = None, extra: Optional[Mapping[str, Any]] = None) -> Dict[str, Any]:
    directory = Path(directory)
    files = sorted(p for p in directory.rglob("*") if p.is_file() and p.name != METADATA_FILE)
    return {
        "schema_version": SCHEMA_VERSION,
        "model_id": model_id,
        "bits": int(bits),
        "strategy": strategy,
        "metrics": dict(metrics or {}),
        "git_sha": git_sha,
        "run_id": run_id,
        "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "files": [
            {"path": str(p.relative_to(directory)), "bytes": p.stat().st_size, "sha256": sha256_file(p)}
            for p in files
        ],
        **dict(extra or {}),
    }


def write_metadata(directory: Path, metadata: Mapping[str, Any]) -> Path:
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    out = directory / METADATA_FILE
    out.write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return out


def read_metadata(directory: Path) -> Dict[str, Any]:
    path = Path(directory) / METADATA_FILE
    if not path.exists():
        raise FileNotFoundError(f"missing {METADATA_FILE} in {directory}")
    data = json.loads(path.read_text(encoding="utf-8"))
    for key in ("model_id", "bits", "files"):
        if key not in data:
            raise ValueError(f"{path}: metadata missing '{key}'")
    return data


def verify_files(directory: Path, metadata: Optional[Mapping[str, Any]] = None) -> list[str]:
    """Return a list of problems (missing files / sha mismatch); empty list == OK."""
    directory = Path(directory)
    metadata = metadata or read_metadata(directory)
    problems = []
    for entry in metadata["files"]:
        p = directory / entry["path"]
        if not p.exists():
            problems.append(f"missing: {entry['path']}")
        elif sha256_file(p) != entry["sha256"]:
            problems.append(f"sha256 mismatch: {entry['path']}")
    return problems
