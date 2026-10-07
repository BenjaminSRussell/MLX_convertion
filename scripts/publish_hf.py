#!/usr/bin/env python3
"""Publish an artifact directory to the Hugging Face Hub (#12).

Dry-run by default: prints the planned repo + files and renders the model card
from metadata.json (#11). Pass --execute to actually upload.

Token: read from the HF_TOKEN environment variable only. Never pass tokens on
the command line or commit them.
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import Any, Mapping

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from utils.artifacts import read_metadata, verify_files  # noqa: E402

TEMPLATE = Path(__file__).resolve().parents[1] / "templates" / "model_card.md"


def default_repo_id(meta: Mapping[str, Any], namespace: str) -> str:
    base = str(meta["model_id"]).split("/")[-1]
    return f"{namespace}/{base}-mlx-{meta['bits']}bit"


def render_model_card(meta: Mapping[str, Any], repo_id: str, template: Path = TEMPLATE) -> str:
    metrics = meta.get("metrics") or {}
    metrics_table = "\n".join(["| metric | value |", "|--------|-------|"]
                              + [f"| {k} | {v} |" for k, v in sorted(metrics.items())]) if metrics else "_none recorded_"
    files_table = "\n".join(["| file | bytes |", "|------|-------|"]
                            + [f"| `{f['path']}` | {f['bytes']} |" for f in meta.get("files", [])])
    values = {
        "repo_id": repo_id,
        "model_id": meta["model_id"],
        "bits": meta["bits"],
        "strategy": meta.get("strategy") or "n/a",
        "git_sha": meta.get("git_sha") or "n/a",
        "created_at": meta.get("created_at") or "n/a",
        "metrics_table": metrics_table,
        "files_table": files_table,
    }
    text = template.read_text(encoding="utf-8")
    for key, val in values.items():
        text = text.replace("{{" + key + "}}", str(val))
    return text


def plan(artifact_dir: Path, namespace: str, repo_id: str | None = None) -> dict:
    meta = read_metadata(artifact_dir)
    problems = verify_files(artifact_dir, meta)
    rid = repo_id or default_repo_id(meta, namespace)
    return {
        "repo_id": rid,
        "files": [f["path"] for f in meta["files"]] + ["metadata.json", "README.md"],
        "problems": problems,
        "model_card": render_model_card(meta, rid),
    }


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("artifact_dir", type=Path, help="artifacts/{model}/{bits}bit directory")
    ap.add_argument("--namespace", default=os.environ.get("HF_NAMESPACE", "BenjaminSRussell"))
    ap.add_argument("--repo-id", help="Override target repo id")
    ap.add_argument("--private", action="store_true")
    ap.add_argument("--execute", action="store_true", help="Actually upload (default: dry-run)")
    args = ap.parse_args(argv)

    p = plan(args.artifact_dir, args.namespace, args.repo_id)
    print(f"[publish] target repo: {p['repo_id']}")
    for f in p["files"]:
        print(f"[publish]   + {f}")
    if p["problems"]:
        for prob in p["problems"]:
            print(f"[publish] ERROR {prob}", file=sys.stderr)
        return 2
    card_path = args.artifact_dir / "README.md"
    if not args.execute:
        print("[publish] dry-run: model card preview below (use --execute to upload)\n")
        print(p["model_card"])
        return 0

    token = os.environ.get("HF_TOKEN")
    if not token:
        print("[publish] HF_TOKEN is not set; refusing to upload", file=sys.stderr)
        return 3
    try:
        from huggingface_hub import HfApi
    except ImportError:
        print("[publish] pip install huggingface_hub to upload", file=sys.stderr)
        return 4
    card_path.write_text(p["model_card"], encoding="utf-8")
    api = HfApi(token=token)
    api.create_repo(p["repo_id"], private=args.private, exist_ok=True)
    api.upload_folder(repo_id=p["repo_id"], folder_path=str(args.artifact_dir))
    print(f"[publish] uploaded to https://huggingface.co/{p['repo_id']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
