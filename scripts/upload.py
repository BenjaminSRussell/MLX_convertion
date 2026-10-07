#!/usr/bin/env python3
"""Upload converted models with tamper-evident manifest."""
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import subprocess
from pathlib import Path
from typing import Dict, List

PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODELS_DIR = PROJECT_ROOT / "models" / "mlx_converted"
RESULTS_DIR = PROJECT_ROOT / "results" / "comparisons"


def compute_sha256(path: Path) -> str:
    sha = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            sha.update(chunk)
    return sha.hexdigest()


def collect_files(model_dir: Path) -> List[Path]:
    return [file for file in model_dir.glob("**/*") if file.is_file()]


def build_manifest(model_name: str, quant_id: str, files: List[Path]) -> Dict:
    model_dir = MODELS_DIR / model_name / quant_id
    return {
        "model": model_name,
        "quantization": quant_id,
        "files": [
            {"path": str(file.relative_to(model_dir)), "sha256": compute_sha256(file)}
            for file in files
        ],
    }


def write_manifest(manifest: Dict, dest: Path) -> Path:
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return dest


def package(model_dir: Path, destination: Path) -> Path:
    destination.parent.mkdir(parents=True, exist_ok=True)
    archive = shutil.make_archive(str(destination), "zip", root_dir=model_dir)
    return Path(archive)


def parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Upload MLX models with proof-of-integrity manifest")
    parser.add_argument("model", nargs="?", help="Model name from models.yaml")
    parser.add_argument("quant", nargs="?", help="Quantization id from models.yaml")
    parser.add_argument("--artifact-dir", type=Path,
                        help="Standard artifacts/{model}/{bits}bit dir with metadata.json (#11)")
    parser.add_argument("--artifact", type=Path, default=PROJECT_ROOT / "results" / "comparisons")
    parser.add_argument("--dry-run", action="store_true", help="Print what would be uploaded; write nothing")
    parser.add_argument("--upload-cmd", nargs=argparse.REMAINDER, help="Command to run for upload (e.g. aws s3 cp)")
    args = parser.parse_args(argv)
    if not args.artifact_dir and not (args.model and args.quant):
        parser.error("give MODEL QUANT or --artifact-dir")
    return args


def _dry_run_artifact_dir(artifact_dir: Path) -> int:
    import sys
    sys.path.insert(0, str(PROJECT_ROOT))
    from utils.artifacts import read_metadata, verify_files

    meta = read_metadata(artifact_dir)
    problems = verify_files(artifact_dir, meta)
    print(f"DRY RUN: {meta['model_id']} ({meta['bits']}-bit, strategy={meta.get('strategy')})")
    print(f"  source: {artifact_dir}")
    for f in meta["files"]:
        print(f"  + {f['path']}  {f['bytes']} bytes  sha256={f['sha256'][:12]}…")
    for prob in problems:
        print(f"  ERROR {prob}")
    return 2 if problems else 0


def main(argv=None) -> int:
    args = parse_args(argv)
    if args.artifact_dir:
        if args.dry_run:
            return _dry_run_artifact_dir(args.artifact_dir)
        model_dir = args.artifact_dir
        label = f"{model_dir.parent.name}_{model_dir.name}"
    else:
        model_dir = MODELS_DIR / args.model / args.quant
        label = f"{args.model}_{args.quant}"
    if not model_dir.exists():
        raise SystemExit(f"Converted model not found: {model_dir}")

    files = collect_files(model_dir)
    if not files:
        raise SystemExit(f"No files to upload under {model_dir}")

    if args.dry_run:
        print(f"DRY RUN: would package {len(files)} files from {model_dir}")
        for f in files:
            print(f"  + {f.relative_to(model_dir)}")
        return 0

    manifest = {
        "model": label,
        "files": [{"path": str(f.relative_to(model_dir)), "sha256": compute_sha256(f)} for f in files],
    }
    manifest_path = write_manifest(manifest, RESULTS_DIR / f"{label}_manifest.json")
    package_path = package(model_dir, args.artifact / label)

    if args.upload_cmd:
        cmd = args.upload_cmd + [str(package_path), str(manifest_path)]
        print("Running upload command:", " ".join(cmd))
        if not shutil.which(cmd[0]):
            raise SystemExit(f"Upload tool not found: {cmd[0]}")
        proc = subprocess.run(cmd, check=False)  # nosec B603
        if proc.returncode != 0:
            raise SystemExit(f"Upload command failed with code {proc.returncode}")

    proof = compute_sha256(manifest_path)
    print("Upload package:", package_path)
    print("Manifest:", manifest_path)
    print("Manifest SHA256:", proof)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
