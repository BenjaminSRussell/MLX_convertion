#!/usr/bin/env python3
"""Post-convert weight optimization (#5).

Passes (all lossless except the optional dtype cast):
  * drop training-only tensors (optimizer state, EMA shadows, num_batches_tracked)
  * tie duplicate tensors (e.g. input embeddings == lm_head) -> stored once,
    recorded in ``__tied__`` so loaders can re-alias them
  * optional float32 -> float16 cast (``--fp16``)

CLI:
    python scripts/optimize.py IN.npz OUT.npz [--fp16]
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any, Dict, Mapping, Tuple

import numpy as np

TRAINING_ONLY_TOKENS = ("optimizer", "ema_", "_ema", "num_batches_tracked", "grad_")
TIED_KEY = "__tied__"


def _digest(arr: np.ndarray) -> str:
    h = hashlib.sha256()
    h.update(str(arr.dtype).encode())
    h.update(str(arr.shape).encode())
    h.update(np.ascontiguousarray(arr).tobytes())
    return h.hexdigest()


def optimize_weights(weights: Mapping[str, np.ndarray], fp16: bool = False) -> Tuple[Dict[str, np.ndarray], Dict[str, Any]]:
    dropped, tied = [], {}
    seen: Dict[str, str] = {}
    out: Dict[str, np.ndarray] = {}
    for name in sorted(weights):
        arr = np.asarray(weights[name])
        if any(tok in name.lower() for tok in TRAINING_ONLY_TOKENS):
            dropped.append(name)
            continue
        if fp16 and arr.dtype == np.float32:
            arr = arr.astype(np.float16)
        d = _digest(arr)
        if d in seen:
            tied[name] = seen[d]
            continue
        seen[d] = name
        out[name] = arr
    if tied:
        out[TIED_KEY] = np.array(json.dumps(tied))
    report = {"dropped": dropped, "tied": tied, "fp16": fp16,
              "tensors_in": len(weights), "tensors_out": len([k for k in out if k != TIED_KEY])}
    return out, report


def restore_ties(weights: Mapping[str, np.ndarray]) -> Dict[str, np.ndarray]:
    out = {k: v for k, v in weights.items() if k != TIED_KEY}
    if TIED_KEY in weights:
        for alias, source in json.loads(str(weights[TIED_KEY])).items():
            out[alias] = out[source]
    return out


def optimize_npz(src: Path, dst: Path, fp16: bool = False) -> Dict[str, Any]:
    with np.load(src) as data:
        weights = {k: data[k] for k in data.files}
    optimized, report = optimize_weights(weights, fp16=fp16)
    dst.parent.mkdir(parents=True, exist_ok=True)
    np.savez(dst, **optimized)
    report.update({"input_bytes": src.stat().st_size, "output_bytes": dst.stat().st_size})
    return report


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("src", type=Path, help="input weights .npz")
    ap.add_argument("dst", type=Path, help="output optimized .npz")
    ap.add_argument("--fp16", action="store_true", help="cast float32 tensors to float16")
    ap.add_argument("--report", type=Path, help="write JSON report here")
    args = ap.parse_args(argv)
    report = optimize_npz(args.src, args.dst, fp16=args.fp16)
    text = json.dumps(report, indent=2)
    if args.report:
        args.report.write_text(text + "\n", encoding="utf-8")
    print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
