#!/usr/bin/env python3
"""Group-wise affine weight quantization (#5).

Same scheme as ``mlx.core.quantize``: each row is split into groups of
``group_size`` values; each group stores ``scale``/``bias`` (float16) and
unsigned ``bits``-bit codes. 4-bit codes are packed two per byte.

Works on plain numpy weight archives (.npz) so it runs on Linux CI. When MLX is
installed, ``quantize_model`` delegates to ``mlx.nn.quantize`` for real models.

CLI:
    python scripts/quantize.py IN.npz OUT.npz --bits 8 [--group-size 64]
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, Mapping, Tuple

import numpy as np

SUPPORTED_BITS = (4, 8)
QUANT_SUFFIXES = ("__q", "__scales", "__biases", "__shape")


def _bits_from(config: Any) -> int:
    bits = config.get("bits", 8) if isinstance(config, Mapping) else config
    bits = int(bits)
    if bits not in SUPPORTED_BITS:
        raise ValueError(f"unsupported bits={bits}; expected one of {SUPPORTED_BITS}")
    return bits


def quantize_array(w: np.ndarray, bits: int = 8, group_size: int = 64) -> Dict[str, np.ndarray]:
    if bits not in SUPPORTED_BITS:
        raise ValueError(f"unsupported bits={bits}")
    w = np.asarray(w, dtype=np.float32)
    shape = np.array(w.shape, dtype=np.int64)
    flat = w.reshape(-1)
    pad = (-flat.size) % group_size
    if pad:
        flat = np.concatenate([flat, np.zeros(pad, dtype=np.float32)])
    groups = flat.reshape(-1, group_size)
    lo = groups.min(axis=1, keepdims=True)
    hi = groups.max(axis=1, keepdims=True)
    levels = (1 << bits) - 1
    scale = (hi - lo) / levels
    scale[scale == 0] = 1.0
    codes = np.clip(np.rint((groups - lo) / scale), 0, levels).astype(np.uint8)
    if bits == 4:
        codes = (codes[:, 0::2] | (codes[:, 1::2] << 4)).astype(np.uint8)
    return {
        "q": codes,
        "scales": scale.astype(np.float16),
        "biases": lo.astype(np.float16),
        "shape": shape,
    }


def dequantize_array(packed: Mapping[str, np.ndarray], bits: int = 8) -> np.ndarray:
    codes = packed["q"]
    if bits == 4:
        lo4 = codes & 0x0F
        hi4 = codes >> 4
        codes = np.empty((codes.shape[0], codes.shape[1] * 2), dtype=np.uint8)
        codes[:, 0::2] = lo4
        codes[:, 1::2] = hi4
    vals = codes.astype(np.float32) * packed["scales"].astype(np.float32) + packed["biases"].astype(np.float32)
    shape = tuple(int(x) for x in packed["shape"])
    n = int(np.prod(shape)) if shape else 1
    return vals.reshape(-1)[:n].reshape(shape)


def should_quantize(name: str, arr: np.ndarray, group_size: int) -> bool:
    """Quantize 2-D+ float weight matrices; keep norms/biases/embeddings-positions in float."""
    if arr.ndim < 2 or not np.issubdtype(arr.dtype, np.floating):
        return False
    lowered = name.lower()
    if any(tok in lowered for tok in ("norm", "layernorm", "position")):
        return False
    return arr.size >= group_size


def quantize_weights(weights: Mapping[str, np.ndarray], bits: int = 8, group_size: int = 64) -> Tuple[Dict[str, np.ndarray], Dict[str, Any]]:
    out: Dict[str, np.ndarray] = {}
    quantized, kept = [], []
    for name, arr in weights.items():
        arr = np.asarray(arr)
        if should_quantize(name, arr, group_size):
            packed = quantize_array(arr, bits, group_size)
            for key, val in packed.items():
                out[f"{name}__{key}"] = val
            quantized.append(name)
        else:
            out[name] = arr
            kept.append(name)
    meta = {"bits": bits, "group_size": group_size, "quantized": quantized, "kept": kept}
    return out, meta


def dequantize_weights(qweights: Mapping[str, np.ndarray], bits: int) -> Dict[str, np.ndarray]:
    out: Dict[str, np.ndarray] = {}
    bases = {k[: -len("__q")] for k in qweights if k.endswith("__q")}
    for base in bases:
        out[base] = dequantize_array({s: qweights[f"{base}__{s}"] for s in ("q", "scales", "biases", "shape")}, bits)
    for k, v in qweights.items():
        if not k.endswith(QUANT_SUFFIXES):
            out[k] = v
    return out


def quantization_error(original: Mapping[str, np.ndarray], restored: Mapping[str, np.ndarray]) -> Dict[str, float]:
    worst_cos, worst_rel = 1.0, 0.0
    for name, a in original.items():
        a = np.asarray(a, dtype=np.float32).reshape(-1)
        b = np.asarray(restored[name], dtype=np.float32).reshape(-1)
        denom = float(np.linalg.norm(a) * np.linalg.norm(b))
        cos = float(np.dot(a, b) / denom) if denom else 1.0
        rel = float(np.linalg.norm(a - b) / (np.linalg.norm(a) or 1.0))
        worst_cos, worst_rel = min(worst_cos, cos), max(worst_rel, rel)
    return {"min_cosine": worst_cos, "max_relative_error": worst_rel}


def quantize_npz(src: Path, dst: Path, bits: int = 8, group_size: int = 64) -> Dict[str, Any]:
    with np.load(src) as data:
        weights = {k: data[k] for k in data.files}
    qweights, meta = quantize_weights(weights, bits, group_size)
    dst.parent.mkdir(parents=True, exist_ok=True)
    np.savez(dst, **qweights)
    dst = dst if dst.suffix == ".npz" else dst.with_suffix(dst.suffix + ".npz")
    restored = dequantize_weights(qweights, bits)
    meta.update(quantization_error(weights, restored))
    meta.update({"input_bytes": src.stat().st_size, "output_bytes": dst.stat().st_size})
    return meta


def quantize_model(model, bits):
    """Quantize an in-memory model.

    MLX ``nn.Module``: delegates to ``mlx.nn.quantize`` (in place).
    Anything else (e.g. a torch model before MLX conversion) is returned
    unchanged; quantize the converted .npz weights with ``quantize_npz``.
    """
    nbits = _bits_from(bits)
    try:
        import mlx.nn as nn  # type: ignore
    except ImportError:
        nn = None
    if nn is not None and isinstance(model, nn.Module):
        nn.quantize(model, bits=nbits)
        return model
    print(f"[quantize] model type {type(model).__name__} is not an MLX module; "
          f"apply {nbits}-bit quantization to converted weights via quantize_npz")
    return model


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("src", type=Path, help="input weights .npz")
    ap.add_argument("dst", type=Path, help="output quantized .npz")
    ap.add_argument("--bits", type=int, choices=SUPPORTED_BITS, default=8)
    ap.add_argument("--group-size", type=int, default=64)
    ap.add_argument("--min-cosine", type=float, default=None, help="fail if worst-tensor cosine falls below")
    args = ap.parse_args(argv)
    meta = quantize_npz(args.src, args.dst, args.bits, args.group_size)
    print(json.dumps({k: v for k, v in meta.items() if k not in ("quantized", "kept")}, indent=2))
    if args.min_cosine is not None and meta["min_cosine"] < args.min_cosine:
        print(f"[quantize] FAIL min_cosine {meta['min_cosine']:.5f} < {args.min_cosine}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
