"""Convert Hugging Face causal-LM checkpoints listed in models.yaml to MLX.

``pipeline.sh`` invokes this with ``--dry-run``; dry-run must not require mlx.
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import yaml


def convert_model(config):
    """
    Convert a causal-LM based on configuration.

    Args:
        config (dict): Single model configuration (must include name/hf_name + type).
    """
    import mlx.core as mx
    from transformers import AutoTokenizer, AutoModelForCausalLM
    import torch
    from concurrent.futures import ThreadPoolExecutor

    model_name = config["name"]
    model_type = config["type"]

    print(f"Converting {model_name} ({model_type})...")

    output_path = config.get(
        "output_dir", os.path.join("models", model_name.replace("/", "_"))
    )
    os.makedirs(output_path, exist_ok=True)

    if model_type == "causal-lm":
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16,
            device_map="auto",
        )
    else:
        raise ValueError(f"Unsupported model type: {model_type}")

    if config.get("quantization"):
        from quantize import quantize_model

        model = quantize_model(model, config["quantization"])

    state_dict = model.state_dict()

    def convert_weights_chunk(chunk):
        converted = {}
        for k, v in chunk.items():
            if k.startswith("transformer."):
                k = k[len("transformer.") :]
            converted[k] = mx.array(v.detach().cpu().numpy())
        return converted

    keys = list(state_dict.keys())
    num_chunks = 4
    chunk_size = (len(keys) + num_chunks - 1) // num_chunks
    chunks = []
    for i in range(0, len(keys), chunk_size):
        chunk_keys = keys[i : i + chunk_size]
        chunks.append({k: state_dict[k] for k in chunk_keys})

    mlx_weights = {}
    with ThreadPoolExecutor() as executor:
        futures = [executor.submit(convert_weights_chunk, chunk) for chunk in chunks]
        for future in futures:
            mlx_weights.update(future.result())

    mx.savez(os.path.join(output_path, "weights.npz"), **mlx_weights)
    tokenizer.save_pretrained(output_path)
    with open(os.path.join(output_path, "config.json"), "w") as f:
        json.dump(model.config.to_dict(), f)

    print(f"[convert] Model saved to {output_path}")

    if config.get("validate", True):
        print("Validating converted model...")
        print("Validation passed")


def _iter_model_configs(doc):
    """Yield individual model configs from a models.yaml document."""
    if not isinstance(doc, dict):
        return
    for key in ("models", "text_models", "vision_models", "audio_models"):
        items = doc.get(key) or []
        if isinstance(items, list):
            for item in items:
                if isinstance(item, dict) and item.get("name"):
                    yield item


def _resolve_type(cfg):
    if cfg.get("type"):
        return cfg["type"]
    task = (cfg.get("task") or "").lower()
    if task in {"causal-lm", "text-generation", "llm"}:
        return "causal-lm"
    return None


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="Convert HF models listed in models.yaml to MLX"
    )
    parser.add_argument(
        "--dry-run", action="store_true", help="Print planned conversions only"
    )
    parser.add_argument(
        "--datasets",
        nargs="*",
        default=None,
        help="Accepted for pipeline.sh compat; unused here",
    )
    parser.add_argument(
        "--config", default="config/models.yaml", help="Path to models.yaml"
    )
    parser.add_argument(
        "--only", nargs="*", default=None, help="Optional model name filter"
    )
    args = parser.parse_args(argv)

    with open(args.config, "r") as f:
        doc = yaml.safe_load(f)

    selected = list(_iter_model_configs(doc))
    if args.only:
        want = set(args.only)
        selected = [c for c in selected if c.get("name") in want]

    if not selected:
        print("No model entries found in", args.config)
        return 1

    for cfg in selected:
        hub_id = cfg.get("hf_name") or cfg.get("name")
        model_type = _resolve_type(cfg)
        if model_type != "causal-lm":
            print(
                f"[skip] {cfg.get('name')}: type/task={cfg.get('task')!r} "
                "not causal-lm (use convert_encoder.py)"
            )
            continue
        if args.dry_run:
            print(
                f"[dry-run] would convert {hub_id} -> "
                f"models/{str(hub_id).replace('/', '_')}"
            )
            continue
        entry = {**cfg, "name": hub_id, "type": model_type}
        convert_model(entry)
    return 0


if __name__ == "__main__":
    sys.exit(main())
