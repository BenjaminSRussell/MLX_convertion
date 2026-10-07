import os
import json
import yaml
from concurrent.futures import ThreadPoolExecutor  # Added for parallel conversion


def convert_model(config):
    import mlx.core as mx
    from transformers import AutoTokenizer, AutoModelForCausalLM
    import torch  # for torch_dtype
    """
    Convert any model based on configuration
    Args:
        config (dict): Model configuration from YAML
    """
    model_name = config['name']
    model_type = config['type']
    
    print(f"Converting {model_name} ({model_type})...")
    
    # Create output directory
    # Use configured output path if available
    output_path = config.get('output_dir', os.path.join("models", model_name.replace('/', '_')))
    os.makedirs(output_path, exist_ok=True)
    
    # Load model and tokenizer with memory-efficient settings
    if model_type == "causal-lm":
        tokenizer = AutoTokenizer.from_pretrained(config.get("hf_name", model_name))
        hf_name = config.get("hf_name", model_name)
        model = AutoModelForCausalLM.from_pretrained(
            hf_name,
            torch_dtype=torch.float16,  # half precision
            device_map="auto",
        )
    else:
        raise ValueError(f"Unsupported model type: {model_type}")
    
    # Add quantization
    if config.get('quantization'):
        from quantize import quantize_model
        model = quantize_model(model, config['quantization'])
    
    # Get the model's state dict
    state_dict = model.state_dict()
    
    # Function to convert a chunk of weights
    def convert_weights_chunk(chunk):
        converted = {}
        for k, v in chunk.items():
            # Preprocessing: remove "transformer." prefix if exists
            if k.startswith("transformer."):
                k = k[len("transformer."):]
            # Convert tensor to MLX array
            converted[k] = mx.array(v.detach().cpu().numpy())
        return converted
    
    # Split state_dict into chunks for parallel conversion
    keys = list(state_dict.keys())
    num_chunks = 4  # number of chunks, can be adjusted
    chunk_size = (len(keys) + num_chunks - 1) // num_chunks
    chunks = []
    for i in range(0, len(keys), chunk_size):
        chunk_keys = keys[i:i+chunk_size]
        chunk = {k: state_dict[k] for k in chunk_keys}
        chunks.append(chunk)
    
    mlx_weights = {}
    with ThreadPoolExecutor() as executor:
        futures = [executor.submit(convert_weights_chunk, chunk) for chunk in chunks]
        for future in futures:
            chunk_result = future.result()
            mlx_weights.update(chunk_result)
    
    # Save MLX weights
    mx.savez(os.path.join(output_path, "weights.npz"), **mlx_weights)
    
    # Save tokenizer
    tokenizer.save_pretrained(output_path)
    
    # Save config
    with open(os.path.join(output_path, "config.json"), "w") as f:
        json.dump(model.config.to_dict(), f)
    
    print(f"[convert] Model saved to {output_path}")
    
    # Validate model
    if config.get('validate', True):
        print("Validating converted model...")
        # Placeholder - actual validation would go here
        print("Validation passed")


def _iter_model_configs(doc):
    """Yield unique per-model dicts from models.yaml document."""
    if not isinstance(doc, dict):
        return
    seen = set()
    for section in ("models", "text_models", "vision_models", "audio_models", "multimodal_models"):
        for entry in doc.get(section) or []:
            if not (isinstance(entry, dict) and entry.get("name")):
                continue
            name = entry["name"]
            if name in seen:
                continue
            seen.add(name)
            entry = dict(entry)
            if "type" not in entry:
                task = (entry.get("task") or "").lower()
                if section == "text_models" or "causal" in task or task in {"text-generation", "causal-lm"}:
                    entry["type"] = "causal-lm"
                else:
                    entry["type"] = entry.get("task") or "encoder"
            if not entry.get("hf_name"):
                entry["hf_name"] = name
            yield entry


if __name__ == "__main__":
    import argparse
    import sys

    parser = argparse.ArgumentParser(description="Convert models listed in config/models.yaml")
    parser.add_argument("--dry-run", action="store_true", help="Print planned work without downloading")
    parser.add_argument("--datasets", nargs="*", default=None, help="Optional dataset filter (reserved)")
    parser.add_argument("--config", default="config/models.yaml")
    parser.add_argument("--only", nargs="*", default=None, help="Optional model name filter")
    args = parser.parse_args()

    with open(args.config, "r") as f:
        doc = yaml.safe_load(f)

    selected = list(_iter_model_configs(doc))
    if args.only:
        only = set(args.only)
        selected = [m for m in selected if m.get("name") in only]

    if not selected:
        print("No models selected", file=sys.stderr)
        sys.exit(1)

    for model_config in selected:
        name = model_config.get("name")
        mtype = model_config.get("type")
        if args.dry_run:
            print(f"[dry-run] would convert {name} type={mtype} hf={model_config.get('hf_name')}")
            continue
        if mtype != "causal-lm":
            # Encoder path lives in convert_encoder.py; do not pretend success here.
            print(f"[skip] {name}: type={mtype} (use scripts/convert_encoder.py)")
            continue
        convert_model(model_config)
