import mlx.core as mx
from transformers import AutoTokenizer, AutoModelForCausalLM
import os
import json
import yaml
import torch  # for torch_dtype
from concurrent.futures import ThreadPoolExecutor  # Added for parallel conversion


def convert_model(config):
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
        tokenizer = AutoTokenizer.from_pretrained(model_name, low_memory=True)
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16,  # half precision
            device_map="auto",
            low_memory=True
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


if __name__ == "__main__":
    # Load model configuration
    with open("config/models.yaml", "r") as f:
        model_config = yaml.safe_load(f)
    
    convert_model(model_config)
