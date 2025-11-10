import mlx.core as mx
from mlx.utils import tree_unflatten
from transformers import AutoTokenizer
import numpy as np

# Simple generation function (placeholder)
def generate(prompt: str, model_weights: dict, tokenizer, max_length=50):
    """Simple text generation function"""
    inputs = tokenizer(prompt, return_tensors="np")
    input_ids = inputs["input_ids"]
    
    # This is a placeholder - you'll need to implement the actual model architecture
    print(f"[test] Prompt: '{prompt}'")
    print("[test] Note: Implement model architecture to use weights for generation")
    return tokenizer.decode(input_ids[0])

def test_model(model_path: str):
    """Test a converted MLX model"""
    # Load weights
    weights = mx.load(f"{model_path}/weights.npz")
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    
    # Test generation
    prompt = "The future of AI is"
    output = generate(prompt, weights, tokenizer)
    print(f"[test] Output: {output}")

if __name__ == "__main__":
    model_path = "models/mlx_converted/microsoft_phi-2"
    print(f"[test] Testing model at {model_path}")
    test_model(model_path)
