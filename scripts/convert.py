import mlx.core as mx
from transformers import AutoTokenizer, AutoModelForCausalLM
import os


def convert_model(model_name: str, output_path: str):
    """Convert a Hugging Face model to MLX format"""
    # Create output directory
    os.makedirs(output_path, exist_ok=True)
    
    # Load model and tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name)
    
    # Save MLX weights
    state_dict = model.state_dict()
    mlx_weights = {k: mx.array(v.detach().cpu().numpy()) for k, v in state_dict.items()}
    mx.savez(os.path.join(output_path, "weights.npz"), **mlx_weights)
    
    # Save tokenizer
    tokenizer.save_pretrained(output_path)
    
    print(f"[convert] Model saved to {output_path}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Convert Hugging Face model to MLX")
    parser.add_argument("--model", type=str, required=True, help="Model name or path")
    parser.add_argument("--output", type=str, default="models/mlx_converted", 
                        help="Output directory")
    args = parser.parse_args()
    
    # Run conversion
    convert_model(args.model, os.path.join(args.output, args.model.replace('/', '_')))
