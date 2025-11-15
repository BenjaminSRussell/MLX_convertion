import argparse
import mlx.core as mx
import mlx.nn as nn
import time
import json
import os
import numpy as np
import yaml
from yaml import SafeLoader
from phi2_model import Phi2Model
from transformers import AutoTokenizer
from datasets import load_dataset


def calculate_perplexity(model, tokenizer, dataset, max_length=512):
    """calculates model perplexity on a dataset"""
    total_loss = 0
    total_count = 0
    
    for i, example in enumerate(dataset):
        if i >= 10:  # Limit to 10 examples for quick evaluation
            break
            
        text = example["text"]
        inputs = tokenizer(text, return_tensors="np", truncation=True, max_length=max_length)
        input_ids = mx.array(inputs["input_ids"])
        
        # Forward pass
        logits = model(input_ids)
        
        # Shift so that tokens < n predict n
        shift_logits = logits[:, :-1, :]
        shift_labels = input_ids[:, 1:]
        
        # Calculate loss
        loss = nn.losses.cross_entropy(shift_logits, shift_labels)
        total_loss += mx.mean(loss).item()
        total_count += 1
    
    perplexity = np.exp(total_loss / total_count)
    return perplexity


def measure_inference_speed(model, tokenizer, prompt: str, num_runs=10):
    """measures avg inference time"""
    inputs = tokenizer(prompt, return_tensors="np")
    input_ids = mx.array(inputs["input_ids"])
    
    # Warmup
    for _ in range(2):
        model(input_ids)
    
    # Timed runs
    start_time = time.time()
    for _ in range(num_runs):
        model(input_ids)
    mx.eval()  # Ensure all operations are completed
    end_time = time.time()
    
    avg_time = (end_time - start_time) / num_runs
    return avg_time


def compare_models(model_path: str, dataset: str, dataset_name: str = "wikitext"):
    """compares original vs quantized models"""
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    
    # Load dataset
    dataset = load_dataset(dataset, split="test")
    
    # Load original model
    print("[evaluate] Loading original model")
    with open(os.path.join(model_path, "config.json")) as f:
        config = json.load(f)
    original_model = Phi2Model(config)
    weights = mx.load(os.path.join(model_path, "weights.npz"))
    original_model.update(weights)
    
    # Load quantized model
    print("[evaluate] Loading quantized model")
    quantized_model = Phi2Model(config)
    quant_weights = mx.load(os.path.join(model_path, "weights_8bit.npz"))
    quantized_model.update(quant_weights)
    
    # Test prompt
    prompt = "The future of artificial intelligence will"
    
    # Evaluate both models
    results = {
        "original": {},
        "quantized": {}
    }
    
    # Perplexity
    print("[evaluate] Calculating perplexity for original model")
    results["original"]["perplexity"] = calculate_perplexity(original_model, tokenizer, dataset)
    print("[evaluate] Calculating perplexity for quantized model")
    results["quantized"]["perplexity"] = calculate_perplexity(quantized_model, tokenizer, dataset)
    
    # Inference speed
    print("[evaluate] Measuring inference speed for original model")
    results["original"]["inference_time"] = measure_inference_speed(original_model, tokenizer, prompt)
    print("[evaluate] Measuring inference speed for quantized model")
    results["quantized"]["inference_time"] = measure_inference_speed(quantized_model, tokenizer, prompt)
    
    # Accuracy comparison (using a simple test)
    print("[evaluate] Running accuracy comparison")
    orig_output = original_model(mx.array(tokenizer.encode(prompt, return_tensors="np")))
    quant_output = quantized_model(mx.array(tokenizer.encode(prompt, return_tensors="np")))
    
    # Calculate similarity
    similarity = mx.mean(mx.abs(orig_output - quant_output)).item()
    results["output_similarity"] = similarity
    
    # Print report
    print("\n=== Evaluation Report ===")
    print(f"Model: {model_path}")
    print(f"Dataset: {dataset_name}")
    print(f"\n--- Perplexity ---")
    print(f"Original: {results['original']['perplexity']:.2f}")
    print(f"Quantized: {results['quantized']['perplexity']:.2f}")
    print(f"Difference: {abs(results['original']['perplexity'] - results['quantized']['perplexity']):.2f}")
    print(f"\n--- Inference Speed (seconds) ---")
    print(f"Original: {results['original']['inference_time']:.4f}")
    print(f"Quantized: {results['quantized']['inference_time']:.4f}")
    print(f"Speedup: {results['original']['inference_time'] / results['quantized']['inference_time']:.2f}x")
    print(f"\n--- Output Similarity ---")
    print(f"Mean Absolute Difference: {similarity:.6f}")
    
    # Save results
    report_path = os.path.join(model_path, "evaluation_report.json")
    with open(report_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nReport saved to {report_path}")


def main():
    parser = argparse.ArgumentParser(description="Model Comparison Tool")
    parser.add_argument("--config", type=str, help="Path to config file")
    args = parser.parse_args()

    # Load dataset configuration from datasets.yaml
    project_root = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
    config_path = os.path.join(project_root, "config", "datasets.yaml")
    with open(config_path, 'r') as f:
        dataset_config = yaml.safe_load(f)

    # Load model configurations
    config_path = os.path.join(project_root, "config", "models.yaml")
    with open(config_path, 'r') as f:
        models_config = yaml.safe_load(f)

    print("\n=== Starting model evaluation ===")
    for category, models in models_config.items():
        print(f"\nProcessing {category} models:")
        for model in models:
            model_name = model['name']
            model_path = model['path']
            dataset_name = model.get('dataset', dataset_config['default_dataset'])
            dataset = load_dataset(dataset_name, split="test")
            print(f"\nEvaluating model: {model_name} with dataset: {dataset_name}")
            try:
                compare_models(model_path, dataset, dataset_name)
                print(f"  - Evaluation completed successfully")
            except Exception as e:
                import traceback
                print(f"  - Error evaluating model: {str(e)}")
                print(traceback.format_exc())


if __name__ == "__main__":
    main()
