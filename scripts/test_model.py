import mlx.core as mx
from mlx.utils import tree_unflatten
from transformers import AutoTokenizer
import numpy as np
import json
import yaml
import os
from datetime import datetime

# Import model classes
from phi2_model import Phi2Model

# Define test functions
def test_forward_pass(model, tokenizer, config):
    input_text = "This is a test"
    inputs = tokenizer(input_text, return_tensors="np")
    input_ids = mx.array(inputs["input_ids"])

    try:
        logits = model(input_ids)
        print(f"[test] Forward pass completed. Logits shape: {logits.shape}")
    except Exception as e:
        print(f"[test] Forward pass failed: {e}")

def test_text_generation(model, tokenizer, config):
    """Test text generation task"""
    prompt = "The future of AI is"
    inputs = tokenizer(prompt, return_tensors="np")
    input_ids = inputs["input_ids"]
    input_ids = mx.array(input_ids)

    for _ in range(50):
        logits = model(input_ids)
        next_token = mx.argmax(logits[:, -1, :], axis=-1)
        input_ids = mx.concatenate([input_ids, next_token[:, None]], axis=1)

    output = tokenizer.decode(input_ids[0])
    print(f"[test] Generated: {output}")

# Map of task to test function for additional tests beyond forward pass
TASK_TEST_FUNCS = {
    "text-generation": test_text_generation,
}

# Map of model name to model class
MODEL_CLASSES = {
    "microsoft/phi-2": Phi2Model,
    # Add other models here
}

def test_model(model_path: str, model_name: str, task: str):
    """Test a converted MLX model"""
    # Load configuration
    with open(f"{model_path}/config.json", "r") as f:
        config = json.load(f)

    # Load weights
    weights = mx.load(f"{model_path}/weights.npz")

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_path)

    # Get model class
    model_class = MODEL_CLASSES.get(model_name)
    if model_class is None:
        print(f"[test] No model class registered for {model_name}. Skipping")
        return None

    # Create model
    model = model_class(config)

    # Update model with weights
    model.update(tree_unflatten(list(weights.items())))

    # Initialize results dictionary
    results = {
        "model": model_name,
        "task": task,
        "forward_pass": {"status": "failed", "logits_shape": None, "error": None},
        "task_specific": {"status": "skipped", "output": None, "error": None}
    }

    # Run the forward pass test
    try:
        input_text = "This is a test"
        inputs = tokenizer(input_text, return_tensors="np")
        input_ids = mx.array(inputs["input_ids"])
        logits = model(input_ids)
        results["forward_pass"]["status"] = "success"
        results["forward_pass"]["logits_shape"] = list(logits.shape)
        print(f"[test] Forward pass completed. Logits shape: {logits.shape}")
    except Exception as e:
        results["forward_pass"]["error"] = str(e)
        print(f"[test] Forward pass failed: {e}")

    # Then, if there is a task-specific test, run it
    test_func = TASK_TEST_FUNCS.get(task)
    if test_func is not None:
        try:
            # For task-specific tests, capture the output if needed
            # Currently, we don't capture the output, just run and mark success
            test_func(model, tokenizer, config)
            results["task_specific"]["status"] = "success"
        except Exception as e:
            results["task_specific"]["status"] = "failed"
            results["task_specific"]["error"] = str(e)

    return results

if __name__ == "__main__":
    # Load models configuration
    config_path = "mlx_conversion/config/models.yaml"
    if not os.path.exists(config_path):
        print(f"[test] Configuration file {config_path} not found")
        exit(1)

    with open(config_path, "r") as f:
        models_config = yaml.safe_load(f)

    # Create results directory
    results_dir = "results/test_results"
    os.makedirs(results_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_file = os.path.join(results_dir, f"test_results_{timestamp}.json")
    
    all_results = []

    # Test each model in the 'models' list
    for model_info in models_config["models"]:
        model_name = model_info["name"]
        task = model_info["task"]
        model_path = f"models/mlx_converted/{model_name}"
        
        if not os.path.exists(model_path):
            print(f"[test] Model directory {model_path} does not exist. Skipping {model_name}")
            continue

        print(f"[test] Testing model: {model_name} (task: {task})")
        model_results = test_model(model_path, model_name, task)
        if model_results is not None:
            all_results.append(model_results)

    # Save results to JSON file
    with open(results_file, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"[test] Test results saved to {results_file}")
