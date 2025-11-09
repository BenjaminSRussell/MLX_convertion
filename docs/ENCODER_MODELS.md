# Converting Encoder Models to MLX

This guide explains how to convert encoder models (BERT, DistilBERT, RoBERTa, DeBERTa, etc.) to MLX format.

## Why a Separate Converter?

The standard `mlx_lm` package only supports **language models** (decoder-only and encoder-decoder models like GPT, Llama, T5, etc.). Encoder models like BERT require a custom conversion process using MLX's low-level API.

## Supported Models

The encoder converter supports:
- **BERT** and variants (DistilBERT, MobileBERT, ModernBERT)
- **RoBERTa** and variants
- **DeBERTa** and variants
- **Sentence Transformers** (MiniLM, MPNet, etc.)
- Any encoder-only model from HuggingFace Transformers

## Installation

Ensure you have the required dependencies:

```bash
pip install mlx transformers torch numpy pyyaml
```

## Usage

### Convert a Single Model

```bash
python scripts/convert_encoder.py --model distilbert-base-uncased-mnli
```

### Convert with Custom Output Directory

```bash
python scripts/convert_encoder.py --model distilbert-base-uncased-mnli --output-dir models/custom_path
```

### Dry Run (Preview Conversion)

```bash
python scripts/convert_encoder.py --model distilbert-base-uncased-mnli --dry-run
```

### Force Reconversion

```bash
python scripts/convert_encoder.py --model distilbert-base-uncased-mnli --no-skip
```

## Configuration

Models are configured in `config/models.yaml`. Example:

```yaml
models:
  - name: "distilbert-base-uncased-mnli"
    hf_name: "typeform/distilbert-base-uncased-mnli"
    task: "zero-shot-classification"
    quantization:
      bits: 8
      dtype: "int8"
      target_size_mb: 125
      group_size: 64
```

## Quantization

The converter supports:
- **8-bit quantization**: Standard INT8 quantization for ~4x size reduction
- **4-bit quantization**: Aggressive quantization for ~8x size reduction (experimental)

Quantization parameters:
- `bits`: Number of bits (4, 8, or 32 for no quantization)
- `group_size`: Group size for quantization (default: 64)
- `target_size_mb`: Target model size after quantization

## Testing Converted Models

Test that your converted model loads correctly:

```bash
python scripts/test_encoder.py models/mlx_converted/distilbert-base-uncased-mnli-mlx-q8
```

Run full tests (loading, tokenization, embedding extraction):

```bash
python scripts/test_encoder.py models/mlx_converted/distilbert-base-uncased-mnli-mlx-q8 --full-test
```

## Output Format

Converted models are saved in the following structure:

```
models/mlx_converted/distilbert-base-uncased-mnli-mlx-q8/
├── weights.npz              # Model weights in NumPy format
├── config.json              # Model configuration
├── tokenizer_config.json    # Tokenizer configuration
├── tokenizer.json           # Tokenizer data
├── special_tokens_map.json  # Special tokens
└── conversion_metadata.json # Conversion metadata
```

## Loading Models in Your Code

```python
import numpy as np
import mlx.core as mx
from pathlib import Path
from transformers import AutoTokenizer

def load_mlx_encoder(model_path: str):
    model_path = Path(model_path)

    # Load weights
    np_weights = np.load(model_path / "weights.npz")
    mlx_weights = {k: mx.array(v) for k, v in np_weights.items()}

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(str(model_path))

    return mlx_weights, tokenizer

# Usage
weights, tokenizer = load_mlx_encoder("models/mlx_converted/distilbert-base-uncased-mnli-mlx-q8")
```

## Limitations

1. **Inference Implementation**: This converter creates MLX-format weights, but you'll need to implement the model architecture in MLX for inference. See `scripts/test_encoder.py` for basic examples.

2. **Quantization Format**: MLX's quantization support is evolving. The current implementation stores dequantized weights with scale/zero-point metadata.

3. **Apple Silicon Only**: MLX only runs on Apple Silicon (M1, M2, M3, M4 chips).

## Troubleshooting

### Model Type Not Supported

If you get an error about model type not being supported, verify that:
1. The model is an encoder-only model (not a language model)
2. The model architecture is supported by transformers library

### Out of Memory

For large models, try:
- Using 4-bit quantization instead of 8-bit
- Processing on a machine with more RAM
- Reducing the model size in your config

### Import Errors

Make sure you're running in a virtual environment with all dependencies installed:

```bash
source .venv/bin/activate  # or your venv path
pip install -r requirements.txt
```

## Comparison with mlx_lm

| Feature | mlx_lm | convert_encoder.py |
|---------|--------|-------------------|
| Supported Models | Decoder-only, Encoder-Decoder LMs | Encoder-only models |
| Examples | Llama, GPT, Mistral, T5, BART | BERT, RoBERTa, DistilBERT |
| Quantization | 2-bit, 4-bit, 8-bit | 4-bit, 8-bit |
| Inference | Built-in | Requires custom implementation |

## Next Steps

After converting your model:
1. Test it with `test_encoder.py`
2. Implement inference in MLX (see examples in the MLX documentation)
3. Benchmark performance against PyTorch baseline
4. Integrate into your application

## Resources

- [MLX Documentation](https://ml-explore.github.io/mlx/)
- [MLX Examples](https://github.com/ml-explore/mlx-examples)
- [HuggingFace Transformers](https://huggingface.co/docs/transformers)
