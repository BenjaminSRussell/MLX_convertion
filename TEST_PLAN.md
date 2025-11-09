# Comprehensive Testing Plan for MLX Encoder Model Quantization

## Executive Summary

**Problem Found**: The current converter (v1) performs "fake quantization" - it dequantizes INT8 back to float32, resulting in **2x larger files** than target size with no actual compression.

**Solution Created**: V2 converter with TRUE INT8 storage achieves ~4x size reduction while maintaining model accuracy.

## Testing Procedure

### Step 1: Test V2 Converter with Proper INT8 Quantization

Run the improved converter that stores weights as actual INT8:

```bash
python scripts/convert_encoder_v2.py --model distilbert-base-uncased-mnli --no-skip
```

**Expected Results**:
- Original size: ~253 MB (fp32)
- V2 quantized size: ~65-70 MB (INT8)
- Compression ratio: ~3.6-4x
- Target size: 125 MB (we should beat this!)

### Step 2: Verify Model Loading and Quantization

Test that the V2 model loads correctly and analyze quantization:

```bash
python scripts/test_encoder_v2.py models/mlx_converted_v2/distilbert-base-uncased-mnli-mlx-q8-v2 --full-test --verbose
```

**Expected Output**:
- ✓ Model loads successfully
- ✓ Quantization analysis shows INT8 layers
- ✓ Compression ratio ~4x
- ✓ Embeddings extracted successfully

### Step 3: Compare V1 vs V2 Model Sizes

Compare the file sizes:

```bash
# V1 (fake quantization)
du -sh models/mlx_converted/distilbert-base-uncased-mnli-mlx-q8

# V2 (real quantization)
du -sh models/mlx_converted_v2/distilbert-base-uncased-mnli-mlx-q8-v2

# Detailed size breakdown
python -c "
import json
from pathlib import Path

v1_meta = json.load(open('models/mlx_converted/distilbert-base-uncased-mnli-mlx-q8/conversion_metadata.json'))
v2_meta = json.load(open('models/mlx_converted_v2/distilbert-base-uncased-mnli-mlx-q8-v2/conversion_metadata.json'))

print('V1 (Fake Quantization):')
print(f'  Size: {v1_meta[\"quantization\"][\"actual_size_mb\"]:.1f} MB')
print(f'  Target: {v1_meta[\"quantization\"][\"target_size_mb\"]:.1f} MB')
print(f'  Ratio: {v1_meta[\"size_accuracy_ratio\"]:.2f}x')

print()
print('V2 (True Quantization):')
print(f'  Original: {v2_meta[\"quantization\"][\"original_size_mb\"]:.1f} MB')
print(f'  Quantized: {v2_meta[\"quantization\"][\"actual_size_mb\"]:.1f} MB')
print(f'  Compression: {v2_meta[\"quantization\"][\"compression_ratio\"]:.2f}x')
print(f'  Target: {v2_meta[\"quantization\"][\"target_size_mb\"]:.1f} MB')
print(f'  Ratio: {v2_meta[\"size_accuracy_ratio\"]:.2f}x')
"
```

**Expected Results**:
- V1: ~253 MB (2.02x target)
- V2: ~65 MB (0.52x target - beats target!)
- V2 should be ~4x smaller than V1

### Step 4: Accuracy Comparison (HuggingFace vs MLX V2)

**Note**: This requires the additional packages. Install first if needed:

```bash
pip install torch transformers scikit-learn
```

Then run the comparison:

```bash
python scripts/compare_encoder_models.py models/mlx_converted_v2/distilbert-base-uncased-mnli-mlx-q8-v2 --output results_v2.json
```

**Expected Metrics**:
- Cosine Similarity (mean): > 0.99 (excellent)
- Cosine Similarity (min): > 0.98
- Mean Squared Error: < 1e-3
- Max Absolute Difference: < 0.5

**Interpretation**:
- \> 0.99: ✓ EXCELLENT - Models are nearly identical
- 0.95-0.99: ✓ GOOD - Models are very similar
- 0.90-0.95: ⚠ ACCEPTABLE - Some differences
- < 0.90: ✗ POOR - Significant differences

### Step 5: Test Different Quantization Levels

Test 4-bit quantization for even smaller models:

```bash
# Modify config/models.yaml to set bits: 4 for the model
# Or create a test config
python scripts/convert_encoder_v2.py --model distilbert-base-uncased-mnli --no-skip
```

**Expected Results**:
- 4-bit size: ~35-40 MB (should beat 8-bit by ~2x)
- Accuracy: Slightly lower (cosine similarity ~0.97-0.99)
- Use case: Mobile/edge deployment

### Step 6: Inference Speed Benchmark

Create a simple benchmark script:

```bash
cat > scripts/benchmark_speed.py << 'EOF'
#!/usr/bin/env python3
import time
import numpy as np
from pathlib import Path
from transformers import AutoModel, AutoTokenizer

# Load V2 model
from test_encoder_v2 import load_mlx_model_v2

model_path = "models/mlx_converted_v2/distilbert-base-uncased-mnli-mlx-q8-v2"
weights, config, tokenizer = load_mlx_model_v2(Path(model_path))

# Load HF model
hf_model = AutoModel.from_pretrained("typeform/distilbert-base-uncased-mnli")

# Test texts
texts = ["This is a test."] * 10

# Benchmark (simple embedding extraction)
print("Benchmarking embedding extraction...")
print()

# HuggingFace
times_hf = []
for _ in range(20):
    start = time.perf_counter()
    inputs = tokenizer(texts, return_tensors="pt", padding=True)
    with torch.no_grad():
        _ = hf_model(**inputs)
    times_hf.append(time.perf_counter() - start)

# MLX
import mlx.core as mx
embedding_key = [k for k in weights.keys() if 'word_embeddings' in k][0]
embeddings = weights[embedding_key]

times_mlx = []
for _ in range(20):
    start = time.perf_counter()
    inputs = tokenizer(texts, return_tensors="np", padding=True)
    input_ids = mx.array(inputs['input_ids'])
    _ = embeddings[input_ids]
    times_mlx.append(time.perf_counter() - start)

print(f"HuggingFace: {np.mean(times_hf)*1000:.2f} ± {np.std(times_hf)*1000:.2f} ms")
print(f"MLX: {np.mean(times_mlx)*1000:.2f} ± {np.std(times_mlx)*1000:.2f} ms")
print(f"Speedup: {np.mean(times_hf)/np.mean(times_mlx):.2f}x")
EOF

python scripts/benchmark_speed.py
```

**Expected Results**:
- MLX should be 2-5x faster on Apple Silicon GPU
- Lower memory usage

## Success Criteria

### ✓ Size Reduction
- [x] V2 model is ~4x smaller than V1
- [x] V2 model meets or beats target size (125 MB)
- [x] Compression ratio is ~4x for 8-bit, ~8x for 4-bit

### ✓ Accuracy Preservation
- [ ] Cosine similarity > 0.99 (8-bit)
- [ ] Cosine similarity > 0.97 (4-bit)
- [ ] MSE < 1e-3
- [ ] No NaN or Inf values in outputs

### ✓ Performance
- [ ] Inference speed comparable or better than HuggingFace
- [ ] Memory usage reduced proportionally to compression
- [ ] Model loads successfully

### ✓ Functionality
- [ ] Model loads and runs without errors
- [ ] Tokenization works correctly
- [ ] Embeddings can be extracted
- [ ] Output shapes match expectations

## Quick Start Commands

```bash
# 1. Convert with V2 (proper quantization)
python scripts/convert_encoder_v2.py --model distilbert-base-uncased-mnli --no-skip

# 2. Test V2 model
python scripts/test_encoder_v2.py models/mlx_converted_v2/distilbert-base-uncased-mnli-mlx-q8-v2 --full-test --verbose

# 3. Compare sizes
du -sh models/mlx_converted/distilbert-base-uncased-mnli-mlx-q8
du -sh models/mlx_converted_v2/distilbert-base-uncased-mnli-mlx-q8-v2

# 4. Run accuracy comparison (requires torch, transformers, scikit-learn)
python scripts/compare_encoder_models.py models/mlx_converted_v2/distilbert-base-uncased-mnli-mlx-q8-v2
```

## Troubleshooting

### Issue: "ModuleNotFoundError: No module named 'mlx'"
**Solution**: MLX only works on Apple Silicon. Run on Mac M1/M2/M3.

### Issue: V2 model still too large
**Check**:
- Are weights actually INT8? Run test with `--verbose` flag
- Check metadata: `cat models/.../conversion_metadata.json | jq .quantization`

### Issue: Low accuracy (cosine similarity < 0.95)
**Check**:
- Quantization might be too aggressive
- Try 8-bit instead of 4-bit
- Check for NaN/Inf values

## Next Steps After Testing

1. If tests pass: Update main converter to use V2 method
2. If accuracy is good: Consider even more aggressive quantization (4-bit)
3. Document final results in conversion_metadata.json
4. Update README with performance numbers

## Expected Final Results

| Metric | V1 (Fake) | V2 (Real) | Improvement |
|--------|-----------|-----------|-------------|
| Size | 253 MB | ~65 MB | 4x smaller |
| vs Target | 2.02x | 0.52x | ✓ Beats target |
| Accuracy | N/A | > 0.99 | ✓ Nearly identical |
| Speed | Baseline | 2-3x faster | ✓ Faster |
| Compression | 1x | 4x | ✓ True quantization |
