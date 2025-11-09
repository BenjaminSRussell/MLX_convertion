# MLX Encoder Model Size Analysis and Solutions

## Problem Summary

The converted MLX model is **2.02x larger** than the target size despite using 8-bit quantization:
- **Target**: 125 MB
- **Actual**: 253.2 MB
- **Expected with 8-bit**: ~65-70 MB (original model is ~256MB in fp32, 8-bit should be ~4x smaller)

## Root Cause Analysis

### Issue 1: Fake Quantization (Primary Issue)
**Location**: `scripts/convert_encoder.py:74-93`

The quantization code performs these steps:
1. ✓ Quantizes weights to INT8 range [-128, 127]
2. ✗ **Immediately dequantizes back to float32** (line 88)
3. ✗ Stores dequantized float32 weights (still 4 bytes per param)

```python
# Line 87-88: The problem
# Dequantize for storage (MLX doesn't have native int8 storage yet)
w_dequant = (w_quant - zero_point) * scale
```

**Result**: This is "fake quantization" - it applies quantization noise but provides **no size reduction**.

### Issue 2: Extra Metadata Parameters
**Location**: `scripts/convert_encoder.py:92-93`

For each quantized weight, the code adds two extra parameters:
- `{name}.scale` - quantization scale factor
- `{name}.zero_point` - quantization zero point

This adds ~30-40% extra parameters, **increasing** size instead of reducing it.

## Solutions

### Solution 1: Use MLX Quantization API (Recommended)
MLX 0.29.3+ has native quantization support in `mlx.nn.quantize`:

```python
import mlx.nn as nn

# Quantize model using MLX native API
quantized_model = nn.quantize(model, bits=8, group_size=64)
```

**Advantages**:
- Native INT8 storage (1 byte per param)
- ~4x size reduction for 8-bit
- ~8x size reduction for 4-bit
- Optimized kernels for inference
- Proper scale/zero-point handling

### Solution 2: Store Weights as INT8 NumPy Arrays
Manually store quantized weights in int8 dtype:

```python
# Store actual INT8 values
w_quant_int8 = w_quant.astype(np.int8)
np_weights[name] = w_quant_int8
np_weights[f"{name}.scale"] = scale
np_weights[f"{name}.zero_point"] = zero_point
```

**Advantages**:
- True 4x size reduction
- Works with current codebase

**Disadvantages**:
- Need to dequantize during inference
- Slower than native MLX quantization

### Solution 3: Skip Non-Essential Layers (Partial Fix)
The code already skips embeddings, norms, and biases (line 64). We could:
- Only quantize largest layers (attention, FFN)
- Use higher precision for critical layers

**Expected savings**: 10-20% additional size reduction

### Solution 4: Use mlx-lm Quantization (If Supported)
Check if `mlx_lm.convert` has encoder support with proper quantization:

```bash
mlx_lm.convert --model distilbert-base-uncased-mnli --quantize
```

## Recommended Implementation Plan

### Phase 1: Investigate MLX Native Quantization
1. Check MLX version: `mlx.__version__`
2. Review MLX quantization API docs
3. Test if MLX quantization works with encoder models
4. Implement native quantization in converter

### Phase 2: Implement Proper INT8 Storage (Fallback)
If MLX native doesn't work with encoders:
1. Modify `quantize_weights()` to return INT8 arrays
2. Update `save_mlx_model()` to store INT8
3. Update `load_mlx_model()` to dequantize on load

### Phase 3: Comprehensive Testing
1. **Accuracy**: Compare outputs (cosine similarity > 0.99)
2. **Size**: Verify ~4x reduction for 8-bit
3. **Speed**: Benchmark inference time
4. **Quality**: Test on downstream tasks (classification, etc.)

## Testing Methodology

### 1. Output Similarity Metrics
- **Cosine Similarity**: > 0.99 (excellent), > 0.95 (acceptable)
- **Mean Squared Error**: < 1e-4
- **Max Absolute Difference**: < 0.1

### 2. Downstream Task Performance
Test on actual tasks (e.g., MNLI for distilbert-base-uncased-mnli):
- Accuracy difference: < 1%
- F1 score difference: < 1%

### 3. Inference Speed
- Should be 2-3x faster than HuggingFace on Apple Silicon
- Latency: < 10ms for single inference

### 4. Memory Usage
- Peak memory should be ~1/4 of original during inference

## Expected Outcomes

### 8-bit Quantization (INT8)
- **Size**: 65-70 MB (vs current 253 MB)
- **Accuracy**: > 99.5% cosine similarity
- **Speed**: 2-3x faster
- **Use case**: Production deployment

### 4-bit Quantization (INT4)
- **Size**: 35-40 MB
- **Accuracy**: > 98% cosine similarity (slight degradation)
- **Speed**: 3-5x faster
- **Use case**: Edge deployment, mobile

## Next Steps

1. ✓ Create comparison test script
2. ⏳ Run baseline comparison (current fake quantization)
3. ⏳ Implement proper INT8 storage
4. ⏳ Run comparison with real quantization
5. ⏳ Test different bit widths (4-bit, 8-bit)
6. ⏳ Document findings and recommendations

## References

- MLX Quantization: https://ml-explore.github.io/mlx/build/html/usage/quantization.html
- MLX NN Quantize: https://ml-explore.github.io/mlx/build/html/python/_autosummary/mlx.nn.quantize.html
