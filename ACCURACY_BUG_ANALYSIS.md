# CRITICAL BUG: Invalid Accuracy Comparison

## The Problem

The `compare_encoder_models.py` script has a **fundamental flaw** that makes accuracy testing invalid.

### What It's Comparing

**HuggingFace model** (line 77):
```python
embeddings = outputs.last_hidden_state[:, 0, :].numpy()  # [CLS] token
```
- Runs **full model inference** (all 12 transformer layers)
- Gets output from **last layer** (after attention, FFN, normalization)
- This is the **final output** of the model

**MLX model** (line 110):
```python
cls_embedding = token_embeddings[0]  # First token
```
- Gets **raw word embeddings** from lookup table
- This is the **input** to the model (before any layers)
- No transformer layers, no attention, nothing

### Why This is Wrong

**This is comparing apples to oranges:**
- HF: Model OUTPUT (after 12 layers of processing)
- MLX: Model INPUT (just word embeddings)

**Example:**
```
Input text: "This is a test"
HF:  word_embedding → layer1 → layer2 → ... → layer12 → [output: (768,)]
MLX: word_embedding → [stop here!] → [output: (768,)]
```

They're extracting from completely different points in the computation graph!

## Why Accuracy Metrics Are Meaningless

The comparison is measuring:
- **Cosine similarity between**: Final model output vs. raw word embeddings
- **This has nothing to do with quantization quality!**

The cosine similarity would be low (~0.3-0.7) even if quantization was perfect, because they're comparing different representations.

## What Should Be Compared

### Option 1: Weight-Level Comparison (CORRECT)
Compare quantization error at the weight level:

```python
# Original weights (HF)
original_weight = hf_model.state_dict()['layer.0.attention.q.weight']

# Quantized then dequantized weights (MLX)
mlx_weight = mlx_weights['layer.0.attention.q.weight']

# Compare these directly
error = np.mean((original_weight - mlx_weight) ** 2)
```

### Option 2: Full Model Comparison (REQUIRES MLX IMPLEMENTATION)
Implement the full transformer in MLX, then compare outputs:

```python
# HF: Full forward pass
hf_output = hf_model(**inputs).last_hidden_state

# MLX: Full forward pass (needs implementation!)
mlx_output = mlx_model_forward(mlx_weights, inputs)

# Now compare these
cosine_similarity(hf_output, mlx_output)
```

## The Real Test: Weight Quantization Error

What we actually want to measure:

```python
For each weight matrix:
    original = float32 weight from HF model
    quantized = int8 quantized weight
    dequantized = int8 → float32 conversion

    quantization_error = MSE(original, dequantized)
```

Expected error for 8-bit quantization:
- MSE: ~1e-5 to 1e-4 (very small)
- Relative error: < 0.5%
- Max absolute error: ~0.01

## Fix Required

The comparison script needs to:
1. Load HF model weights
2. Load MLX dequantized weights
3. Compare weights layer-by-layer
4. Report quantization error per layer
5. NOT try to compare model outputs (we don't have full MLX model)

## Impact

The current comparison script:
- ❌ Does NOT test quantization accuracy
- ❌ Compares wrong things (input vs output)
- ❌ Metrics are meaningless
- ❌ Would show low similarity even with perfect quantization

We need a proper weight-level comparison instead.
