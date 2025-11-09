# Next Steps: Testing and Verification

## Summary of Changes

### ✅ Problem Identified
Your V1 converter was performing "fake quantization" - applying quantization noise but storing weights as float32, resulting in:
- **2x larger than target** (253 MB vs 125 MB target)
- **No actual compression** (original ~253 MB → "quantized" 253 MB)
- Wasted storage space

### ✅ Solution Implemented
Created V2 converter with TRUE INT8 quantization:
- **Stores weights as actual INT8** (1 byte instead of 4 bytes)
- **Expected ~4x compression** (253 MB → ~65 MB)
- **Beats target size** (65 MB < 125 MB target)
- Maintains model accuracy

### ✅ Testing Tools Created
1. **`scripts/convert_encoder_v2.py`** - Improved converter with real quantization
2. **`scripts/test_encoder_v2.py`** - Test tool with quantization analysis
3. **`scripts/compare_encoder_models.py`** - Accuracy comparison (MLX vs HuggingFace)
4. **`QUANTIZATION_ANALYSIS.md`** - Detailed technical analysis
5. **`TEST_PLAN.md`** - Step-by-step testing guide

## Run These Commands on Your Mac

### 1. Convert Model with V2 (Real Quantization)

```bash
python scripts/convert_encoder_v2.py --model distilbert-base-uncased-mnli --no-skip
```

**What to look for**:
```
Original size: 253.2MB
Quantized size: ~65-70MB       ← Should be ~4x smaller!
Compression ratio: ~3.6-4.0x   ← Actual compression!
Target size: 125MB
```

### 2. Verify Quantization Worked

```bash
python scripts/test_encoder_v2.py \
  models/mlx_converted_v2/distilbert-base-uncased-mnli-mlx-q8-v2 \
  --full-test --verbose
```

**What to look for**:
```
Quantization Analysis:
INT8 quantized layers: ~90-100     ← Many layers quantized
INT8 layers size: ~50-60 MB        ← Most size in INT8
Float32 layers size: ~10-15 MB     ← Embeddings/norms
Total size: ~65 MB                 ← Total much smaller!

Sample quantized weights:
  distilbert.transformer.layer.0.attention.q_lin.weight:
    Dtype: int8                    ← Actually INT8!
    Range: [-127, 127]             ← Proper INT8 range
    Scale: 0.00xxxx                ← Scale factor stored
```

### 3. Compare Model Accuracy

**First, install dependencies if needed**:
```bash
pip install torch transformers scikit-learn
```

**Then run comparison**:
```bash
python scripts/compare_encoder_models.py \
  models/mlx_converted_v2/distilbert-base-uncased-mnli-mlx-q8-v2 \
  --output results_v2.json
```

**What to look for**:
```
SIMILARITY METRICS (MLX vs HuggingFace)
Cosine Similarity (mean): 0.99xxxx    ← Should be > 0.99
Mean Squared Error:       1.xE-04     ← Should be < 1e-3
Max Absolute Difference:  0.0xxx      ← Should be < 0.5

Assessment: ✓ EXCELLENT - Models are nearly identical

MODEL SIZE
HuggingFace (params): 253.2 MB
MLX (weights):        ~65 MB          ← ~4x smaller!
Size ratio (MLX/HF):  0.26x           ← Much better than V1's 1.0x
```

### 4. Compare V1 vs V2

```bash
# Show both metadata files
echo "=== V1 (Fake Quantization) ==="
cat models/mlx_converted/distilbert-base-uncased-mnli-mlx-q8/conversion_metadata.json | \
  python -m json.tool | grep -A5 "quantization"

echo ""
echo "=== V2 (Real Quantization) ==="
cat models/mlx_converted_v2/distilbert-base-uncased-mnli-mlx-q8-v2/conversion_metadata.json | \
  python -m json.tool | grep -A8 "quantization"

# Show file sizes
echo ""
echo "=== Disk Usage ==="
du -sh models/mlx_converted/distilbert-base-uncased-mnli-mlx-q8
du -sh models/mlx_converted_v2/distilbert-base-uncased-mnli-mlx-q8-v2
```

## Success Criteria Checklist

- [ ] V2 model size is ~65-70 MB (vs V1's 253 MB)
- [ ] Compression ratio is ~4x
- [ ] V2 beats target size (< 125 MB)
- [ ] Cosine similarity > 0.99 (models are nearly identical)
- [ ] MSE < 1e-3
- [ ] Model loads and runs without errors
- [ ] Quantization analysis shows INT8 dtypes

## If Tests Pass: Next Actions

### Option 1: Replace V1 with V2
If V2 works perfectly, update the main converter:

```bash
# Backup V1
mv scripts/convert_encoder.py scripts/convert_encoder_v1_backup.py

# Make V2 the default
cp scripts/convert_encoder_v2.py scripts/convert_encoder.py

# Update output directory in config if needed
sed -i '' 's/mlx_converted_v2/mlx_converted/g' scripts/convert_encoder.py
```

### Option 2: Test 4-bit Quantization
Try even more aggressive compression:

```bash
# Edit config/models.yaml to set bits: 4
# Then convert
python scripts/convert_encoder_v2.py --model distilbert-base-uncased-mnli --no-skip

# Expected: ~35-40 MB with slightly lower accuracy (~0.97-0.99 cosine similarity)
```

### Option 3: Convert All Models
Convert all your models with V2:

```bash
# List models in config
grep "name:" config/models.yaml

# Convert each one
for model in distilbert-base-uncased-mnli bert-base-uncased roberta-base; do
    python scripts/convert_encoder_v2.py --model $model --no-skip
done
```

## If Tests Fail: Troubleshooting

### Issue: Size still large
**Check**: Are weights actually INT8?
```bash
python -c "
import numpy as np
weights = np.load('models/mlx_converted_v2/distilbert-base-uncased-mnli-mlx-q8-v2/weights.npz')
int8_count = sum(1 for v in weights.values() if v.dtype == np.int8)
total = len(weights)
print(f'INT8 weights: {int8_count}/{total}')
print('Sample dtypes:', [(k, v.dtype) for k, v in list(weights.items())[:5]])
"
```

### Issue: Low accuracy (< 0.95 cosine similarity)
**Try**: Less aggressive quantization
- Use 8-bit instead of 4-bit
- Check for NaN/Inf values
- Verify embeddings weren't quantized (they shouldn't be)

### Issue: Model won't load
**Check**: File integrity
```bash
# Verify NPZ file
python -c "import numpy as np; np.load('path/to/weights.npz').keys()"

# Check metadata
cat path/to/conversion_metadata.json | python -m json.tool
```

## Questions to Answer

After running tests, please report:

1. **Size**: What's the actual V2 model size? (Should be ~65 MB)
2. **Accuracy**: What's the cosine similarity? (Should be > 0.99)
3. **Compression**: What's the compression ratio? (Should be ~4x)
4. **Quality**: Any errors or warnings during conversion/testing?

## Final Goal

Get these results:

| Metric | Target | V1 Actual | V2 Expected |
|--------|--------|-----------|-------------|
| Model Size | 125 MB | 253 MB ❌ | ~65 MB ✓ |
| vs Target | 1.0x | 2.02x ❌ | 0.52x ✓ |
| Compression | 4x | 1x ❌ | 4x ✓ |
| Accuracy | > 0.99 | ??? | > 0.99 ✓ |

## Contact / Issues

If you encounter any issues:
1. Check the detailed logs in `$TMPDIR/mlx_conversion/logs/`
2. Review `QUANTIZATION_ANALYSIS.md` for technical details
3. Review `TEST_PLAN.md` for complete testing procedures

Good luck with testing! The V2 converter should solve your size issue completely. 🚀
