# Troubleshooting Guide

## Library Version Issues

### Problem: Compilation Errors or Version Conflicts

The pipeline requires specific version ranges to work correctly. Here's how to fix version issues:

### Quick Fix

```bash
# Option 1: Install stable, tested versions
pip install -r requirements-stable.txt

# Option 2: Check compatibility and auto-fix
python scripts/check_compatibility.py --fix
```

### Manual Version Check

```bash
# Check current versions
python scripts/check_compatibility.py

# Check what's installed
pip list | grep -E "torch|transformers|numpy"
```

---

## Common Issues

### Issue 1: NumPy 2.x Breaking Changes

**Symptoms:**
- `AttributeError` with NumPy arrays
- Import errors related to NumPy

**Cause:** NumPy 2.x has breaking API changes

**Fix:**
```bash
pip install "numpy<2.0.0"
# Or use stable version
pip install numpy==1.24.3
```

### Issue 2: PyTorch Quantization Deprecation Warnings

**Symptoms:**
- Warnings about `torch.ao.quantization` being deprecated
- Messages about migrating to torchao

**Fix:** Already handled in the code! The converter automatically:
1. Suppresses deprecation warnings
2. Tries new API first, falls back to old API
3. Works with PyTorch 2.0 through 2.9+

If you still see warnings:
```bash
# Use warnings filter
export PYTHONWARNINGS="ignore::DeprecationWarning"
python scripts/convert_pytorch.py --model MODEL_NAME
```

### Issue 3: Transformers Version Conflicts

**Symptoms:**
- Model loading errors
- Missing methods on tokenizer
- `AttributeError` on model objects

**Fix:**
```bash
# Install compatible version
pip install "transformers>=4.30.0,<5.0.0"
# Or use stable version
pip install transformers==4.35.0
```

### Issue 4: CUDA/GPU Issues

**Symptoms:**
- CUDA errors
- Out of memory errors
- GPU detection issues

**Fix:** The converter works fine on CPU!
```bash
# Force CPU usage (already default)
export CUDA_VISIBLE_DEVICES=""
python scripts/convert_pytorch.py --model MODEL_NAME
```

---

## Version Compatibility Matrix

| Package | Minimum | Maximum | Recommended | Reason |
|---------|---------|---------|-------------|--------|
| torch | 2.0.0 | 2.9.x | 2.1.0 | Quantization API stability |
| transformers | 4.30.0 | 4.60.0 | 4.35.0 | Model loading/saving |
| numpy | 1.20.0 | 1.26.x | 1.24.3 | NumPy 2.x breaks things |
| datasets | 2.14.0 | latest | 2.14.5 | Dataset loading |
| scikit-learn | 1.3.0 | latest | 1.3.2 | Metrics calculation |

---

## Installation Approaches

### Approach 1: Stable Versions (Recommended for Production)

```bash
pip install -r requirements-stable.txt
```

**Pros:**
- ✅ All versions tested together
- ✅ No compatibility issues
- ✅ Reproducible builds

**Cons:**
- ⚠️ Slightly older packages

### Approach 2: Latest Compatible (Development)

```bash
pip install -r requirements.txt
```

**Pros:**
- ✅ Latest features
- ✅ Security updates

**Cons:**
- ⚠️ Possible version conflicts
- ⚠️ May need adjustments

### Approach 3: Virtual Environment (Best Practice)

```bash
# Create clean environment
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# Install stable versions
pip install -r requirements-stable.txt

# Test it works
python scripts/diagnose.py
```

---

## Diagnostic Tools

### 1. Full System Diagnostic

```bash
python scripts/diagnose.py
```

**Checks:**
- Python version
- All package versions
- CUDA/GPU availability
- Converter import
- Simple quantization test

### 2. Version Compatibility Check

```bash
python scripts/check_compatibility.py
```

**Checks:**
- Version ranges
- Known incompatibilities
- Recommendations

**Auto-fix:**
```bash
python scripts/check_compatibility.py --fix
```

### 3. Quick Conversion Test

```bash
python scripts/convert_pytorch.py --model distilbert-base-uncased-mnli --dry-run
```

---

## Platform-Specific Issues

### Linux

**Issue:** Permission errors with pip cache

**Fix:**
```bash
pip install --user -r requirements-stable.txt
# Or use virtual environment
```

### macOS

**Issue:** MLX not needed but listed

**Fix:** MLX lines are commented out in requirements.txt - the converter uses PyTorch instead

### Windows

**Issue:** Long path errors

**Fix:**
```bash
# Enable long paths in Windows
# Or use shorter output directory
python scripts/convert_pytorch.py --output-dir models/q8
```

---

## Getting Help

If you encounter issues not covered here:

1. **Run diagnostics:**
   ```bash
   python scripts/diagnose.py > diagnostic_output.txt
   ```

2. **Check compatibility:**
   ```bash
   python scripts/check_compatibility.py > version_check.txt
   ```

3. **Try stable versions:**
   ```bash
   pip install -r requirements-stable.txt
   ```

4. **Share output:**
   - diagnostic_output.txt
   - version_check.txt
   - The exact error message
   - Your OS and Python version

---

## Success Verification

After fixing issues, verify everything works:

```bash
# 1. Run diagnostics (should all pass)
python scripts/diagnose.py

# 2. Check compatibility (no critical errors)
python scripts/check_compatibility.py

# 3. Test conversion
python scripts/convert_pytorch.py --model distilbert-base-uncased-mnli

# 4. Verify output
ls -lh models/mlx_converted/distilbert-base-uncased-mnli-mlx-q8/
```

**Expected output:**
- ✅ All diagnostics pass
- ✅ No critical version errors
- ✅ Conversion succeeds in 1-3 seconds
- ✅ Output directory contains model files (config.json, pytorch_model.bin, tokenizer files)

---

## Prevention

To avoid future issues:

1. **Use virtual environments**
   ```bash
   python -m venv venv
   source venv/bin/activate
   ```

2. **Pin versions** with requirements-stable.txt

3. **Check before updating:**
   ```bash
   python scripts/check_compatibility.py
   pip install --upgrade PACKAGE
   python scripts/check_compatibility.py
   ```

4. **Test after changes:**
   ```bash
   python scripts/convert_pytorch.py --dry-run --model test-model
   ```
