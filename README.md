# MLX Model Conversion & Testing Pipeline

A comprehensive pipeline for converting HuggingFace models to 8-bit quantized MLX format, with performance testing and comparison against PyTorch baselines.

## Features

- **Automatic Model Conversion**: Convert any HuggingFace model to 8-bit MLX format
- **6 Performance Metrics**: Accuracy, Speed (QPM), Model Size, Token Throughput, Memory Usage, Latency Percentiles
- **Multi-YAML Support**: Load models and datasets from multiple configuration files
- **Duplicate Detection**: Skip already-converted models automatically
- **Temp Folder Management**: All logs and results stored in system temp directory
- **Parallel Processing**: Convert multiple models concurrently
- **Quality Gates**: Automated validation of accuracy, speed, and size requirements

## Quick Start

### 1. Installation

```bash
# Install dependencies
pip install -r requirements.txt
```

### 2. Convert Models

```bash
# Convert all models in config/models.yaml
./pipeline.sh

# Convert specific model
python scripts/convert.py --model distilbert-base-uncased-mnli

# Convert models from multiple YAML files
python scripts/convert.py --config "config/*.yaml"

# Dry-run (show commands without executing)
python scripts/convert.py --dry-run
```

### 3. Test Models

```bash
# Test all converted models
python scripts/test.py

# Test specific model on specific dataset
python scripts/test.py --model distilbert-base-uncased-mnli --dataset mnli
```

## Configuration

### Model Configuration (`config/models.yaml`)

Define models to convert with quantization parameters:

```yaml
models:
  - name: "model-identifier"
    hf_name: "huggingface/repo-name"
    task: "classification-type"
    quantization:
      bits: 8
      dtype: "int8"
      group_size: 64  # Optional, defaults to 64
      target_size_mb: 125
      max_accuracy_drop: 0.015  # 1.5% threshold
    benchmarks:
      - dataset1
      - dataset2
```

### Dataset Configuration (`config/datasets.yaml`)

Define benchmark datasets for testing:

```yaml
datasets:
  dataset_name:
    provider: huggingface
    name: dataset_path
    subset: optional_subset  # Optional
    splits:
      train: train
      validation: validation
    metrics:
      - accuracy
    preprocessing:
      max_length: 512
      truncation: longest_first
      padding: longest
      cache_dir: /tmp/hf_datasets  # Optional
```

## Performance Metrics

The pipeline tracks 6 comprehensive metrics:

1. **Accuracy**: Model prediction accuracy vs ground truth
2. **Speed (QPM)**: Queries per minute throughput
3. **Model Size (MB)**: Disk space used by converted model
4. **Token Throughput**: Tokens processed per second
5. **Memory Usage**: Average and peak memory consumption (MB)
6. **Latency Percentiles**: p50, p95, p99 response times (ms)

## Directory Structure

```
MLX_convertion/
├── config/
│   ├── models.yaml          # Model definitions
│   └── datasets.yaml        # Dataset configurations
├── scripts/
│   ├── convert.py          # Model conversion script
│   ├── test.py             # Testing & benchmarking
│   ├── monitor.py          # Log monitoring utility
│   └── upload.py           # Model packaging & upload
├── models/
│   └── mlx_converted/      # Converted MLX models
├── pipeline.sh             # Main orchestration script
└── requirements.txt        # Python dependencies
```

## Output Locations

All logs and results are stored in the system temp directory (`/tmp/mlx_conversion/` on Linux/Mac):

```
/tmp/mlx_conversion/
├── logs/
│   ├── conversion_*.log
│   └── testing_8bit.log
├── results/
│   └── *_conversion.json
├── test_results/
│   └── *_8bit_results.json
└── comparisons/
    └── *_comparison.json
```

**Environment Variable**: Set `MLX_TEMP_DIR` to use a custom temp location:
```bash
export MLX_TEMP_DIR=/path/to/custom/temp
```

## Advanced Usage

### Adding New Models

Create a new YAML file in `config/` or add to existing:

```yaml
models:
  - name: "my-custom-model"
    hf_name: "username/model-repo"
    task: "classification"
    quantization:
      bits: 8
      dtype: "int8"
      target_size_mb: 200
      max_accuracy_drop: 0.02
    benchmarks:
      - mnli
      - sentiment
```

Then convert:
```bash
python scripts/convert.py --config "config/*.yaml"
```

### Skip Duplicate Detection

Force re-conversion of already-converted models:

```bash
python scripts/convert.py --no-skip --model model-name
```

### Monitor Conversions

Watch conversion progress in real-time:

```bash
python scripts/monitor.py --interval 30
```

### Pipeline Options

```bash
./pipeline.sh [options]

Options:
  --models "m1 m2"       Limit to specific models (space-separated)
  --datasets "d1 d2"     Limit to specific datasets
  --dry-run              Print commands without executing
  --upload MODEL:QUANT   Upload specific model/quant pair
  -h, --help             Show help message
```

## Pre-configured Models

The pipeline includes 14 pre-configured models:

**NLI/Zero-shot Models:**
- distilbert-base-uncased-mnli (125 MB)
- bart-large-mnli (480 MB)
- roberta-large-mnli (450 MB)
- modernbert-large-zeroshot-v2 (500 MB)
- deberta-v3-base-mnli (200 MB)

**Semantic Similarity:**
- all-MiniLM-L6-v2 (40 MB)
- all-mpnet-base-v2 (130 MB)
- all-MiniLM-L12-v2 (50 MB)

**Efficient Models:**
- mobilebert-uncased-mnli (40 MB)
- deberta-v3-xsmall-mnli-binary (100 MB)
- xtremedistil-l6-h256-nli-binary (40 MB)

## Quality Gates

Converted models must pass all quality gates:

- **Accuracy Gate**: Drop ≤ configured threshold (typically 1-2%)
- **Speed Gate**: ≥1.2x faster than PyTorch baseline
- **Size Gate**: Within ±10% of target size

## Troubleshooting

### Model Not Found Error

```
MLX model not found at models/mlx_converted/model-name-mlx-q8
Make sure to run conversion first: python scripts/convert.py --model model-name
```

**Solution**: Convert the model before testing.

### Dataset Loading Issues

If datasets fail to load, check:
1. Dataset name is correct in `datasets.yaml`
2. Cache directory is writable: `/tmp/hf_datasets`
3. Internet connection for first download

### Memory Issues

For large models, consider:
1. Reducing `max_samples` in test script
2. Using smaller batch sizes
3. Testing one model at a time

## Contributing

To add new models or datasets:

1. Add configuration to appropriate YAML file
2. Run conversion with `--dry-run` first to validate
3. Test thoroughly with quality gates
4. Document any special requirements

## License

[Your License Here]
