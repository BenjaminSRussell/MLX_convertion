# MLX_convertion

Convert, quantize, and verify Hugging Face models for Apple Silicon via [MLX](https://github.com/ml-explore/mlx).

## Prerequisites

- **Apple Silicon Mac** for MLX conversion/quantize (Linux CI runs non-MLX unit stubs only).
- Python 3.10+ recommended.
- Hugging Face access for gated models (`huggingface-cli login`).

## Requirements files

| File | Use |
|------|-----|
| `requirements.txt` | Floating ranges (`mlx`, `torch`, `transformers`, …) for day-to-day work |
| `requirements-stable.txt` | Pinned tested versions when you hit dependency conflicts |

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt          # or requirements-stable.txt
```

## Pipeline stages

`./pipeline.sh` orchestrates:

1. **Download datasets** — `scripts/download_datasets.py` + `config/datasets.yaml`
2. **Convert** — HF → MLX (`scripts/convert.py` / `convert_encoder.py` / `convert_llm.py`)
3. **Quantize** — `scripts/quantize.py`
4. **Verify / evaluate** — accuracy gates (`scripts/verify_quantization_accuracy.py`, `evaluate.py`)
5. **Optional upload** — `scripts/upload.py` (HF publish)

Config: [`config/models.yaml`](config/models.yaml), [`config/datasets.yaml`](config/datasets.yaml).

### Copy-paste: encoder (NLI / MiniLM)

```bash
# Dry-run plan
./pipeline.sh --dry-run --datasets "mnli"

# Convert + quantize a small encoder from models.yaml (example key)
python scripts/convert_encoder.py --model all-MiniLM-L6-v2
python scripts/quantize.py --model all-MiniLM-L6-v2
python scripts/verify_quantization_accuracy.py --model all-MiniLM-L6-v2
```

### Copy-paste: LLM path

```bash
./pipeline.sh --dry-run
python scripts/convert_llm.py --help
# Then quantize + verify using the model key from config/models.yaml
```

### Full pipeline

```bash
./pipeline.sh --clear-cache          # wipe output/
./pipeline.sh --datasets "mnli sts"  # limit datasets while testing
./pipeline.sh --upload MODEL:QUANT   # after verification
```

## Linux / CI note

MLX kernels require Apple Silicon. On Linux, install non-MLX deps and run unit tests that do not import `mlx` (see pytest / CI workflow if present). Do not expect conversion jobs to succeed on x86_64 Linux runners.

## Layout

- `pipeline.sh` — CLI entry
- `scripts/` — convert / quantize / verify / upload
- `config/` — models + datasets YAML
- `output/` — artifacts (local; not required in git)
