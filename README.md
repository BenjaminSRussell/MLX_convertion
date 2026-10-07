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

## Post-convert stages: optimize → quantize → metadata

| stage | script | what it does |
|-------|--------|--------------|
| optimize | `scripts/optimize.py IN.npz OUT.npz [--fp16]` | drops training-only tensors (optimizer/EMA), stores duplicate tensors once (`__tied__` alias map), optional fp16 cast |
| quantize | `scripts/quantize.py IN.npz OUT.npz --bits 8\|4 [--min-cosine 0.99]` | group-wise affine quantization (same scheme as `mlx.core.quantize`), 4-bit packed two per byte; norms/biases stay float; exits 1 if the worst tensor's cosine is under the gate |
| metadata | `utils/artifacts.py` | writes `artifacts/{model}/{bits}bit/metadata.json` |

`pipeline.sh` runs these when `WEIGHTS_NPZ` is set:

```bash
WEIGHTS_NPZ=output/minilm/weights.npz MODEL_ID=sentence-transformers/all-MiniLM-L6-v2 BITS=8 ./pipeline.sh
```

With MLX installed, `quantize_model()` delegates to `mlx.nn.quantize` for in-memory MLX modules.

## Artifact layout (`metadata.json`)

```
artifacts/{org}__{name}/{bits}bit/
  metadata.json   # schema_version, model_id, bits, strategy, metrics, git_sha, run_id, files[{path,bytes,sha256}]
  weights.npz
  ...
```

Upload consumers should read `metadata.json` and check `files[].sha256`.
`python scripts/upload.py --artifact-dir artifacts/<model>/8bit --dry-run` lists what would be shipped.

## Publishing to Hugging Face

```bash
python scripts/publish_hf.py artifacts/<model>/8bit            # dry-run: repo id, file list, rendered model card
HF_TOKEN=... python scripts/publish_hf.py artifacts/<model>/8bit --execute
```

- The token is read **only** from the `HF_TOKEN` environment variable. Never pass it as an argument or commit it.
- The model card is rendered from `templates/model_card.md` using `metadata.json`.
- The default repo is `{HF_NAMESPACE or BenjaminSRussell}/{name}-mlx-{bits}bit`. Override with `--repo-id`.

## Run registry (SQLite)

Conversion runs are stored in `./mlx_convertion.db`. Override the location with `MLX_CONVERTION_DB`.

- `QualityGateEnforcer.record_to_registry(result)` records every gate result, passed **or** failed.
- `python scripts/list_runs.py [-n 20] [--model M] [--status failed] [--json]`
- `python scripts/report_gates.py [-n 20] [--model M]` prints a per-gate PASS/FAIL table. It exits **1** if any listed run failed.
- Resume: `RunRegistry.run_stages(run_id, [("convert", fn), ("quantize", fn), ...])` skips stages already marked `done` and retries `failed` ones. Attempts are counted per stage.

## Reproducible eval datasets

- `config/datasets.yaml` entries may pin `revision:` (passed to `load_dataset`) and `sha256:` (an eval-split fingerprint).
- A mismatch fails the download. Unpinned datasets print their fingerprint so you can lock them.
- CI uses committed fixtures (`config/dataset_fixtures.yaml` → `tests/fixtures/datasets/`) instead of downloading.

## Tests / CI

- `PYTHONPATH=. pytest tests -q` runs on Linux without MLX or torch. `utils/__init__` imports heavy modules lazily.
- `.github/workflows/ci-linux.yml` runs ruff, bandit and pytest on every PR.
- `.github/workflows/ci-macos-mlx.yml` is an optional `workflow_dispatch` job on `macos-14` that installs `mlx` and runs the suite plus an `mlx.nn.quantize` smoke test. Trigger it from Actions → "CI macOS (MLX, optional)" → Run workflow.
