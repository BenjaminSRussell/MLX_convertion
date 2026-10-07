#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR=$(cd "$(dirname "$0")" && pwd)
SCRIPTS_DIR="$ROOT_DIR/scripts"
CONFIG_DIR="$ROOT_DIR/config"
OUTPUT_DIR="$ROOT_DIR/output"

usage() {
  cat <<USAGE
Usage: $(basename "$0") [options]

Options:
  --datasets "d1 d2" Limit to dataset keys for testing
  --dry-run           Don't execute conversions/tests, just print
  --upload MODEL:QUANT Upload the MODEL/QUANT pair after verification
  --clear-cache       Clear the output directory before running
  -h, --help          Show this message
USAGE
}

DATASETS=()
DRY_RUN=""
UPLOAD_TARGET=""
CLEAR_CACHE=""

while [[ $# -gt 0 ]]; do
  case "$1" in
    --datasets)
      shift
      DATASETS=($1)
      ;;
    --dry-run)
      DRY_RUN="--dry-run"
      ;;
    --upload)
      shift
      UPLOAD_TARGET="$1"
      ;;
    --clear-cache)
      CLEAR_CACHE="true"
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "Unknown argument: $1" >&2
      usage
      exit 1
      ;;
  esac
  shift
done

clear_cache() {
  if [[ -n "$CLEAR_CACHE" ]]; then
    echo "[pipeline] Clearing output directory"
    if [[ -d "$OUTPUT_DIR" ]]; then
      find "$OUTPUT_DIR" -mindepth 1 -delete
      echo "[pipeline] Cache cleared"
    else
      echo "[pipeline] Cache directory not found, skipping"
    fi
  fi
}

run_convert() {
  echo "[pipeline] Starting conversion jobs"
  local convert_args=()
  if [[ -n "$DRY_RUN" ]]; then
    convert_args+=("--dry-run")
  fi
  PYTHONPATH="$ROOT_DIR" python "$SCRIPTS_DIR/convert.py" "${convert_args[@]}"
}

# Post-convert stages (#5). Operate on an exported weights archive:
#   WEIGHTS_NPZ=output/<model>/weights.npz BITS=8 MODEL_ID=org/name ./pipeline.sh
# optimize: drop training-only tensors, tie duplicates (lossless)
# quantize: group-wise affine int8/int4; fails if worst-tensor cosine < MIN_COSINE
# metadata: write artifacts/{model}/{bits}bit/metadata.json for upload/publish (#11)
run_optimize_quantize() {
  if [[ -z "${WEIGHTS_NPZ:-}" ]]; then
    echo "[pipeline] WEIGHTS_NPZ not set; skipping optimize/quantize stages"
    return
  fi
  local bits="${BITS:-8}"
  local model_id="${MODEL_ID:-$(basename "$(dirname "$WEIGHTS_NPZ")")}"
  local slug="${model_id//\//__}"
  local art_dir="$ROOT_DIR/artifacts/$slug/${bits}bit"
  mkdir -p "$art_dir"
  if [[ -n "$DRY_RUN" ]]; then
    echo "[pipeline] (dry-run) optimize $WEIGHTS_NPZ -> quantize ${bits}-bit -> $art_dir"
    return
  fi
  echo "[pipeline] Optimizing $WEIGHTS_NPZ"
  PYTHONPATH="$ROOT_DIR" python "$SCRIPTS_DIR/optimize.py" "$WEIGHTS_NPZ" "$art_dir/optimized.npz" \
    --report "$art_dir/optimize_report.json"
  echo "[pipeline] Quantizing to ${bits}-bit"
  PYTHONPATH="$ROOT_DIR" python "$SCRIPTS_DIR/quantize.py" "$art_dir/optimized.npz" "$art_dir/weights.npz" \
    --bits "$bits" --min-cosine "${MIN_COSINE:-0.99}"
  rm -f "$art_dir/optimized.npz"
  PYTHONPATH="$ROOT_DIR" python - "$model_id" "$bits" "$art_dir" <<'PYEOF'
import sys
from pathlib import Path
from utils.artifacts import build_metadata, write_metadata
from utils.run_registry import current_git_sha
model_id, bits, d = sys.argv[1], int(sys.argv[2]), Path(sys.argv[3])
write_metadata(d, build_metadata(model_id, bits, d, strategy="npz-affine", git_sha=current_git_sha()))
print(f"[pipeline] wrote {d / 'metadata.json'}")
PYEOF
}

run_tests() {
  echo "[pipeline] Running evaluations"
  ARGS=()
  if [[ -n "$DRY_RUN" ]]; then
    ARGS+=("$DRY_RUN")
  fi
  if [[ ${#DATASETS[@]} -gt 0 ]]; then
    ARGS+=("--datasets" "${DATASETS[@]}")
  fi
  pytest tests/
}

run_upload() {
  if [[ -z "$UPLOAD_TARGET" ]]; then
    return
  fi
  IFS=":" read -r model quant <<< "$UPLOAD_TARGET"
  echo "[pipeline] Uploading $model:$quant"
  ARGS=()
  if [[ -n "$DRY_RUN" ]]; then
    ARGS+=("$DRY_RUN")
  fi
  python "$SCRIPTS_DIR/upload.py" "$model" "$quant" "${ARGS[@]}"
}

clear_cache
run_convert
run_optimize_quantize
run_tests
run_upload
