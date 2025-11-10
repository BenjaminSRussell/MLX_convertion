#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR=$(cd "$(dirname "$0")" && pwd)
SCRIPTS_DIR="$ROOT_DIR/scripts"
CONFIG_DIR="$ROOT_DIR/config"
MLX_MODELS_DIR="$ROOT_DIR/models/mlx_converted"

usage() {
  cat <<USAGE
Usage: $(basename "$0") [options]

Options:
  --models "m1 m2"   Limit to specific model names (space separated)
  --datasets "d1 d2" Limit to dataset keys for testing
  --dry-run           Don't execute conversions/tests, just print
  --upload MODEL:QUANT Upload the MODEL/QUANT pair after verification
  --clear-cache       Clear the models/mlx_converted directory before running
  -h, --help          Show this message
USAGE
}

MODELS=()
DATASETS=()
DRY_RUN=""
UPLOAD_TARGET=""
CLEAR_CACHE=""

while [[ $# -gt 0 ]]; do
  case "$1" in
    --models)
      shift
      IFS=' ' read -r -a MODELS <<< "$1"
      ;;
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
    echo "[pipeline] Clearing MLX converted models cache"
    if [[ -d "$MLX_MODELS_DIR" ]]; then
      find "$MLX_MODELS_DIR" -mindepth 1 -delete
      echo "[pipeline] Cache cleared"
    else
      echo "[pipeline] Cache directory not found, skipping"
    fi
  fi
}

run_convert() {
  echo "[pipeline] Starting conversion jobs"
  if [[ ${#MODELS[@]} -eq 0 ]]; then
    echo "[pipeline] No models specified. Running for all models in config."
    local convert_args=()
    if [[ -n "$DRY_RUN" ]]; then
      convert_args+=("--dry-run")
    fi
    PYTHONPATH="$ROOT_DIR" python "$SCRIPTS_DIR/convert.py" "${convert_args[@]}"
    return
  fi

  for model in "${MODELS[@]}"; do
    echo "[pipeline] Converting model: $model"
    local convert_args=()
    if [[ -n "$DRY_RUN" ]]; then
      convert_args+=("--dry-run")
    fi
    convert_args+=("--model" "$model")
    PYTHONPATH="$ROOT_DIR" python "$SCRIPTS_DIR/convert.py" "${convert_args[@]}"
  done
}

run_tests() {
  echo "[pipeline] Running evaluations"
  ARGS=()
  if [[ -n "$DRY_RUN" ]]; then
    ARGS+=("$DRY_RUN")
  fi
  if [[ ${#MODELS[@]} -gt 0 ]]; then
    ARGS+=("--models" "${MODELS[@]}")
  fi
  if [[ ${#DATASETS[@]} -gt 0 ]]; then
    ARGS+=("--datasets" "${DATASETS[@]}")
  fi
  PYTHONPATH="$ROOT_DIR" python "$SCRIPTS_DIR/test.py" "${ARGS[@]}"
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
run_tests
run_upload
