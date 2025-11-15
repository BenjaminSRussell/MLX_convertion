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
run_tests
run_upload
