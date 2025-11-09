#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR=$(cd "$(dirname "$0")" && pwd)
SCRIPTS_DIR="$ROOT_DIR/scripts"
CONFIG_DIR="$ROOT_DIR/config"

usage() {
  cat <<USAGE
Usage: $(basename "$0") [options]

Options:
  --models "m1 m2"   Limit to specific model names (space separated)
  --datasets "d1 d2" Limit to dataset keys for testing
  --dry-run           Don't execute conversions/tests, just print
  --upload MODEL:QUANT Upload the MODEL/QUANT pair after verification
  -h, --help          Show this message
USAGE
}

MODELS=""
DATASETS=""
DRY_RUN=""
UPLOAD_TARGET=""

while [[ $# -gt 0 ]]; do
  case "$1" in
    --models)
      shift
      MODELS=($1)
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

run_convert() {
  echo "[pipeline] Starting conversion jobs"
  ARGS=()
  if [[ -n "$DRY_RUN" ]]; then
    ARGS+=("$DRY_RUN")
  fi
  if [[ ${#MODELS[@]} -gt 0 ]]; then
    ARGS+=("--models" "$@")
  fi
  PYTHONPATH="$ROOT_DIR" python "$SCRIPTS_DIR/convert.py" "${ARGS[@]}"
}

run_tests() {
  echo "[pipeline] Running evaluations"
  ARGS=()
  if [[ -n "$DRY_RUN" ]]; then
    ARGS+=("$DRY_RUN")
  fi
  if [[ ${#MODELS[@]} -gt 0 ]]; then
    ARGS+=("--models" "$@")
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

run_convert "${MODELS[@]}"
run_tests "${MODELS[@]}"
run_upload
