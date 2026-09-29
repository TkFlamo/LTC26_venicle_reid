#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
export PYTHONPATH="$ROOT/src${PYTHONPATH:+:$PYTHONPATH}"

DEVICE="${GPU:-0}"
OUT="${TRT_OUT:-deploy/tensorrt_v094}"
PRECISION="${TRT_PRECISION:-fp16}"
CN_WEIGHT="${CONVNEXT_WEIGHT:-0.45}"

python scripts/40_export_tensorrt_v094.py \
  --out "$OUT" \
  --precision "$PRECISION" \
  --device "$DEVICE" \
  "$@"

python scripts/41_benchmark_tensorrt_v094.py \
  --export-dir "$OUT" \
  --device "$DEVICE" \
  --convnext-weight "$CN_WEIGHT"
