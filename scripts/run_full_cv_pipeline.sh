#!/usr/bin/env bash
set -euo pipefail
CONFIG="${CONFIG:-configs/full_cv_pipeline.yaml}"
python scripts/30_full_cv_pipeline.py --config "$CONFIG" "$@"
