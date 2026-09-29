#!/usr/bin/env bash
set -euo pipefail
python -m uvicorn vehicle_fingerprint.service.app:app --host 0.0.0.0 --port "${PORT:-8000}"
