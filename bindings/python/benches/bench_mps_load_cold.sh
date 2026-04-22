#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."

PYTHON="${PYTHON:-/Users/salia/Desktop/pytorch/.venv/bin/python}"
SIZE_GB="${SIZE_GB:-16}"
ITERS="${ITERS:-3}"
CACHE_FILE="${CACHE_FILE:-/tmp/mps_bench_llm_${SIZE_GB}gb.safetensors}"

sudo -v
( while true; do sudo -n true; sleep 30; done ) &
trap "kill $! 2>/dev/null || true" EXIT INT TERM

exec "$PYTHON" benches/bench_mps_load.py \
    --file "$CACHE_FILE" --gb "$SIZE_GB" --iters "$ITERS" --cold "$@"
