#!/usr/bin/env bash
# DPR v1 on the local RTX A3000 (6 GB, so batch 16).
set -euo pipefail
cd "$(dirname "$0")/.."

${PY:-python} train_dpr.py model=hf_encoder train.epochs=3 train.batch_size=16 train.lr=2e-5 device=cuda
