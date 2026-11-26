#!/usr/bin/env bash
# The second FiQA run (config/train_dpr_modernbert.yaml) on the local RTX A3000.
#   OUT=<dir> BASE=<repo with retrieval_data> scripts/train_modernbert.sh
set -euo pipefail
cd "$(dirname "$0")/.."
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
PY=${PY:-python}
mkdir -p "$OUT"
[ -f "$OUT/hard_negatives.json" ] || $PY scripts/mine_hard_negatives.py --base-path "$BASE" --out "$OUT/hard_negatives.json"
$PY train_dpr.py --config-name train_dpr_modernbert base_path="$BASE" \
  train.hard_negatives_file="$OUT/hard_negatives.json" output_dir="$OUT/full" "$@" 2>&1 | tee "$OUT/full/train.log"
