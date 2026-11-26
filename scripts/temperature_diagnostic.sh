#!/usr/bin/env bash
# Short temperature check before the long ModernBERT run: the first 1024 training
# questions, one epoch (8 steps of 128), then validation on the full corpus.
#   OUT=<dir> BASE=<repo with retrieval_data> NEG=<hard_negatives.json> scripts/temperature_diagnostic.sh
set -euo pipefail
cd "$(dirname "$0")/.."
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
for tau in ${TAUS:-1.0 0.05 0.02}; do
  ${PY:-python} train_dpr.py --config-name train_dpr_modernbert \
    base_path="$BASE" train.hard_negatives_file="$NEG" output_dir="$OUT/tau-$tau" \
    limit_pairs=1024 train.epochs=1 train.temperature="$tau" 2>&1 | tee "$OUT/tau-$tau.log"
  rm -f "$OUT/tau-$tau"/hf_*/query_encoder.pt
done
