#!/usr/bin/env bash
# Pooling ablation over the HF tower, one checkpoint dir per pooling type.
set -euo pipefail
cd "$(dirname "$0")/.."

PY=/home/hanan/venvs/retrieval/bin/python

for p in cls mean max; do
  echo "=== pooling=${p} ==="
  ${PY} train_dpr.py model.pooling_type="${p}" train.batch_size=16 device=cuda 2>&1 | tee "checkpoints/ablation_${p}.log"
  ${PY} - <<EOF
import json
pooling = "${p}"
with open(f"checkpoints/hf_{pooling}/meta.json") as f:
    meta = json.load(f)
val = meta.get("val") or {}
print(f"pooling={pooling} final_loss={meta.get('final_train_loss', float('nan')):.4f}")
for metric in ("recall", "mrr"):
    # JSON object keys are strings; older checkpoints store recall directly in val.
    scores = val.get(metric, val if metric == "recall" else {})
    values = "/".join(f"{scores.get(str(k), float('nan')):.4f}" for k in (1, 5, 10, 20))
    print(f"{metric}@1/5/10/20={values}")
EOF
done
