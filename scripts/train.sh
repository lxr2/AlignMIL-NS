#!/usr/bin/env bash
# Train on 5 folds for each given config (default: AlignMIL in forward and reverse task orders).
# Folds must run in order: the seed=5 run aggregates metrics of folds 1-5 into an xlsx.
# Usage: GPU=0 bash scripts/train.sh [config ...]
#   e.g. GPU=0 bash scripts/train.sh configs/qpmil_vl_fw.yaml configs/qpmil_vl_rv.yaml
set -e
GPU=${GPU:-0}
CONFIGS=("$@")
[ ${#CONFIGS[@]} -eq 0 ] && CONFIGS=(configs/ex_cl_fw.yaml configs/ex_cl_rv.yaml)
for cfg in "${CONFIGS[@]}"; do
  for seed in 1 2 3 4 5; do
    uv run python main.py -f "$cfg" seed=$seed name=main cuda_id=$GPU
  done
done
