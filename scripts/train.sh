#!/usr/bin/env bash
# Train AlignMIL on 5 folds in forward and reverse task orders.
# Folds must run in order: the seed=5 run aggregates metrics of folds 1-5 into an xlsx.
# Usage: GPU=0 bash scripts/train.sh
set -e
GPU=${GPU:-0}
for cfg in configs/ex_cl_fw.yaml configs/ex_cl_rv.yaml; do
  for seed in 1 2 3 4 5; do
    uv run python main.py -f "$cfg" seed=$seed name=alignmil cuda_id=$GPU
  done
done
