#!/bin/bash
#SBATCH --account=yzhao010_1531
#SBATCH --partition=gpu
#SBATCH --constraint=a40|a100|l40s
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gpus-per-task=1
#SBATCH --mem=48G
#SBATCH --time=0:15:00
#SBATCH --job-name=hye-08-inference
#SBATCH --output=logs/%x-%j.out

# One evaluator per job. 05_eval ran all three together and asked for 45
# minutes, which sat pending for nine hours while the 7-minute training stages
# got through. These are independent (each only reads finished checkpoints),
# so they need no dependencies and can run in parallel.

set -euo pipefail
mkdir -p logs
if [ ! -f jobs/env.sh ]; then
    echo "jobs/env.sh not found: submit from the repo root" >&2
    exit 1
fi
source jobs/env.sh

ARMS="stock=Qwen/Qwen3-0.6B stage1=$CK/${TGT}_stage1 \
      stage2=$CK/${TGT}_stage2 base=$CK/${TGT}_baseline_stage2"

# Lowest priority of the three: token_ratio is a property of the tokenizer
# rather than of training, so the smoke run already measured it at 0.294.
# This re-measures it with warmup 5 and 20 repeats on the real checkpoint.
OUT="results/inference_${TGT}_stage2.json"
banner "inference benchmark"
if [ -f "$OUT" ]; then
    echo "$OUT already exists; nothing to do"
else
    require_cuda
    $PY -m tokpremium.train.benchmark_inference --lang "$TGT" --out "$OUT" \
        stock=Qwen/Qwen3-0.6B stage2="$CK/${TGT}_stage2"
fi
banner "done"
