#!/bin/bash
#SBATCH --account=yzhao010_1531
#SBATCH --partition=debug
#SBATCH --constraint=a40
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gpus-per-task=1
#SBATCH --mem=48G
#SBATCH --time=0:40:00
#SBATCH --job-name=hye-07-belebele
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

# The one that matters: 900 paired binary outcomes, which is what McNemar and
# a Wilson interval need. On debug because its a40 turns jobs around in
# minutes, and 900 records took 8 minutes in the smoke extrapolation.
OUT="results/belebele_${TGT}_stage2.json"
banner "belebele, all 900 records, 4 arms"
if [ -f "$OUT" ]; then
    echo "$OUT already exists; nothing to do"
else
    require_cuda
    $PY -m tokpremium.train.evaluate_belebele --lang "$TGT" --out "$OUT" $ARMS
fi
banner "done"
