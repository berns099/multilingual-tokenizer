#!/bin/bash
#SBATCH --account=yzhao010_1531
#SBATCH --partition=gpu
#SBATCH --constraint=a40|a100|l40s
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gpus-per-task=1
#SBATCH --mem=48G
#SBATCH --time=0:40:00
#SBATCH --job-name=hye-03-stage2
#SBATCH --output=logs/%x-%j.out

set -euo pipefail
mkdir -p logs

# Slurm starts a batch job in the directory it was submitted from, which is
# also what --output is relative to. So submit from the repo root.
if [ ! -f jobs/env.sh ]; then
    echo "jobs/env.sh not found: submit from the repo root, not from jobs/" >&2
    exit 1
fi
source jobs/env.sh

banner "stage 2: unfreeze the body, with English replay"
require_cuda
if done_already "$CK/${TGT}_stage2"; then
    echo "stage 2 already finished; nothing to do"
else
    $PY -m tokpremium.train.stage2 \
        --ckpt "$CK/${TGT}_stage1" --out "$CK/${TGT}_stage2" \
        --trainable all $COMMON
fi
banner "done"
