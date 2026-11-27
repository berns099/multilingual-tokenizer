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
#SBATCH --job-name=hye-04-baseline
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

# The matched control: ORIGINAL tokenizer, body unfrozen, same replay mixture.
# $COMMON is byte-identical to stage 2's, which is the whole point of env.sh.
banner "baseline: original tokenizer, body unfrozen, same replay"
require_cuda
if done_already "$CK/${TGT}_baseline_stage2"; then
    echo "baseline already finished; nothing to do"
else
    $PY -m tokpremium.train.baseline \
        --out "$CK/${TGT}_baseline_stage2" --trainable all $COMMON
fi
banner "done"
