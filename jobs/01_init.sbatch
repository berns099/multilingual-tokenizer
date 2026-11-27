#!/bin/bash
#SBATCH --account=yzhao010_1531
#SBATCH --partition=main
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=0:20:00
#SBATCH --job-name=hye-01-init
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

# init_embeddings never moves the model to a device: it loads in bfloat16 and
# averages rows on the CPU. So this one runs on `main`, which has idle nodes,
# instead of waiting in the gpu queue for hardware it would not touch.
banner "init embeddings"
if [ -d "$CK/${TGT}_${NNEW}_meaninit" ]; then
    echo "already built; nothing to do"
else
    $PY -m tokpremium.vocab.init_embeddings \
        --lang "$TGT" --pretokenizer "$PRETOK" --n "$NNEW" \
        --out "$CK/${TGT}_${NNEW}_meaninit"
fi
banner "done"
