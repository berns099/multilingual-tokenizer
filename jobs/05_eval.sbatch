#!/bin/bash
#SBATCH --account=yzhao010_1531
#SBATCH --partition=gpu
#SBATCH --constraint=a40|a100|l40s
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gpus-per-task=1
#SBATCH --mem=48G
#SBATCH --time=0:45:00
#SBATCH --job-name=hye-05-eval
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

banner "bits per byte and generation"
require_cuda
ARMS="stock=Qwen/Qwen3-0.6B stage1=$CK/${TGT}_stage1 \
      stage2=$CK/${TGT}_stage2 base=$CK/${TGT}_baseline_stage2"

$PY -m tokpremium.train.evaluate --lang "$TGT" \
    --out "results/vocab/${TGT}/eval_stage2.json" $ARMS

banner "belebele, all 900 records"
$PY -m tokpremium.train.evaluate_belebele --lang "$TGT" \
    --out "results/belebele_${TGT}_stage2.json" $ARMS

banner "inference benchmark"
$PY -m tokpremium.train.benchmark_inference --lang "$TGT" \
    --out "results/inference_${TGT}_stage2.json" \
    stock=Qwen/Qwen3-0.6B stage2="$CK/${TGT}_stage2"

banner "done"
echo
echo "results/ is on /home1 and backed up; checkpoints are on /scratch1 and are not."
