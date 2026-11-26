#!/bin/bash
# Stage 2 and its matched control, for a run whose stage 1 has already finished.
#
#   sbatch carc_armenian_stage2.sbatch
#
# carc_armenian.sbatch runs the whole pipeline from the corpora up. Use this
# script instead when stage 1 succeeded and only the later stages need to run,
# which is the situation after a stage-2 failure: re-running the full script
# would retrain stage 1 and overwrite a checkpoint that was fine.
#
# The settings block below is copied from carc_armenian.sbatch and must stay
# in step with it. Every flag in COMMON is passed identically to both arms.

#SBATCH --account=yzhao010_1531
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gpus-per-task=1
#SBATCH --mem=64G
#SBATCH --time=8:00:00
#SBATCH --job-name=hye-stage2
#SBATCH --output=logs/%x-%j.out

set -euo pipefail

module purge
module load gcc/13.3.0
module load python/3.11.9

REPO=/home1/$USER/Multilingual_Tokenizer
WORK=/scratch1/$USER/tokpremium
mkdir -p logs

export HF_HOME="$WORK/hf"
export TORCH_HOME="$WORK/torch"
export TOKENIZERS_PARALLELISM=false

cd "$REPO"
PY="$WORK/venv/bin/python"

$PY -c "import torch; print('torch', torch.__version__, '| cuda', torch.cuda.is_available(),
       '|', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'NO GPU')"

# ------------------------------------------------------------------ settings
LANG=hye_Armn
LR=2e-5
SEED=0
BUDGET="--tokens 2e6"
REPLAY_RATIO=0.25

DATA="$WORK/data"
CORPUS="$DATA/${LANG}_train.txt"
HELDOUT="$DATA/${LANG}_heldout.txt"
PROBE="$DATA/eng_probe.txt"      # measured, never trained on
REPLAY="$DATA/eng_replay.txt"    # trained on, never measured
CK="$WORK/checkpoints"

# Fail now, with a readable message, rather than inside Transformers.
if [ ! -d "$CK/${LANG}_stage1" ]; then
    echo "no stage-1 checkpoint at $CK/${LANG}_stage1" >&2
    echo "run carc_armenian.sbatch first, or fix the path above." >&2
    exit 1
fi

# ------------------------------- stage 2 and its control, identical flags
COMMON="--corpus $CORPUS --heldout $HELDOUT --english $PROBE \
        --replay $REPLAY --replay-ratio $REPLAY_RATIO $BUDGET --lr $LR --seed $SEED"

$PY -m tokpremium.train.stage2 \
    --ckpt "$CK/${LANG}_stage1" --out "$CK/${LANG}_stage2" --trainable all $COMMON

$PY -m tokpremium.train.baseline \
    --out "$CK/${LANG}_baseline_stage2" --trainable all $COMMON

# --------------------------------------------------------------- evaluate
$PY -m tokpremium.train.evaluate --lang "$LANG" \
    --out "results/vocab/${LANG}/eval_stage2.json" \
    stock=Qwen/Qwen3-0.6B \
    stage1="$CK/${LANG}_stage1" \
    stage2="$CK/${LANG}_stage2" \
    base="$CK/${LANG}_baseline_stage2"

$PY -m tokpremium.train.evaluate_belebele --lang "$LANG" \
    --out "results/belebele_${LANG}_stage2.json" \
    stock=Qwen/Qwen3-0.6B \
    stage1="$CK/${LANG}_stage1" \
    stage2="$CK/${LANG}_stage2" \
    base="$CK/${LANG}_baseline_stage2"

$PY -m tokpremium.train.benchmark_inference --lang "$LANG" \
    --out "results/inference_${LANG}_stage2.json" \
    stock=Qwen/Qwen3-0.6B stage2="$CK/${LANG}_stage2"

echo "done. results/ is on /home1 and is backed up; checkpoints are on"
echo "/scratch1 and are not."
