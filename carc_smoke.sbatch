#!/bin/bash
# Smoke test: run every stage with a tiny budget, to prove the pipeline works.
#
#   sbatch carc_smoke.sbatch
#
# WHY THIS EXISTS. The real run (carc_armenian.sbatch) sits in the gpu queue
# for many hours, because the class account is running well over its fairshare.
# Discovering a crash at the far end of that wait is expensive. This script
# exercises the same code paths at 1/100th the budget on the debug partition,
# which has a one hour limit and usually a free node.
#
# IT TOUCHES NOTHING THE REAL RUN USES. Checkpoints go to $WORK/smoke and
# results to /dev/null. Delete $WORK/smoke afterwards.
#
# WHAT A PASS MEANS. torch sees a GPU; embedding init works; stage 1, stage 2
# and the baseline all parse their flags, mix replay and take real optimizer
# steps; and both evaluators load four checkpoints and score. It does NOT tell
# you the result is meaningful: 4 steps trains nothing.

#SBATCH --account=yzhao010_1531
#SBATCH --partition=debug
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gpus-per-task=1
#SBATCH --mem=24G
#SBATCH --time=0:50:00
#SBATCH --job-name=hye-smoke
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

LANG=hye_Armn
NNEW=4096
PRETOK=shipped
SEED=0
TINY="--tokens 2e4"        # 2e4 / (16 * 256) = 4 optimizer steps per arm
DATA="$WORK/data"
SMOKE="$WORK/smoke"        # throwaway; the real run uses $WORK/checkpoints
mkdir -p "$SMOKE"

CORPUS="$DATA/${LANG}_train.txt"
HELDOUT="$DATA/${LANG}_heldout.txt"
PROBE="$DATA/eng_probe.txt"
REPLAY="$DATA/eng_replay.txt"

for f in "$CORPUS" "$HELDOUT" "$PROBE" "$REPLAY"; do
    [ -f "$f" ] || { echo "missing $f; run tokpremium.data.prepare first" >&2; exit 1; }
done

# The merges already exist in results/, so this should be a no-op. If it is
# not, that is worth knowing: it is the step whose cost we cannot predict.
echo "--- init embeddings ---"
$PY -m tokpremium.vocab.init_embeddings \
    --lang "$LANG" --pretokenizer "$PRETOK" --n "$NNEW" \
    --out "$SMOKE/meaninit"

echo "--- stage 1 (4 steps) ---"
$PY -m tokpremium.train.stage1 \
    --ckpt "$SMOKE/meaninit" --out "$SMOKE/stage1" \
    --corpus "$CORPUS" --heldout "$HELDOUT" --english "$PROBE" \
    $TINY --lr 1e-3 --seed "$SEED" --eval-docs 5 --save-every 0

# The two arms below are the ones that were broken: stage2.py read args.replay
# while common.py had no such flag. A pass here is the thing worth knowing.
COMMON="--corpus $CORPUS --heldout $HELDOUT --english $PROBE \
        --replay $REPLAY --replay-ratio 0.25 $TINY --lr 2e-5 --seed $SEED \
        --eval-docs 5 --save-every 0"

echo "--- stage 2 (4 steps, with replay) ---"
$PY -m tokpremium.train.stage2 \
    --ckpt "$SMOKE/stage1" --out "$SMOKE/stage2" --trainable all $COMMON

echo "--- baseline (4 steps, with replay) ---"
$PY -m tokpremium.train.baseline \
    --out "$SMOKE/baseline" --trainable all $COMMON

echo "--- evaluate ---"
$PY -m tokpremium.train.evaluate --lang "$LANG" \
    stock=Qwen/Qwen3-0.6B stage1="$SMOKE/stage1" \
    stage2="$SMOKE/stage2" base="$SMOKE/baseline"

echo "--- belebele (20 records) ---"
$PY -m tokpremium.train.evaluate_belebele --lang "$LANG" --limit 20 \
    stock=Qwen/Qwen3-0.6B stage1="$SMOKE/stage1" \
    stage2="$SMOKE/stage2" base="$SMOKE/baseline"

echo "--- inference benchmark ---"
$PY -m tokpremium.train.benchmark_inference --lang "$LANG" \
    --warmup 1 --repeats 3 \
    stock=Qwen/Qwen3-0.6B stage2="$SMOKE/stage2"

echo
echo "SMOKE TEST PASSED. Every stage ran. Remove $SMOKE when done."
echo "This proves the code path, not the science: 4 steps trains nothing."
