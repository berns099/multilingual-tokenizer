#!/bin/bash
# Reproducible Armenian comparison on CARC Discovery, end to end.
#
#   sbatch carc_armenian.sbatch
#
# FILL IN before the first run:
#   --account below   -> run `myaccount` on Discovery; it looks like ttrojan_123
#   REPLAY            -> an English replay file (see "prerequisites" at the end)
#
# Partition, GPU request, account format, memory and time flags follow CARC's
# running-jobs guide; the module versions below are the ones in its verbatim
# example. Run `module avail python` once and adjust if they have moved.

#SBATCH --account=<project_id>
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gpus-per-task=1
#SBATCH --mem=64G
#SBATCH --time=8:00:00
#SBATCH --job-name=hye-repair
#SBATCH --output=logs/%x-%j.out

set -euo pipefail

# ---------------------------------------------------------------- environment
module purge
module load gcc/13.3.0
module load python/3.11.9

REPO=/home1/$USER/Multilingual_Tokenizer      # 100 GB quota, backed up
WORK=/scratch1/$USER/tokpremium               # 10 TB, high I/O, NOT backed up
mkdir -p "$WORK" logs

# Model and dataset caches are tens of GB and would eat the home quota.
# Scratch is purged above 80% capacity, so treat everything here as rebuildable.
export HF_HOME="$WORK/hf"
export TORCH_HOME="$WORK/torch"
export TOKENIZERS_PARALLELISM=false

cd "$REPO"
if [ ! -d "$WORK/venv" ]; then
    python3 -m venv "$WORK/venv"
    "$WORK/venv/bin/pip" install -q --upgrade pip
    "$WORK/venv/bin/pip" install -q -e ".[train,dev]"
fi
PY="$WORK/venv/bin/python"

$PY -c "import torch; print('torch', torch.__version__, '| cuda', torch.cuda.is_available(),
       '|', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'NO GPU')"

# ------------------------------------------------------------------ settings
LANG=hye_Armn
NNEW=4096
PRETOK=shipped            # Armenian measures identically under all three rules
LR=2e-5                   # same for both arms; 1e-3 is for fresh rows only
SEED=0
BUDGET="--tokens 2e6"     # matched compute. Swap for: --bytes 12950000
REPLAY_RATIO=0.25

DATA="$WORK/data"
CORPUS="$DATA/${LANG}_train.txt"
HELDOUT="$DATA/${LANG}_heldout.txt"
PROBE="$DATA/eng_probe.txt"      # measured, never trained on
REPLAY="$DATA/eng_replay.txt"    # trained on, never measured
CK="$WORK/checkpoints"

# ------------------------------------------------------- 0. build the corpora
# Deterministic under --seed: train/heldout disjoint, probe/replay disjoint,
# and every line appearing in FLORES dev or devtest removed from training.
# Re-running is a no-op once the files exist.
if [ ! -f "$CORPUS" ]; then
    $PY -m tokpremium.data.prepare --lang "$LANG" --out "$DATA" --seed "$SEED"
fi

# ------------------------------------------------- 1. merges and init (CPU-ish)
# Learns merges for every language in the config; --lang narrows it.
$PY -m tokpremium.vocab.continue_bpe --config configs/vocab.yaml --lang "$LANG"
$PY -m tokpremium.vocab.measure      --config configs/vocab.yaml --lang "$LANG"

$PY -m tokpremium.vocab.init_embeddings \
    --lang "$LANG" --pretokenizer "$PRETOK" --n "$NNEW" \
    --out "$CK/${LANG}_${NNEW}_meaninit"

# ------------------------------------------------------------- 2. stage 1
$PY -m tokpremium.train.stage1 \
    --ckpt "$CK/${LANG}_${NNEW}_meaninit" \
    --out  "$CK/${LANG}_stage1" \
    --corpus "$CORPUS" --heldout "$HELDOUT" --english "$PROBE" \
    $BUDGET --lr 1e-3 --seed "$SEED"

# ------------------------------- 3. stage 2 and its control, identical flags
# Every flag after --out matches between these two. If only one arm replays
# English, the English result measures the replay and not the tokenizer.
COMMON="--corpus $CORPUS --heldout $HELDOUT --english $PROBE \
        --replay $REPLAY --replay-ratio $REPLAY_RATIO $BUDGET --lr $LR --seed $SEED"

$PY -m tokpremium.train.stage2 \
    --ckpt "$CK/${LANG}_stage1" --out "$CK/${LANG}_stage2" --trainable all $COMMON

$PY -m tokpremium.train.baseline \
    --out "$CK/${LANG}_baseline_stage2" --trainable all $COMMON

# --------------------------------------------------------------- 4. evaluate
# Every row records its checkpoint's tokenizer hash, so these numbers stay
# linkable to the artifacts that produced them.
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

echo "done. results/ is inside the repo on /home1 and is backed up;"
echo "checkpoints are on /scratch1 and are not."

# ---------------------------------------------------------------------------
# PREREQUISITES this script does not create
#
# 1. Step 0 now builds all four corpus files and prints their SHA-256 hashes.
#    Those same hashes appear in each run's provenance block, so a mismatch
#    between two runs is visible rather than inferred.
#
# 2. The English source for probe and replay is FineWeb sample-10BT, chosen
#    here because the project config names a source per target language and
#    none for English. Change ENGLISH_SOURCE in prepare.py to use another.
#
# 3. The replay ratio of 0.25 is a starting point, not a finding: the config
#    says "telugu_plus_english" and fixes no number. If the budget allows,
#    run 0, 0.1 and 0.25 and let English retention decide.
#
# 4. Discovery's gpu partition caps a job at 48 hours. 8 is generous for a 2M
#    token budget; raise it if you switch to --bytes, since the baseline arm
#    then runs roughly 3.5x more steps to read the same text.
