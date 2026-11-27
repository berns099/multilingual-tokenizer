# Shared settings for every stage of the Armenian run. Sourced, never run.
#
# One file so the arms cannot drift. Stage 2 and the baseline must differ in
# exactly one thing, the tokenizer; if any other flag differs, the comparison
# measures that flag instead. Keeping COMMON here is what enforces it.
#
# Deliberately NOT named LANG: that is the shell's locale variable, and
# assigning to it exports a bogus locale to every child process.

module purge
module load gcc/13.3.0
module load python/3.11.9

REPO=/home1/$USER/Multilingual_Tokenizer
WORK=/scratch1/$USER/tokpremium

export HF_HOME="$WORK/hf"
export TORCH_HOME="$WORK/torch"
export TOKENIZERS_PARALLELISM=true

TGT=hye_Armn
NNEW=4096
PRETOK=shipped          # Armenian measures 1.4615 under all three rules
SEED=0
BUDGET="--tokens 2e6"   # matched compute: 488 steps at batch 16, seq 256
REPLAY_RATIO=0.25
LR_STAGE1=1e-3          # fresh embedding rows only
LR_STAGE2=2e-5          # pretrained body; 1e-3 would damage it

DATA="$WORK/data"
CK="$WORK/checkpoints"
CORPUS="$DATA/${TGT}_train.txt"
HELDOUT="$DATA/${TGT}_heldout.txt"
PROBE="$DATA/eng_probe.txt"      # measured, never trained on
REPLAY="$DATA/eng_replay.txt"    # trained on, never measured
PY="$WORK/venv/bin/python"

# Identical between stage 2 and the baseline. Change it in one place only.
COMMON="--corpus $CORPUS --heldout $HELDOUT --english $PROBE \
        --replay $REPLAY --replay-ratio $REPLAY_RATIO $BUDGET \
        --lr $LR_STAGE2 --seed $SEED"

cd "$REPO"

# Refuse the allocation rather than waste it: device() once returned
# mps-or-cpu, so a run held an idle GPU and trained on four CPU cores.
require_cuda() {
    $PY - <<'GUARD'
import sys
from tokpremium.train import common
dev = common.device()
print(f"common.device() -> {dev}", flush=True)
if dev.type != "cuda":
    sys.exit(f"refusing to run: GPU allocated but common.device() returns {dev}")
GUARD
}

# A finished stage leaves report.json and no ckpt.pt, so re-running would
# retrain from scratch. Resubmitting a chain should be a no-op for the
# stages that already succeeded.
done_already() {
    [ -f "$1/report.json" ]
}

banner() { echo; echo "=== $* === $(date +%H:%M:%S)"; }
