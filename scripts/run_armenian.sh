#!/bin/sh
# Armenian stage 1 (+4096 tokens, new rows only), then the matched baseline.
# Each run retries up to 5 times, resuming from <out>/ckpt.pt (saved every 50 steps).
# Safe to re-run after a crash/restart: finished runs (report.json) are skipped.
cd "$(dirname "$0")/.."
. .venv/bin/activate
export HF_HOME=/Volumes/T7/tokpremium/hf_cache
run() {  # name outdir data-prefix ; remaining args = module + args
  name=$1; out=$2; pre=$3; shift 3
  [ -f $out/report.json ] && return 0
  D="--corpus data/${pre}_train.txt --heldout data/${pre}_heldout.txt --english data/eng_sample.txt --tokens 2e6"
  for try in 1 2 3 4 5; do
    python -m "$@" --out $out $D >> checkpoints/$name.log 2>&1
    rc=$?; echo "$name try $try exit $rc $(date +%H:%M)" >> checkpoints/runs2.log
    [ -f $out/report.json ] && return 0
    sleep 5
  done
}
run hye_stage1_4096 checkpoints/hye_stage1_4096 hye tokpremium.train.stage1 --ckpt checkpoints/hye_Armn_4096_meaninit
run hye_baseline_lr1e-4 checkpoints/hye_baseline_lr1e-4 hye tokpremium.train.baseline --lr 1e-4
