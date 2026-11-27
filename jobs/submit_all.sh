#!/bin/bash
# Submit the Armenian run as five short chained jobs.
#
#   bash jobs/submit_all.sh
#
# WHY SHORT JOBS. The class allocation runs at roughly nine times its
# fairshare, so a long job waits for a long window: one 2-hour job sat pending
# for 22 hours while a 12-minute job started immediately. Short jobs get in
# through backfill, which is the scheduler using gaps ahead of reserved work.
#
# Each stage waits for the previous to succeed (afterok), so this runs
# unattended. A failed stage leaves the rest queued but never started; fix it
# and rerun this script, because every stage skips itself if already finished.
#
# Step 1 goes to `main` rather than `gpu`: init_embeddings never moves the
# model to a device, so queuing it for a GPU would wait for hardware it does
# not touch.

set -euo pipefail
cd "$(dirname "$0")/.."

if [ ! -f jobs/env.sh ]; then
    echo "run this from anywhere, but the repo must contain jobs/env.sh" >&2
    exit 1
fi

mkdir -p logs

MAIL="--mail-type=END,FAIL --mail-user=${USER}@usc.edu"

prev=""
for stage in jobs/01_init.sbatch jobs/02_stage1.sbatch jobs/03_stage2.sbatch \
             jobs/04_baseline.sbatch jobs/05_eval.sbatch; do
    if [ -z "$prev" ]; then
        id=$(sbatch $MAIL --parsable "$stage")
    else
        id=$(sbatch $MAIL --parsable --dependency=afterok:"$prev" "$stage")
    fi
    printf '%-28s %s\n' "$stage" "$id"
    prev="$id"
done

echo
echo "chain submitted. watch it with:  squeue -u $USER"
echo "logs land in logs/ as each stage starts."
