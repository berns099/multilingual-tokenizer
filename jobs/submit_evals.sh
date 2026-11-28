#!/bin/bash
# Submit the three evaluators as independent short jobs.
#
#   bash jobs/submit_evals.sh
#
# Cancel 05_eval first: it does all three and would overwrite these outputs
# while they are being written. Each job here skips itself if its output
# already exists, so rerunning this script is safe.

set -euo pipefail
cd "$(dirname "$0")/.."
mkdir -p logs

MAIL="--mail-type=END,FAIL --mail-user=${USER}@usc.edu"

for stage in jobs/07_belebele.sbatch jobs/06_eval_bpb.sbatch jobs/08_inference.sbatch; do
    id=$(sbatch $MAIL --parsable "$stage")
    printf '%-28s %s\n' "$stage" "$id"
done

echo
echo "three independent jobs, no dependencies: any order, possibly in parallel."
echo "belebele is the one that matters for the report."
