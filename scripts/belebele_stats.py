"""Paired analysis of the Belebele results.

    python scripts/belebele_stats.py results/belebele_hye_Armn_stage2.json

Accuracy alone cannot say whether two arms differ, because the arms answer the
SAME questions. McNemar's test uses only the discordant pairs, which is the
right denominator: two arms differing on 20 of 900 questions are being compared
on those 20. The exact binomial form is used rather than chi-square, because
the discordant counts are small enough that the approximation is unreliable.

PAIRING IS BY POSITION. `question_number` is the question's index within its
passage, not a unique identifier, so it cannot pair the arms; an earlier
version of this script keyed on it and silently collapsed 900 examples into 2.
Every arm scores the same dataset in the same order, so index i is the same
question across arms, and that is checked rather than assumed: `gold` and
`question_number` must agree at every index, and each arm's recomputed correct
count must match the count the file itself reports.

BOTH SCORING RULES ARE REPORTED. Belebele scores four candidate answers.
Summed log-probability favours short answers; dividing by bytes removes that
but introduces its own bias. The two rules agree on only 44 to 63 percent of
questions here and they do not even rank the arms the same way, so reporting
one without the other would be a quiet choice with a loud effect.
"""

import argparse
import json
import math
import sys

RULES = [("pred_sum_logprob", "correct_sum_logprob", "summed log-prob"),
         ("pred_logprob_per_byte", "correct_logprob_per_byte", "log-prob per byte")]

CONTRASTS = [("stage2", "base"), ("stage2", "stock"), ("stage1", "stock"),
             ("stage2", "stage1"), ("base", "stock"), ("stage1", "base")]


def wilson(k, n, z=1.96):
    """Score interval. Preferable to the normal approximation here because
    these accuracies sit close to the 0.25 chance floor."""
    if n == 0:
        return 0.0, 0.0
    p = k / n
    den = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / den
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return max(0.0, centre - half), min(1.0, centre + half)


def mcnemar_exact(b, c):
    """Two-sided exact test on the b+c discordant pairs."""
    nd = b + c
    if nd == 0:
        return 1.0
    tail = sum(math.comb(nd, i) for i in range(min(b, c) + 1)) / 2 ** nd
    return min(1.0, 2 * tail)


ap = argparse.ArgumentParser(description=__doc__)
ap.add_argument("path", nargs="?", default="results/belebele_hye_Armn_stage2.json")
args = ap.parse_args()

d = json.load(open(args.path, encoding="utf-8"))
arms = list(d["models"])
rows = {name: d["models"][name]["examples"] for name in arms}

counts = {name: len(v) for name, v in rows.items()}
if len(set(counts.values())) != 1:
    sys.exit(f"arms hold different numbers of examples: {counts}")
n = counts[arms[0]]

ref = rows[arms[0]]
for name in arms[1:]:
    for i, (a, b) in enumerate(zip(ref, rows[name])):
        if a["gold"] != b["gold"] or a["question_number"] != b["question_number"]:
            sys.exit(f"{arms[0]} and {name} are not aligned at index {i}; "
                     "pairing by position is invalid for this file")

print(f"Belebele {d['lang']}  n={n}  chance={d['chance_accuracy']:.3f}")
print(f"file's primary_metric: {d['primary_metric']}")

for pred_key, file_key, label in RULES:
    correct = {name: [int(e[pred_key] == e["gold"]) for e in rows[name]] for name in arms}

    print()
    print(f"================ scoring rule: {label} ({pred_key}) ================")
    print()
    print(f"{'arm':10s} {'correct':>8s} {'accuracy':>9s}  {'95% Wilson CI':18s}  check")
    for name in arms:
        k = sum(correct[name])
        lo, hi = wilson(k, n)
        said = d["models"][name].get(file_key)
        chk = "ok" if said in (None, k) else f"MISMATCH: file says {said}"
        print(f"{name:10s} {k:8d} {k / n:9.4f}  [{lo:.4f}, {hi:.4f}]  {chk}")

    print()
    print(f"{'contrast':24s} {'b':>4s} {'c':>4s} {'disc':>5s} {'acc diff':>9s} {'p':>8s}")
    for a, z in CONTRASTS:
        if a not in correct or z not in correct:
            continue
        b = sum(1 for i in range(n) if correct[a][i] and not correct[z][i])
        c = sum(1 for i in range(n) if not correct[a][i] and correct[z][i])
        diff = (sum(correct[a]) - sum(correct[z])) / n
        print(f"{a + ' vs ' + z:24s} {b:4d} {c:4d} {b + c:5d} {diff:+9.4f} "
              f"{mcnemar_exact(b, c):8.4f}")

print()
print("b = left arm right and right arm wrong; c = the reverse.")
print("Agreement between the two rules, per arm:")
for name in arms:
    print(f"  {name:10s} {d['models'][name]['scoring_agreement']:.4f}")
