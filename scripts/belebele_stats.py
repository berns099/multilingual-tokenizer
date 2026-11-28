"""Paired analysis of the Belebele results.

    python scripts/belebele_stats.py results/belebele_hye_Armn_stage2.json

Accuracy alone cannot say whether two arms differ, because the arms answer the
SAME 900 questions. McNemar's test uses only the discordant pairs, which is the
right denominator: two arms that differ on 20 questions out of 900 are being
compared on those 20, not on all 900.

The exact binomial form is used rather than the chi-square approximation,
because the discordant counts here are small enough that the approximation is
unreliable.
"""

import argparse
import json
import math
import sys

PRED = "pred_sum_logprob"  # matches primary_metric accuracy_sum_logprob

ap = argparse.ArgumentParser(description=__doc__)
ap.add_argument("path", nargs="?", default="results/belebele_hye_Armn_stage2.json")
args = ap.parse_args()

d = json.load(open(args.path, encoding="utf-8"))
arms = list(d["models"])

correct = {}
for name, m in d["models"].items():
    correct[name] = {ex["question_number"]: int(ex[PRED] == ex["gold"]) for ex in m["examples"]}

qs = sorted(set.intersection(*(set(v) for v in correct.values())))
n = len(qs)


def wilson(k, n, z=1.96):
    """Score interval. Better than normal-approximation near 0 or 1, and these
    accuracies sit close to the 0.25 chance floor."""
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


print(f"Belebele {d['lang']}  n={n}  chance={d['chance_accuracy']:.3f}  "
      f"metric={d['primary_metric']}")
print()
print(f"{'arm':10s} {'correct':>8s} {'accuracy':>9s}  {'95% Wilson CI':20s} check")
for name in arms:
    k = sum(correct[name][q] for q in qs)
    lo, hi = wilson(k, n)
    said = d["models"][name].get("correct_sum_logprob")
    chk = "ok" if said in (None, k) else f"MISMATCH: file says {said}"
    print(f"{name:10s} {k:8d} {k / n:9.4f}  [{lo:.4f}, {hi:.4f}]      {chk}")

print()
print("Paired contrasts, McNemar exact, two-sided.")
print("b = left arm right and right arm wrong; c = the reverse.")
print()
print(f"{'contrast':24s} {'b':>4s} {'c':>4s} {'disc':>5s} {'acc diff':>9s} {'p':>8s}")
for a, z in [("stage2", "base"), ("stage2", "stock"), ("stage1", "stock"),
             ("stage2", "stage1"), ("base", "stock"), ("stage1", "base")]:
    if a not in correct or z not in correct:
        continue
    b = sum(1 for q in qs if correct[a][q] and not correct[z][q])
    c = sum(1 for q in qs if not correct[a][q] and correct[z][q])
    diff = (sum(correct[a][q] for q in qs) - sum(correct[z][q] for q in qs)) / n
    print(f"{a + ' vs ' + z:24s} {b:4d} {c:4d} {b + c:5d} {diff:+9.4f} "
          f"{mcnemar_exact(b, c):8.4f}")

print()
print("Scoring-rule sensitivity. Belebele scores four candidate answers, and")
print("summed log-probability favours short answers while per-byte does not.")
print("Low agreement means the headline number depends on that choice.")
print()
for name in arms:
    m = d["models"][name]
    print(f"  {name:10s} agreement={m['scoring_agreement']:.4f}  "
          f"acc_sum_logprob={m['accuracy_sum_logprob']:.4f}  "
          f"acc_per_byte={m['accuracy_logprob_per_byte']:.4f}")

sys.exit(0)
