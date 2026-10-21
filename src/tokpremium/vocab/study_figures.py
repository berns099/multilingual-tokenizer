"""Figures for the tokenizer study, drawn from committed CSVs only.

    python -m tokpremium.vocab.study_figures

  results/vocab_study/figures/ceiling.png
      Median premium vs new tokens, one panel per (model, language), shipped vs
      keep_marks_target. The flat shipped line is the vowel-sign ceiling.
  results/vocab_study/figures/ceiling_slide.png
      The same finding cut down for a slide: Telugu vs Amharic only, plain
      labels, results written on the chart.
  results/vocab_study/figures/unreachable.png
      Unreachable vocabulary per tokenizer, and Qwen's by script and cause.

Nothing is recomputed here; rerun continue_bpe/measure/audit to change the data.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

INK, INK_2, MUTED, GRID, AXIS = "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7"
BLUE, ORANGE = "#2a78d6", "#eb6834"

# (summary.csv, lang, panel title); the first is the team's step B run.
CEILING_RUNS = [
    ("results/vocab/summary.csv", "tel_Telu", "Telugu · Qwen3"),
    ("results/vocab_study/qwen/summary.csv", "tam_Taml", "Tamil · Qwen3"),
    ("results/vocab_study/llama/summary.csv", "tel_Telu", "Telugu · Llama 3.2"),
    ("results/vocab_study/qwen/summary.csv", "amh_Ethi", "Amharic · Qwen3 (no vowel marks)"),
]
RULES = [("shipped", "split regex as shipped", BLUE, "-", "o"),
         ("keep_marks_target", "target script's marks kept in words", ORANGE, "--", "s")]


def _pct(share: float) -> str:
    """Percent with enough digits that a nonzero share never prints as 0.0%."""
    return f"{share:.1f}%" if share >= 0.1 or share == 0 else f"{share:.2f}%"


def _style(ax):
    ax.grid(axis="y", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color(AXIS)
    ax.tick_params(colors=INK_2, length=0)


def ceiling(out: Path) -> None:
    runs = [(pd.read_csv(path).query("lang == @lang"), title) for path, lang, title in CEILING_RUNS]
    top = max(df["premium_median"].max() for df, _ in runs) * 1.08
    fig, axes = plt.subplots(1, len(runs), figsize=(13, 3.9), sharey=True)
    for ax, (df, title) in zip(axes, runs, strict=True):
        sizes = sorted(df["n_new"].unique())
        pos = {n: i for i, n in enumerate(sizes)}
        labelled = set()
        for rule, label, color, ls, marker in RULES:
            s = df[df["pretokenizer"] == rule].sort_values("n_new")
            x = [pos[n] for n in s["n_new"]]
            ax.plot(x, s["premium_median"], color=color, linestyle=ls, linewidth=2, marker=marker,
                    markersize=6, markeredgecolor="white", markeredgewidth=1, label=label, zorder=3)
            end = f"{s['premium_median'].iloc[-1]:.2f}×"
            if end not in labelled:  # both rules end at the same value for Amharic
                labelled.add(end)
                ax.annotate(end, (x[-1], s["premium_median"].iloc[-1]), xytext=(6, 0),
                            textcoords="offset points", va="center", fontsize=8.5, color=INK_2)
        ax.axhline(1, color=MUTED, linewidth=1, linestyle=":", zorder=1)
        ax.set_xticks(range(len(sizes)), [f"+{n // 1024}k" if n else "0" for n in sizes])
        ax.set_xlim(-0.3, len(sizes) - 0.4)
        ax.set_ylim(0, top)
        ax.set_title(title, loc="left", color=INK, fontsize=10.5)
        ax.set_xlabel("new tokens", color=INK_2)
        _style(ax)
    axes[0].set_ylabel("median tokens vs English", color=INK_2)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper left", bbox_to_anchor=(0.01, 1.0), ncol=2,
               frameon=False, fontsize=9, labelcolor=INK_2)
    fig.text(0.01, 0.01, "FLORES+ devtest, paired median premium. With the regex as shipped, "
             "scripts whose vowels are combining marks stop improving; "
             "Amharic has none and does not.", color=MUTED, fontsize=8)
    fig.tight_layout(rect=(0, 0.04, 1, 0.92))
    fig.savefig(out, dpi=150)
    plt.close(fig)


def ceiling_slide(out: Path) -> None:
    """Two panels, big type, every number the audience needs written on the chart."""
    panels = [
        ("results/vocab/summary.csv", "tel_Telu",
         "Telugu: vowels written as separate signs"),
        ("results/vocab_study/qwen/summary.csv", "amh_Ethi",
         "Amharic: vowels built into each letter"),
    ]
    rules = [("shipped", "Qwen's original word-splitting (cuts at vowel signs)", BLUE, "-", "o"),
             ("keep_marks_target", "Vowel signs kept with their word", ORANGE, "--", "s")]
    data = [(pd.read_csv(p).query("lang == @lang"), t) for p, lang, t in panels]
    top = max(df["premium_median"].max() for df, _ in data) * 1.12

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.6), sharey=True)
    for ax, (df, title) in zip(axes, data, strict=True):
        sizes = sorted(df["n_new"].unique())
        pos = {n: i for i, n in enumerate(sizes)}
        ends = {}
        for rule, label, color, ls, marker in rules:
            s = df[df["pretokenizer"] == rule].sort_values("n_new")
            ax.plot([pos[n] for n in s["n_new"]], s["premium_median"], color=color, linestyle=ls,
                    linewidth=3, marker=marker, markersize=8, markeredgecolor="white",
                    markeredgewidth=1.5, label=label, zorder=3)
            ends[rule] = s["premium_median"].iloc[-1]
        start = df.loc[df["n_new"] == 0, "premium_median"].iloc[0]
        ax.annotate(f"{start:.1f}× today", (0, start), xytext=(10, 0), textcoords="offset points",
                    va="center", fontsize=11, color=INK)
        x_end = len(sizes) - 1
        if abs(ends["shipped"] - ends["keep_marks_target"]) < 0.05:
            ax.annotate(f"{ends['shipped']:.2f}×\nno gap", (x_end, ends["shipped"]),
                        xytext=(10, 0), textcoords="offset points", va="center", fontsize=11,
                        color=INK, fontweight="bold")
        else:
            ax.annotate(f"{ends['shipped']:.2f}×  stuck", (x_end, ends["shipped"]),
                        xytext=(10, 0), textcoords="offset points", va="center", fontsize=11,
                        color=BLUE, fontweight="bold")
            ax.annotate(f"{ends['keep_marks_target']:.2f}×  keeps falling",
                        (x_end, ends["keep_marks_target"]), xytext=(10, 0),
                        textcoords="offset points", va="center", fontsize=11, color=ORANGE,
                        fontweight="bold")
        ax.axhline(1, color=MUTED, linewidth=1.2, linestyle=":", zorder=1)
        ax.text(0, 1, "English = 1×", va="bottom", ha="left", fontsize=9.5, color=MUTED)
        ax.set_xticks(range(len(sizes)), [f"{n // 1024}k" if n else "0" for n in sizes],
                      fontsize=11)
        ax.set_xlim(-0.2, len(sizes) + 0.6)
        ax.set_ylim(0, top)
        ax.set_title(title, loc="left", color=INK, fontsize=13)
        ax.set_xlabel("new tokens added to Qwen's vocabulary", color=INK_2, fontsize=11)
        ax.tick_params(axis="y", labelsize=11)
        _style(ax)
    axes[0].set_ylabel("tokens per sentence vs English", color=INK_2, fontsize=11)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper left", bbox_to_anchor=(0.01, 1.0), ncol=2,
               frameon=False, fontsize=11, labelcolor=INK)
    fig.text(0.01, 0.01, "Same result for Tamil, and for Llama 3.2's tokenizer. "
             "Measured on FLORES+ test sentences.", color=MUTED, fontsize=9.5)
    fig.tight_layout(rect=(0, 0.04, 1, 0.91))
    fig.savefig(out, dpi=200)
    plt.close(fig)


def unreachable(out: Path, csv: str = "results/vocab_study/unreachable_tokens.csv") -> None:
    d = pd.read_csv(csv)
    d["unreachable"] = d["split_by_pretokenizer"] + d["changed_by_normalizer"] + d["merge_order"]
    fig, (left, right) = plt.subplots(1, 2, figsize=(12, 4.2),
                                      gridspec_kw={"width_ratios": [1, 1.6]})

    totals = d.groupby("tokenizer", sort=False)[["unreachable", "tokens"]].sum()
    totals = totals.sort_values("unreachable")
    names = [t.split("/")[-1] for t in totals.index]
    left.barh(names, totals["unreachable"], color=BLUE, height=0.55)
    for y, (n, t) in enumerate(zip(totals["unreachable"], totals["tokens"], strict=True)):
        left.annotate(f"{n:,}  ({_pct(100 * n / t)} of vocab)", (n, y), xytext=(4, 0),
                      textcoords="offset points", va="center", fontsize=8.5, color=INK_2)
    left.set_xlim(0, totals["unreachable"].max() * 1.6)
    left.set_title("Vocabulary entries the tokenizer can never produce", loc="left",
                   color=INK, fontsize=10.5)
    left.set_xlabel("tokens", color=INK_2)
    _style(left)
    left.grid(axis="x", color=GRID, linewidth=0.8)
    left.grid(axis="y", visible=False)

    q = d[(d["tokenizer"] == "Qwen/Qwen3-0.6B") & (d["unreachable"] >= 10)]
    q = q.sort_values("unreachable")
    y = range(len(q))
    right.barh(y, q["split_by_pretokenizer"], color=BLUE, height=0.55,
               label="cut apart by the split regex")
    right.barh(y, q["changed_by_normalizer"], left=q["split_by_pretokenizer"], color=ORANGE,
               height=0.55, label="rewritten by NFC normalization")
    for k, (_, r) in enumerate(q.iterrows()):
        right.annotate(f"{r['unreachable']:,} = {_pct(100 * r['unreachable'] / r['tokens'])} of "
                       f"{r['script']} tokens", (r["unreachable"], k), xytext=(4, 0),
                       textcoords="offset points", va="center", fontsize=8.5, color=INK_2)
    right.set_yticks(list(y), q["script"])
    right.set_xlim(0, q["unreachable"].max() * 1.7)
    right.set_title("Qwen3 by script and cause", loc="left", color=INK, fontsize=10.5)
    right.set_xlabel("unreachable tokens", color=INK_2)
    right.legend(loc="lower right", frameon=False, fontsize=8.5, labelcolor=INK_2)
    _style(right)
    right.grid(axis="x", color=GRID, linewidth=0.8)
    right.grid(axis="y", visible=False)

    qwen = d[d["tokenizer"] == "Qwen/Qwen3-0.6B"]
    blocked = qwen["split_by_pretokenizer"].sum()
    unlocked = int(qwen["unlocked_by_keep_marks"].sum())
    fig.text(0.01, 0.01, "Each token's own text is encoded; unreachable = it never comes back as "
             f"itself. Keeping combining marks in words unlocks {unlocked:,} of the {blocked:,} "
             "regex-blocked Qwen tokens.", color=MUTED, fontsize=8)
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    fig.savefig(out, dpi=150)
    plt.close(fig)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--outdir", default="results/vocab_study/figures")
    args = ap.parse_args(argv)
    out = Path(args.outdir)
    out.mkdir(parents=True, exist_ok=True)
    ceiling(out / "ceiling.png")
    ceiling_slide(out / "ceiling_slide.png")
    unreachable(out / "unreachable.png")
    print(f"wrote ceiling.png, ceiling_slide.png and unreachable.png to {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
