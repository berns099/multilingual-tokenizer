"""Which vocabulary entries can a tokenizer never produce?

    python -m tokpremium.vocab.audit

Motivated by side_effects.py: loosening Qwen's split regex, with no new merges,
cut Thai by 43% of its tokens, because Qwen's vocabulary already holds whole
Thai words its own regex cuts apart before BPE runs. Those rows are dead
weight in the embedding matrix. This counts them, per tokenizer and script.

For every token in a byte-level BPE tokenizer's model vocabulary (special
tokens excluded), the token's own text is encoded and classified:

  reachable              encodes to exactly that token
  split_by_pretokenizer  the split regex cuts its text into more than one chunk,
                         so BPE never sees it whole
  changed_by_normalizer  the normalizer rewrites its text (e.g. NFC), so the
                         raw form never reaches BPE
  merge_order            one chunk, unchanged, yet BPE's merge priorities build
                         a different split of it
  fragment               not valid UTF-8 on its own (part of a multi-byte
                         character); produced inside longer text, not audited

For split_by_pretokenizer tokens, `unlocked_by_keep_marks` counts those that
encode to themselves once combining marks are kept inside words
(continue_bpe.keep_marks), for tokenizers whose regex has Qwen's shape.

The test is on each token's text in isolation, the usual self-encoding check.
Context can attach a character to the front of a chunk but does not remove a
cut inside the token's own text, so split_by_pretokenizer is a firm lower bound
on unreachable tokens; merge_order is softer, since a token can occasionally
appear inside longer text even when its own string splits differently.

SentencePiece-style tokenizers (Mistral) are left out: their pieces depend on
position within a word, so a token's isolated text is not a meaningful probe.

Writes results/vocab_study/unreachable_tokens.csv (counts per tokenizer and
script) and unreachable_examples.csv (a few blocked tokens per script).
"""

from __future__ import annotations

import argparse
import json
import sys
import unicodedata
from collections import Counter
from pathlib import Path

import pandas as pd

from . import continue_bpe

TOKENIZERS = ["Qwen/Qwen3-0.6B", "meta-llama/Llama-3.2-1B", "Xenova/gpt-4", "Xenova/gpt-4o"]
CATEGORIES = ["reachable", "split_by_pretokenizer", "changed_by_normalizer", "merge_order",
              "fragment"]


def script_of(text: str) -> str:
    """Majority Unicode script of the letters and marks in `text`, from character names."""
    names = [
        unicodedata.name(ch, "UNKNOWN").split(" ")[0]
        for ch in text
        if ch.isalpha() or unicodedata.category(ch).startswith("M")
    ]
    if not names:
        return "none"
    script = Counter(names).most_common(1)[0][0]
    return {"CJK": "Han", "COMBINING": "Combining"}.get(script, script.title())


def classify(spec: dict) -> pd.DataFrame:
    """One row per model-vocab token: id, text, script, category, chunks."""
    from tokenizers import Tokenizer

    tok = Tokenizer.from_str(json.dumps(spec))
    special = {t["content"] for t in spec.get("added_tokens", [])}
    items = [(t, i) for t, i in spec["model"]["vocab"].items() if t not in special]

    rows, texts = [], []
    for token, tid in items:
        try:
            text = continue_bpe.piece_bytes(token).decode("utf-8")
        except (UnicodeDecodeError, KeyError):
            rows.append({"id": tid, "text": "", "category": "fragment"})
            continue
        rows.append({"id": tid, "text": text, "category": None})
        texts.append(len(rows) - 1)

    encoded = tok.encode_batch([rows[k]["text"] for k in texts], add_special_tokens=False)
    for k, enc in zip(texts, encoded, strict=True):
        row = rows[k]
        if enc.ids == [row["id"]]:
            row["category"] = "reachable"
            continue
        text = row["text"]
        norm = tok.normalizer.normalize_str(text) if tok.normalizer else text
        chunks = [c for c, _ in tok.pre_tokenizer.pre_tokenize_str(norm)]
        row["chunks"] = len(chunks)
        if len(chunks) > 1:
            row["category"] = "split_by_pretokenizer"
        elif norm != text:
            row["category"] = "changed_by_normalizer"
        else:
            row["category"] = "merge_order"

    df = pd.DataFrame(rows)
    if "chunks" not in df:  # every token reachable: no row was pre-tokenized
        df["chunks"] = pd.NA
    df["script"] = [script_of(t) for t in df["text"]]
    df["unlocked_by_keep_marks"] = False
    try:
        marked = Tokenizer.from_str(json.dumps(continue_bpe.keep_marks(spec)))
    except ValueError:  # regex not Qwen-shaped (GPT-4o already keeps marks)
        df["unlocked_by_keep_marks"] = pd.NA
        return df
    split = df.index[df["category"] == "split_by_pretokenizer"]
    enc = marked.encode_batch(df.loc[split, "text"].tolist(), add_special_tokens=False)
    df.loc[split, "unlocked_by_keep_marks"] = [
        e.ids == [i] for e, i in zip(enc, df.loc[split, "id"], strict=True)
    ]
    return df


def summarize(df: pd.DataFrame, tokenizer_id: str) -> pd.DataFrame:
    counts = pd.crosstab(df["script"], df["category"]).reindex(columns=CATEGORIES, fill_value=0)
    counts["tokens"] = counts.sum(axis=1)
    unlocked = df.groupby("script")["unlocked_by_keep_marks"].sum(min_count=1)
    counts["unlocked_by_keep_marks"] = unlocked.reindex(counts.index)
    counts.insert(0, "tokenizer", tokenizer_id)
    return counts.reset_index()


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--tokenizer", action="append", help="limit to these ids (repeatable)")
    ap.add_argument("--out", default="results/vocab_study")
    ap.add_argument("--examples", type=int, default=5, help="blocked examples kept per script")
    args = ap.parse_args(argv)

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    summaries, examples = [], []
    for tid in args.tokenizer or TOKENIZERS:
        df = classify(continue_bpe.load_base_spec(tid))
        summaries.append(summarize(df, tid))
        blocked = df[df["category"] == "split_by_pretokenizer"]
        if len(blocked):
            examples.append(
                blocked.groupby("script").head(args.examples)
                .assign(tokenizer=tid)[["tokenizer", "script", "id", "text", "chunks",
                                        "unlocked_by_keep_marks"]]
            )
        total = df["category"].value_counts()
        print(f"{tid}: {len(df):,} tokens | " + ", ".join(
            f"{c} {total.get(c, 0):,}" for c in CATEGORIES), flush=True)

    pd.concat(summaries, ignore_index=True).to_csv(out / "unreachable_tokens.csv", index=False)
    pd.concat(examples, ignore_index=True).to_csv(out / "unreachable_examples.csv", index=False)
    print(f"wrote {out / 'unreachable_tokens.csv'} and {out / 'unreachable_examples.csv'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
