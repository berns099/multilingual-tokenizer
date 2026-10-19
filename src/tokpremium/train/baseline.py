"""Matched-compute baseline: ORIGINAL tokenizer, same Telugu documents.

    python -m tokpremium.train.baseline --out checkpoints/tel_baseline_emb \
        --corpus data/tel_train.txt --heldout data/tel_heldout.txt \
        --english data/eng_sample.txt --tokens 2e6

Same loop, token budget, batch, steps, LR schedule and seed as stage 1, so
compute is matched; the two arms differ in tokenizer, hence in bytes of text
seen (reported as bytes_seen). --trainable emb trains the whole embedding
matrix (closest analogue of stage 1: nothing else moves); `all` unfreezes the
body (for comparing against stage 2 later).
"""

from __future__ import annotations

import argparse
import sys

import torch
from transformers import AutoModelForCausalLM, PreTrainedTokenizerFast

from . import common


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", default="Qwen/Qwen3-0.6B")
    ap.add_argument("--trainable", choices=["emb", "all"], default="emb")
    common.add_common_args(ap)
    args = ap.parse_args(argv)

    tok = PreTrainedTokenizerFast.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(args.model, dtype=torch.float32).to(common.device())
    for p in model.parameters():
        p.requires_grad_(args.trainable == "all")
    emb = model.get_input_embeddings().weight
    emb.requires_grad_(True)
    params = [p for p in model.parameters() if p.requires_grad]
    common.run_training(model, tok, args, params)
    return 0


if __name__ == "__main__":
    sys.exit(main())
