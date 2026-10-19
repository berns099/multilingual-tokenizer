"""Stage 1: train ONLY the new embedding rows on Telugu. Body + old rows frozen.

    python -m tokpremium.train.stage1 --ckpt checkpoints/tel_Telu_1024_meaninit \
        --out checkpoints/tel_stage1 --corpus data/tel_train.txt \
        --heldout data/tel_heldout.txt --english data/eng_sample.txt --tokens 2e6

The model is tied, so the one embedding matrix is also the lm-head; freezing
rows < first_new freezes both roles for old tokens.
"""

from __future__ import annotations

import argparse
import sys

import torch
from transformers import AutoModelForCausalLM, PreTrainedTokenizerFast

from . import common


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--first-new", type=int, default=151669, help="first appended id")
    common.add_common_args(ap)
    args = ap.parse_args(argv)

    tok = PreTrainedTokenizerFast.from_pretrained(args.ckpt)
    model = AutoModelForCausalLM.from_pretrained(args.ckpt, dtype=torch.float32).to(common.device())
    assert model.config.tie_word_embeddings, "untied model: freeze lm_head rows too"
    for p in model.parameters():
        p.requires_grad_(False)
    emb = model.get_input_embeddings().weight
    emb.requires_grad_(True)
    common.freeze_old_rows(emb, args.first_new)
    common.run_training(model, tok, args, [emb], old_rows=args.first_new)
    return 0


if __name__ == "__main__":
    sys.exit(main())
