"""Stage 2: unfreeze the body and continue from a stage-1 checkpoint.

    python -m tokpremium.train.stage2 --ckpt checkpoints/hye_stage1_4096 \
        --out checkpoints/hye_stage2 --corpus data/hye_train.txt \
        --heldout data/hye_heldout.txt --english data/eng_probe.txt \
        --replay data/eng_replay.txt --replay-ratio 0.25 \
        --tokens 2e6 --lr 2e-5 --seed 0

Stage 1 trains only the appended embedding rows against a frozen body, so the
body has never seen the new tokens. That is the condition this stage repairs,
and it is the likeliest explanation for the Belebele drop we measured after
stage 1 (28.2% against stock's 31.4%, McNemar p = 0.046). `configs/vocab.yaml`
specifies this stage as `trainable: all`, `data: telugu_plus_english`.

MATCHED CONTROL. The comparison arm is the existing baseline with the ORIGINAL
tokenizer and the body unfrozen, given the SAME replay mixture:

    python -m tokpremium.train.baseline --trainable all \
        --out checkpoints/hye_baseline_stage2 --corpus data/hye_train.txt \
        --heldout data/hye_heldout.txt --english data/eng_probe.txt \
        --replay data/eng_replay.txt --replay-ratio 0.25 \
        --tokens 2e6 --lr 2e-5 --seed 0

Every flag after --out must match between the two arms. Replay especially: if
only one arm replays English, the English result measures the replay rather
than the tokenizer, which is a fourth confound on top of the three the midterm
feedback already named.

BUDGET. --tokens is matched compute; --bytes is matched text. They are
different experiments and cannot both hold, because the extended tokenizer
packs more bytes into every token. run_training records which was used.

REPLAY. --replay must not be the --english probe; that file is held out for
measuring retention and training on it would make the measurement meaningless.
run_training refuses if the two paths resolve to the same file. The config
fixes no ratio, so 0.25 below is a starting point rather than a finding.

LEARNING RATE. add_common_args defaults to 1e-3, which suits freshly
initialized rows and will damage a pretrained body. This module defaults to
2e-5 instead. Raise it deliberately or not at all.

MEMORY. Every parameter is trainable, so AdamW holds two moments for all of
them: roughly 2.4 GB of parameters, 2.4 GB of gradients and 4.8 GB of
optimizer state at float32, before activations. That fits a 32 GB machine and
will not fit a 16 GB one. --save-every defaults to 100 here rather than 50
because each resume checkpoint now writes the whole model.
"""

from __future__ import annotations

import argparse
import sys

import torch
from transformers import AutoModelForCausalLM, PreTrainedTokenizerFast

from . import common


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--ckpt", required=True, help="stage-1 output directory")
    ap.add_argument(
        "--trainable",
        choices=["all", "body"],
        default="all",
        help="all: body and every embedding row, which is what the config "
        "specifies. body: embeddings held fixed, transformer only; use it to "
        "attribute a stage-2 gain to the body rather than to further "
        "embedding movement.",
    )
    common.add_common_args(ap)
    # Override add_common_args' stage-1 defaults; see the module docstring.
    ap.set_defaults(lr=2e-5, save_every=100)
    args = ap.parse_args(argv)

    if not args.replay:
        print(
            "warning: no --replay. The config specifies target-plus-English for "
            "this stage, and our October control lost 5.7% of its English "
            "without it.",
            file=sys.stderr,
            flush=True,
        )

    tok = PreTrainedTokenizerFast.from_pretrained(args.ckpt)
    model = AutoModelForCausalLM.from_pretrained(args.ckpt, dtype=torch.float32).to(
        common.device()
    )

    for p in model.parameters():
        p.requires_grad_(True)
    if args.trainable == "body":
        model.get_input_embeddings().weight.requires_grad_(False)
        if not model.config.tie_word_embeddings:
            model.get_output_embeddings().weight.requires_grad_(False)

    params = [p for p in model.parameters() if p.requires_grad]
    total = sum(p.numel() for p in model.parameters())
    print(
        f"stage 2: trainable={args.trainable}, "
        f"{sum(p.numel() for p in params) / 1e6:.1f}M of {total / 1e6:.1f}M params, "
        f"lr={args.lr:g}, vocab={model.get_input_embeddings().weight.shape[0]:,}",
        flush=True,
    )

    # old_rows=None deliberately: run_training's snapshot check asserts that
    # rows below an index stayed bit-identical, which is exactly what this
    # stage is meant to change. Passing an index would record a certain False.
    common.run_training(model, tok, args, params, old_rows=None)
    return 0


if __name__ == "__main__":
    sys.exit(main())
