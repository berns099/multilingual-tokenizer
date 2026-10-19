"""Evaluate checkpoints on FLORES+ dev (never devtest): bits/byte + a generation check.

    python -m tokpremium.train.evaluate --lang hye_Armn \
        stock=Qwen/Qwen3-0.6B stage1=checkpoints/hye_stage1_4096 base=checkpoints/hye_baseline_lr1e-4

FLORES dev is held out from the FineWeb-2 training text (a different domain), and
the English side is the full parallel dev set, not a 5-sentence sample. The
generation check reports a repetition rate (share of repeated token 4-grams)
because multiple-choice-style scores can look healthy while a model loops.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, PreTrainedTokenizerFast

from ..data.flores import load_sentences
from . import common


def repetition_rate(ids: list[int], n: int = 4) -> float:
    grams = [tuple(ids[i : i + n]) for i in range(len(ids) - n + 1)]
    return 1 - len(set(grams)) / len(grams) if grams else 0.0


@torch.no_grad()
def generate(model, tok, prompt: str, new_tokens: int = 60) -> tuple[str, float]:
    dev = next(model.parameters()).device
    ids = torch.tensor([[common.EOT] + tok(prompt, add_special_tokens=False)["input_ids"]], device=dev)
    out = model.generate(ids, max_new_tokens=new_tokens, do_sample=False, repetition_penalty=1.0)
    new = out[0, ids.shape[1] :].tolist()
    return tok.decode(new), repetition_rate(new)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("models", nargs="+", help="name=path_or_hub_id")
    ap.add_argument("--lang", required=True)
    ap.add_argument("--out", default=None)
    ap.add_argument("--n-prompts", type=int, default=5)
    args = ap.parse_args(argv)

    target = load_sentences(args.lang, split="dev")
    english = load_sentences("eng_Latn", split="dev")
    results = {}
    for spec in args.models:
        name, path = spec.split("=", 1)
        tok = PreTrainedTokenizerFast.from_pretrained(path)
        model = AutoModelForCausalLM.from_pretrained(path, dtype=torch.bfloat16).to(common.device())
        row = {
            "bpb_target": common.bits_per_byte(model, tok, target),
            "bpb_english": common.bits_per_byte(model, tok, english),
            "tokens_target": sum(len(x) for x in tok(target, add_special_tokens=False)["input_ids"]),
            "tokens_english": sum(len(x) for x in tok(english, add_special_tokens=False)["input_ids"]),
        }
        gens = {"target": [], "english": []}
        for key, sents in (("target", target), ("english", english)):
            for s in sents[: args.n_prompts]:
                text, rep = generate(model, tok, s[:60])
                gens[key].append({"prompt": s[:60], "continuation": text, "repetition": rep})
            row[f"repetition_{key}"] = sum(g["repetition"] for g in gens[key]) / args.n_prompts
        row["generations"] = gens
        results[name] = row
        print(name, {k: round(v, 4) if isinstance(v, float) else v for k, v in row.items()
                     if k != "generations"}, flush=True)
        del model
        torch.mps.empty_cache()

    out = Path(args.out or f"results/vocab/{args.lang}/eval_flores_dev.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(results, ensure_ascii=False, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
