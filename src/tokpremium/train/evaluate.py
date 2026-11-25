"""Evaluate checkpoints on FLORES+ dev (never devtest): bits/byte + generation checks.

    python -m tokpremium.train.evaluate --lang hye_Armn \
        stock=Qwen/Qwen3-0.6B \
        stage1=checkpoints/hye_stage1_4096 \
        base=checkpoints/hye_baseline_lr1e-4

FLORES dev is held out from the FineWeb-2 training text (a different domain),
and the English side is the full parallel dev set, not a 5-sentence sample.

Every row carries a `provenance` block naming the checkpoint it came from,
including a hash of its tokenizer.json, so a result can always be tied back to
the exact artifact that produced it.

The generation check reports repeated token 4-gram statistics. Repetition is
reported as a distribution rather than only a mean because a small number of
severely looping generations can otherwise be hidden by the aggregate.
"""

from __future__ import annotations

import argparse
import json
import platform
import statistics
import sys
from datetime import datetime, timezone
from importlib.metadata import version
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, PreTrainedTokenizerFast

from ..data.flores import load_sentences
from . import common


def repetition_rate(ids: list[int], n: int = 4) -> float:
    """Share of token n-grams that are repetitions of an earlier n-gram."""
    grams = [tuple(ids[i : i + n]) for i in range(len(ids) - n + 1)]
    return 1 - len(set(grams)) / len(grams) if grams else 0.0


def summarize_repetition(values: list[float]) -> dict[str, float]:
    """Summarize repetition rates without imposing an arbitrary threshold."""
    if not values:
        raise ValueError("no repetition measurements")

    ordered = sorted(values)
    n = len(ordered)

    # Nearest-rank p90: keep the percentile as an observed measurement rather
    # than interpolating between two prompts.
    p90 = ordered[max(0, (9 * n + 9) // 10 - 1)]

    return {
        "mean": statistics.mean(ordered),
        "median": statistics.median(ordered),
        "p90": p90,
        "max": ordered[-1],
    }


@torch.no_grad()
def generate(
    model,
    tok,
    prompt: str,
    new_tokens: int = 60,
) -> tuple[str, float]:
    """Greedy generation plus repeated-4gram rate."""
    dev = next(model.parameters()).device
    prompt_ids = tok(prompt, add_special_tokens=False)["input_ids"]
    ids = torch.tensor([[common.EOT] + prompt_ids], device=dev)

    out = model.generate(
        ids,
        max_new_tokens=new_tokens,
        do_sample=False,
        repetition_penalty=1.0,
    )

    new = out[0, ids.shape[1] :].tolist()
    return tok.decode(new), repetition_rate(new)


def model_provenance(path: str, tok, model) -> dict:
    """Identify exactly which artifact produced a row of results.

    The midterm feedback was that evaluation files could not be linked to
    exact checkpoints, and two runs of the Armenian pipeline disagreed with
    nothing in the repository recording which vocabulary build each used. A
    row is only as traceable as the identity stored beside it, so every row
    carries its own: hashing tokenizer.json distinguishes two extensions of
    the same base model even when every other field matches, and repeating
    the run-level git and environment block per row keeps each row readable
    on its own rather than only as part of the file it arrived in.
    """
    p = Path(path)
    prov: dict = {
        "spec": path,
        "local": p.exists(),
        "vocab_size": len(tok),
        "n_parameters": sum(x.numel() for x in model.parameters()),
        "git": common.git_provenance(),
        "environment": {
            "python": platform.python_version(),
            "torch": str(torch.__version__),
            "transformers": version("transformers"),
        },
        "evaluated_at_utc": datetime.now(timezone.utc).isoformat(),
    }

    if not p.exists():
        # A hub id: record the revision transformers resolved, when it kept one.
        prov["hub_id"] = path
        commit = getattr(model.config, "_commit_hash", None)
        if commit:
            prov["hub_commit"] = commit
        return prov

    prov["resolved_path"] = str(p.resolve())
    # tokenizer.json is the file that distinguishes two vocabulary builds;
    # the weight shards identify the trained model itself.
    wanted = [p / "tokenizer.json", p / "config.json"]
    wanted += sorted(p.glob("*.safetensors")) + sorted(p.glob("*.bin"))
    prov["files"] = {
        f.name: common.file_fingerprint(f) for f in wanted if f.is_file()
    }
    return prov


def clear_device_cache(dev: torch.device) -> None:
    """Release accelerator cache when the backend provides one."""
    if dev.type == "mps":
        torch.mps.empty_cache()
    elif dev.type == "cuda":
        torch.cuda.empty_cache()


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("models", nargs="+", help="name=path_or_hub_id")
    ap.add_argument("--lang", required=True)
    ap.add_argument("--out", default=None)
    ap.add_argument("--n-prompts", type=int, default=5)
    args = ap.parse_args(argv)

    if args.n_prompts <= 0:
        ap.error("--n-prompts must be positive")

    target = load_sentences(args.lang, split="dev")
    english = load_sentences("eng_Latn", split="dev")

    results = {}
    dev = common.device()

    for spec in args.models:
        if "=" not in spec:
            ap.error(f"model must be name=path_or_hub_id, got {spec!r}")

        name, path = spec.split("=", 1)
        if not name or not path:
            ap.error(f"model must be name=path_or_hub_id, got {spec!r}")
        if name in results:
            ap.error(f"duplicate model name: {name}")

        tok = PreTrainedTokenizerFast.from_pretrained(path)
        model = AutoModelForCausalLM.from_pretrained(
            path,
            dtype=torch.bfloat16,
        ).to(dev)

        row = {
            "bpb_target": common.bits_per_byte(model, tok, target),
            "bpb_english": common.bits_per_byte(model, tok, english),
            "tokens_target": sum(
                len(x)
                for x in tok(
                    target,
                    add_special_tokens=False,
                )["input_ids"]
            ),
            "tokens_english": sum(
                len(x)
                for x in tok(
                    english,
                    add_special_tokens=False,
                )["input_ids"]
            ),
        }

        gens = {"target": [], "english": []}

        for key, sents in (("target", target), ("english", english)):
            selected = sents[: args.n_prompts]

            for sentence in selected:
                prompt = sentence[:60]
                text, rep = generate(model, tok, prompt)
                gens[key].append(
                    {
                        "prompt": prompt,
                        "continuation": text,
                        "repetition": rep,
                    }
                )

            reps = [generation["repetition"] for generation in gens[key]]
            stats = summarize_repetition(reps)

            # Preserve the original scalar field so existing result consumers
            # remain compatible.
            row[f"repetition_{key}"] = stats["mean"]
            row[f"repetition_{key}_stats"] = stats
            row[f"n_prompts_{key}"] = len(reps)

        row["generations"] = gens
        row["provenance"] = model_provenance(path, tok, model)
        results[name] = row

        printable = {
            key: round(value, 4) if isinstance(value, float) else value
            for key, value in row.items()
            if key not in ("generations", "provenance")
        }
        print(name, printable, flush=True)

        del model
        clear_device_cache(dev)

    out = Path(
        args.out
        or f"results/vocab/{args.lang}/eval_flores_dev.json"
    )
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(
        json.dumps(results, ensure_ascii=False, indent=1),
        encoding="utf-8",
    )

    print(f"wrote {out}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())