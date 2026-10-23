"""Zero-shot Belebele multiple-choice evaluation.

Scores each answer by its conditional likelihood given the passage and
question. Reports both summed answer log-probability and UTF-8-byte-normalized
log-probability so tokenizer/answer-length sensitivity is visible.

Example:

    python -m tokpremium.train.evaluate_belebele \
        --lang hye_Armn \
        stock=Qwen/Qwen3-0.6B \
        stage1=checkpoints/hye_stage1_4096 \
        base=checkpoints/hye_baseline_lr1e-4
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

from . import common


def build_prompt(passage: str, question: str) -> str:
    """Shared context presented before every candidate answer."""
    return f"Passage: {passage}\nQuestion: {question}\nAnswer:"


@torch.no_grad()
def answer_scores(
    model,
    tok,
    prompt: str,
    answer: str,
) -> dict[str, float | int]:
    """Conditional log-probability of one answer.

    Prompt and answer are tokenized separately so the answer boundary is
    explicit and cannot move when a BPE tokenizer sees the concatenated text.
    A leading space is included with the answer because the answer follows the
    ``Answer:`` label in the rendered prompt.

    Only answer tokens contribute to the score.
    """
    dev = next(model.parameters()).device

    prompt_ids = tok(
        prompt,
        add_special_tokens=False,
    )["input_ids"]

    answer_text = " " + answer
    answer_ids = tok(
        answer_text,
        add_special_tokens=False,
    )["input_ids"]

    if not answer_ids:
        raise ValueError("answer produced no scoreable tokens")

    full_ids = [common.EOT] + prompt_ids + answer_ids
    ids = torch.tensor([full_ids], device=dev)

    logits = model(ids).logits[0]

    # full_ids layout:
    # [EOT] [prompt ...] [answer ...]
    #
    # The first answer token is at position 1 + len(prompt_ids), and is
    # predicted by logits from the immediately preceding position.
    first_answer_position = 1 + len(prompt_ids)

    answer_token_ids = ids[0, first_answer_position:]
    prediction_logits = logits[
        first_answer_position - 1 : ids.shape[1] - 1
    ]

    if prediction_logits.shape[0] != answer_token_ids.shape[0]:
        raise RuntimeError("answer/logit alignment mismatch")

    log_probs = torch.log_softmax(
        prediction_logits.float(),
        dim=-1,
    )

    token_log_probs = log_probs.gather(
        1,
        answer_token_ids.unsqueeze(1),
    ).squeeze(1)

    total = token_log_probs.sum().item()
    n_tokens = answer_token_ids.numel()

    # Normalize by the actual answer content rather than the artificial
    # separator space used to establish the tokenization boundary.
    n_bytes = len(answer.encode("utf-8"))
    if n_bytes == 0:
        raise ValueError("answer is empty")

    return {
        "sum_logprob": total,
        "logprob_per_byte": total / n_bytes,
        "answer_tokens": n_tokens,
        "answer_bytes": n_bytes,
    }


def choose_answer(
    scores: list[dict[str, float | int]],
    metric: str,
) -> int:
    """Return the zero-based index of the highest-scoring answer."""
    if not scores:
        raise ValueError("no answer scores")
    if metric not in {"sum_logprob", "logprob_per_byte"}:
        raise ValueError(f"unknown metric: {metric}")

    return max(
        range(len(scores)),
        key=lambda i: float(scores[i][metric]),
    )


def evaluate_examples(
    model,
    tok,
    examples,
) -> dict:
    """Evaluate Belebele examples using both answer-scoring rules."""
    correct_sum = 0
    correct_byte = 0
    agreement = 0
    rows = []

    for example in examples:
        prompt = build_prompt(
            example["flores_passage"],
            example["question"],
        )

        answers = [
            example[f"mc_answer{i}"]
            for i in range(1, 5)
        ]

        scores = [
            answer_scores(model, tok, prompt, answer)
            for answer in answers
        ]

        pred_sum = choose_answer(
            scores,
            "sum_logprob",
        )
        pred_byte = choose_answer(
            scores,
            "logprob_per_byte",
        )
        gold = int(example["correct_answer_num"]) - 1

        correct_sum += int(pred_sum == gold)
        correct_byte += int(pred_byte == gold)
        agreement += int(pred_sum == pred_byte)

        rows.append(
            {
                "question_number": example["question_number"],
                "gold": gold + 1,
                "pred_sum_logprob": pred_sum + 1,
                "pred_logprob_per_byte": pred_byte + 1,
                "scores": scores,
            }
        )

    n = len(rows)
    if n == 0:
        raise ValueError("no Belebele examples")

    return {
        "total": n,
        "correct_sum_logprob": correct_sum,
        "accuracy_sum_logprob": correct_sum / n,
        "correct_logprob_per_byte": correct_byte,
        "accuracy_logprob_per_byte": correct_byte / n,
        "scoring_agreement": agreement / n,
        "examples": rows,
    }


def clear_device_cache(dev: torch.device) -> None:
    """Release accelerator cache when supported by the backend."""
    if dev.type == "mps":
        torch.mps.empty_cache()
    elif dev.type == "cuda":
        torch.cuda.empty_cache()


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "models",
        nargs="+",
        help="name=path_or_hub_id",
    )
    ap.add_argument(
        "--lang",
        default="hye_Armn",
    )
    ap.add_argument(
        "--out",
        default=None,
    )
    ap.add_argument(
        "--limit",
        type=int,
        default=None,
        help="evaluate only the first N questions for a smoke test",
    )

    args = ap.parse_args(argv)

    if args.limit is not None and args.limit <= 0:
        ap.error("--limit must be positive")

    # Keep heavyweight imports after argument parsing so --help works without
    # initializing the Transformers model backend.
    from datasets import load_dataset
    from transformers import AutoModelForCausalLM, PreTrainedTokenizerFast

    ds = load_dataset(
        "facebook/belebele",
        args.lang,
        split="test",
    )

    if args.limit is not None:
        ds = ds.select(
            range(min(args.limit, len(ds)))
        )

    dev = common.device()
    results = {}

    for spec in args.models:
        if "=" not in spec:
            ap.error(
                f"model must be name=path_or_hub_id, got {spec!r}"
            )

        name, path = spec.split("=", 1)

        if not name or not path:
            ap.error(
                f"model must be name=path_or_hub_id, got {spec!r}"
            )

        if name in results:
            ap.error(f"duplicate model name: {name}")

        print(
            f"evaluating {name}={path} "
            f"on Belebele {args.lang} ({len(ds)} questions) ...",
            flush=True,
        )

        tok = PreTrainedTokenizerFast.from_pretrained(path)
        model = AutoModelForCausalLM.from_pretrained(
            path,
            dtype=torch.bfloat16,
        ).to(dev)
        model.eval()

        results[name] = evaluate_examples(
            model,
            tok,
            ds,
        )

        summary = {
            key: value
            for key, value in results[name].items()
            if key != "examples"
        }
        print(name, summary, flush=True)

        del model
        clear_device_cache(dev)

    report = {
        "dataset": "facebook/belebele",
        "lang": args.lang,
        "questions": len(ds),
        "chance_accuracy": 0.25,
        "primary_metric": "accuracy_sum_logprob",
        "models": results,
    }

    out = Path(
        args.out
        or f"results/vocab/{args.lang}/belebele.json"
    )
    out.parent.mkdir(
        parents=True,
        exist_ok=True,
    )
    out.write_text(
        json.dumps(
            report,
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    print(f"wrote {out}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())