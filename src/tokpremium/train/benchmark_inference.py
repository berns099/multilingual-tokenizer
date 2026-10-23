"""Wall-clock inference benchmark utilities.

The vocabulary experiments estimate inference cost analytically.  This module
measures whether token-count reductions translate into real wall-clock savings.

The primary comparison is prefill on identical raw text.  Generation throughput
is secondary because a fixed number of generated tokens does not represent the
same amount of text across tokenizers.
"""

from __future__ import annotations

import statistics
import time
from collections.abc import Callable

import torch

from . import common


def sync_device(device: torch.device) -> None:
    """Wait for outstanding accelerator work before reading the clock."""
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    elif device.type == "mps":
        torch.mps.synchronize()


def summarize_timings(seconds: list[float]) -> dict[str, float]:
    """Median and p90 latency for a non-empty collection of measurements."""
    if not seconds:
        raise ValueError("no timing measurements")

    ordered = sorted(seconds)
    n = len(ordered)
    # Nearest-rank p90: smallest observed value whose cumulative proportion
    # reaches 90%.  This stays an actual measurement even for small samples.
    p90 = ordered[max(0, (9 * n + 9) // 10 - 1)]

    return {
        "median_seconds": statistics.median(ordered),
        "p90_seconds": p90,
    }


def timed_call(
    fn: Callable[[], object],
    device: torch.device,
    warmup: int = 5,
    repeats: int = 20,
) -> list[float]:
    """Time a callable after warm-up, synchronizing accelerator work."""
    if warmup < 0:
        raise ValueError("warmup must be non-negative")
    if repeats <= 0:
        raise ValueError("repeats must be positive")

    for _ in range(warmup):
        fn()

    sync_device(device)

    times = []
    for _ in range(repeats):
        sync_device(device)
        start = time.perf_counter()
        fn()
        sync_device(device)
        times.append(time.perf_counter() - start)

    return times


def benchmark_prefill(
    model,
    tok,
    texts: list[str],
    *,
    warmup: int = 5,
    repeats: int = 20,
) -> dict[str, float | int]:
    """Benchmark model prefill on identical raw texts.

    Each text is encoded independently and prefixed with Qwen's EOT token,
    matching the evaluation convention in common.bits_per_byte(). Raw UTF-8
    bytes are tokenizer-independent, so bytes/sec is the primary throughput
    metric for comparisons across tokenizers.
    """
    if not texts:
        raise ValueError("no texts to benchmark")

    device = next(model.parameters()).device
    encoded = [
        [common.EOT] + tok(text, add_special_tokens=False)["input_ids"]
        for text in texts
    ]

    input_tokens = sum(len(ids) - 1 for ids in encoded)
    input_bytes = sum(len(text.encode("utf-8")) for text in texts)

    model.eval()

    def run_once() -> None:
        with torch.inference_mode():
            for ids in encoded:
                batch = torch.tensor([ids], device=device)
                model(batch)

    times = timed_call(run_once, device, warmup=warmup, repeats=repeats)
    summary = summarize_timings(times)
    median = summary["median_seconds"]

    return {
        "sentences": len(texts),
        "input_bytes": input_bytes,
        "input_tokens": input_tokens,
        "bytes_per_token": input_bytes / input_tokens,
        **summary,
        "tokens_per_second": input_tokens / median,
        "bytes_per_second": input_bytes / median,
    }


def comparison_ratio(
    result: dict[str, float | int],
    reference: dict[str, float | int],
) -> dict[str, float]:
    """Compare one benchmark result against a reference model."""
    return {
        "token_ratio": float(result["input_tokens"]) / float(reference["input_tokens"]),
        "prefill_time_ratio": (
            float(result["median_seconds"]) / float(reference["median_seconds"])
        ),
        "bytes_per_second_ratio": (
            float(result["bytes_per_second"]) / float(reference["bytes_per_second"])
        ),
    }


def main(argv: list[str] | None = None) -> int:
    """Benchmark prefill latency for one or more model/tokenizer pairs."""
    import argparse
    import json
    import platform
    import sys
    from pathlib import Path

    from ..data.flores import load_sentences

    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "models",
        nargs="+",
        help="name=path_or_hub_id; include a model named 'stock' as the reference",
    )
    ap.add_argument("--lang", default="hye_Armn")
    ap.add_argument("--split", default="dev")
    ap.add_argument("--out", default=None)
    ap.add_argument("--warmup", type=int, default=5)
    ap.add_argument("--repeats", type=int, default=20)
    ap.add_argument(
        "--max-sentences",
        type=int,
        default=100,
        help="number of FLORES sentences to benchmark",
    )
    args = ap.parse_args(argv)

    import transformers
    from transformers import AutoModelForCausalLM, PreTrainedTokenizerFast

    specs = {}
    for spec in args.models:
        if "=" not in spec:
            ap.error(f"model must be name=path_or_hub_id, got {spec!r}")
        name, path = spec.split("=", 1)
        if not name or not path:
            ap.error(f"model must be name=path_or_hub_id, got {spec!r}")
        if name in specs:
            ap.error(f"duplicate model name: {name}")
        specs[name] = path

    if "stock" not in specs:
        ap.error("a model named 'stock' is required as the comparison reference")
    if args.max_sentences <= 0:
        ap.error("--max-sentences must be positive")

    texts = load_sentences(args.lang, split=args.split)[: args.max_sentences]
    dev = common.device()

    results = {}
    for name, path in specs.items():
        print(f"benchmarking {name}={path} on {dev} ...", flush=True)

        tok = PreTrainedTokenizerFast.from_pretrained(path)
        model = AutoModelForCausalLM.from_pretrained(
            path,
            dtype=torch.bfloat16,
        ).to(dev)

        results[name] = benchmark_prefill(
            model,
            tok,
            texts,
            warmup=args.warmup,
            repeats=args.repeats,
        )

        del model
        if dev.type == "mps":
            torch.mps.empty_cache()
        elif dev.type == "cuda":
            torch.cuda.empty_cache()

    stock = results["stock"]
    comparisons = {
        f"{name}_vs_stock": comparison_ratio(result, stock)
        for name, result in results.items()
        if name != "stock"
    }

    report = {
        "benchmark": {
            "lang": args.lang,
            "split": args.split,
            "sentences": len(texts),
            "warmup": args.warmup,
            "repeats": args.repeats,
        },
        "environment": {
            "device": str(dev),
            "platform": platform.platform(),
            "python": sys.version.split()[0],
            "torch": torch.__version__,
            "transformers": transformers.__version__,
            "dtype": "bfloat16",
        },
        "models": results,
        "comparisons": comparisons,
    }

    out = Path(
        args.out
        or f"results/vocab/{args.lang}/inference_benchmark.json"
    )
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(
        json.dumps(report, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    print(json.dumps(report, indent=2, ensure_ascii=False))
    print(f"wrote {out}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
