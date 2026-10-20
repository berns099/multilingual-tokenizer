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
