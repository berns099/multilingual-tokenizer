import pytest

torch = pytest.importorskip("torch")

from tokpremium.train.benchmark_inference import (  # noqa: E402
    benchmark_prefill,
    comparison_ratio,
    summarize_timings,
    timed_call,
)


def test_summarize_timings():
    out = summarize_timings([0.5, 0.1, 0.4, 0.2, 0.3])
    assert out["median_seconds"] == pytest.approx(0.3)
    assert out["p90_seconds"] == pytest.approx(0.5)


def test_summarize_timings_rejects_empty_input():
    with pytest.raises(ValueError, match="no timing"):
        summarize_timings([])


def test_timed_call_runs_warmups_and_measurements():
    calls = []

    def work():
        calls.append(1)

    times = timed_call(
        work,
        torch.device("cpu"),
        warmup=2,
        repeats=3,
    )

    assert len(calls) == 5
    assert len(times) == 3
    assert all(t >= 0 for t in times)


@pytest.mark.parametrize(
    ("warmup", "repeats"),
    [
        (-1, 1),
        (0, 0),
    ],
)
def test_timed_call_rejects_bad_counts(warmup, repeats):
    with pytest.raises(ValueError):
        timed_call(
            lambda: None,
            torch.device("cpu"),
            warmup=warmup,
            repeats=repeats,
        )


class FakeTokenizer:
    def __call__(self, text, add_special_tokens=False):
        assert add_special_tokens is False
        # Deterministic toy encoding: one token per character.
        return {"input_ids": list(range(len(text)))}


class FakeModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.anchor = torch.nn.Parameter(torch.zeros(1))
        self.seen_shapes = []

    def forward(self, input_ids):
        self.seen_shapes.append(tuple(input_ids.shape))
        return input_ids


def test_benchmark_prefill_counts_tokens_and_bytes():
    model = FakeModel()
    tok = FakeTokenizer()

    out = benchmark_prefill(
        model,
        tok,
        ["abc", "é"],
        warmup=0,
        repeats=1,
    )

    assert out["sentences"] == 2
    assert out["input_tokens"] == 4
    assert out["input_bytes"] == 5
    assert out["bytes_per_token"] == pytest.approx(1.25)
    assert out["median_seconds"] >= 0
    assert out["p90_seconds"] >= 0
    assert out["tokens_per_second"] > 0
    assert out["bytes_per_second"] > 0

    # EOT is prepended for the model call but excluded from input_tokens.
    assert model.seen_shapes == [(1, 4), (1, 2)]


def test_benchmark_prefill_rejects_empty_texts():
    with pytest.raises(ValueError, match="no texts"):
        benchmark_prefill(
            FakeModel(),
            FakeTokenizer(),
            [],
            warmup=0,
            repeats=1,
        )


def test_comparison_ratio():
    stock = {
        "input_tokens": 100,
        "median_seconds": 2.0,
        "bytes_per_second": 500.0,
    }
    stage1 = {
        "input_tokens": 30,
        "median_seconds": 0.8,
        "bytes_per_second": 1250.0,
    }

    out = comparison_ratio(stage1, stock)

    assert out["token_ratio"] == pytest.approx(0.3)
    assert out["prefill_time_ratio"] == pytest.approx(0.4)
    assert out["bytes_per_second_ratio"] == pytest.approx(2.5)
