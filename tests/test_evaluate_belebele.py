import pytest

torch = pytest.importorskip("torch")

from tokpremium.train.evaluate_belebele import (  # noqa: E402
    answer_scores,
    build_prompt,
    choose_answer,
)


def test_build_prompt():
    out = build_prompt("A passage.", "A question?")
    assert out == "Passage: A passage.\nQuestion: A question?\nAnswer:"


def test_choose_answer_sum_logprob():
    scores = [
        {"sum_logprob": -5.0, "logprob_per_byte": -0.5},
        {"sum_logprob": -2.0, "logprob_per_byte": -0.8},
        {"sum_logprob": -4.0, "logprob_per_byte": -0.4},
    ]
    assert choose_answer(scores, "sum_logprob") == 1


def test_choose_answer_byte_normalized():
    scores = [
        {"sum_logprob": -5.0, "logprob_per_byte": -0.5},
        {"sum_logprob": -2.0, "logprob_per_byte": -0.8},
        {"sum_logprob": -4.0, "logprob_per_byte": -0.4},
    ]
    assert choose_answer(scores, "logprob_per_byte") == 2


def test_choose_answer_rejects_empty_scores():
    with pytest.raises(ValueError, match="no answer"):
        choose_answer([], "sum_logprob")


def test_choose_answer_rejects_unknown_metric():
    with pytest.raises(ValueError, match="unknown metric"):
        choose_answer(
            [{"sum_logprob": -1.0}],
            "something_else",
        )


class TinyTokenizer:
    def __call__(self, text, add_special_tokens=False):
        assert add_special_tokens is False

        # Prompt and answer are intentionally tokenized separately.
        mapping = {
            "P": [10],
            " A": [20],
        }
        return {"input_ids": mapping[text]}


class TinyOutput:
    def __init__(self, logits):
        self.logits = logits


class TinyModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.anchor = torch.nn.Parameter(torch.zeros(1))

    def forward(self, ids):
        # Sequence is [EOT, prompt=10, answer=20].
        # Logits at position 1 predict answer token 20.
        vocab_size = 32
        logits = torch.zeros(
            (1, ids.shape[1], vocab_size),
            device=ids.device,
        )
        logits[0, 1, 20] = 4.0
        return TinyOutput(logits)


def test_answer_scores_only_scores_answer_tokens():
    out = answer_scores(
        TinyModel(),
        TinyTokenizer(),
        "P",
        "A",
    )

    assert out["answer_tokens"] == 1
    assert out["answer_bytes"] == 1

    # One answer byte means byte-normalized and total scores are identical.
    assert out["logprob_per_byte"] == pytest.approx(
        out["sum_logprob"]
    )

    # Token 20 has the dominant logit at its prediction position.
    assert out["sum_logprob"] > -1.0


class BoundaryTokenizer:
    def __call__(self, text, add_special_tokens=False):
        assert add_special_tokens is False

        mapping = {
            "prompt": [5, 6],
            " answer": [7, 8],
        }
        return {"input_ids": mapping[text]}


class BoundaryModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.anchor = torch.nn.Parameter(torch.zeros(1))

    def forward(self, ids):
        # Sequence:
        # [EOT, prompt_1=5, prompt_2=6, answer_1=7, answer_2=8]
        #
        # Position 2 predicts answer token 7.
        # Position 3 predicts answer token 8.
        vocab_size = 16
        logits = torch.zeros(
            (1, ids.shape[1], vocab_size),
            device=ids.device,
        )
        logits[0, 2, 7] = 5.0
        logits[0, 3, 8] = 5.0
        return TinyOutput(logits)


def test_answer_scores_handles_multi_token_answer_boundary():
    out = answer_scores(
        BoundaryModel(),
        BoundaryTokenizer(),
        "prompt",
        "answer",
    )

    assert out["answer_tokens"] == 2
    assert out["answer_bytes"] == len(b"answer")

    # Log probabilities are negative, but both correct answer tokens receive
    # dominant logits, so the total should remain close to zero.
    assert out["sum_logprob"] < 0
    assert out["sum_logprob"] > -1.0