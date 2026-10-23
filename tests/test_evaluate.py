import pytest

torch = pytest.importorskip("torch")

from tokpremium.train.evaluate import (  # noqa: E402
    repetition_rate,
    summarize_repetition,
)


def test_repetition_rate_no_repeated_fourgrams():
    ids = [1, 2, 3, 4, 5, 6, 7]
    assert repetition_rate(ids) == pytest.approx(0.0)


def test_repetition_rate_all_same():
    ids = [1, 1, 1, 1, 1, 1, 1]
    # Four 4-grams, but only one unique.
    assert repetition_rate(ids) == pytest.approx(0.75)


def test_repetition_rate_short_sequence():
    assert repetition_rate([1, 2, 3]) == 0.0


def test_summarize_repetition():
    out = summarize_repetition([0.1, 0.2, 0.3, 0.4, 0.9])

    assert out["mean"] == pytest.approx(0.38)
    assert out["median"] == pytest.approx(0.3)
    assert out["p90"] == pytest.approx(0.9)
    assert out["max"] == pytest.approx(0.9)


def test_summarize_repetition_rejects_empty_input():
    with pytest.raises(ValueError, match="no repetition"):
        summarize_repetition([])
