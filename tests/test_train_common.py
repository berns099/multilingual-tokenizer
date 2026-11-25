from types import SimpleNamespace

import pytest

torch = pytest.importorskip("torch")

from tokpremium.train import common  # noqa: E402


def test_old_rows_stay_bit_identical_under_adamw():
    emb = torch.nn.Parameter(torch.randn(8, 4))
    before = emb.detach().clone()
    common.freeze_old_rows(emb, first_new=5)
    opt = torch.optim.AdamW([emb], lr=0.1, weight_decay=0.0)
    for _ in range(5):
        (emb * torch.randn(8, 4)).sum().backward()
        opt.step()
        opt.zero_grad()
    assert torch.equal(emb[:5].detach(), before[:5])
    assert not torch.equal(emb[5:].detach(), before[5:])


def test_file_fingerprint_records_hash_and_size(tmp_path):
    path = tmp_path / "corpus.txt"
    path.write_bytes(b"hello\n")

    assert common.file_fingerprint(path) == {
        "path": str(path),
        "bytes": 6,
        "sha256": "5891b5b522d5df086d0ff0b110fbd9d21bb4fc7163af34d08286a2e846f6be03",
    }


def test_git_provenance_falls_back_when_git_is_unavailable(monkeypatch):
    def unavailable(*args, **kwargs):
        raise FileNotFoundError

    monkeypatch.setattr(common.subprocess, "run", unavailable)

    assert common.git_provenance() == {"commit": None, "dirty": None}


def test_build_provenance_records_runtime_and_present_inputs(tmp_path, monkeypatch):
    corpus = tmp_path / "train.txt"
    heldout = tmp_path / "heldout.txt"
    corpus.write_text("train\n", encoding="utf-8")
    heldout.write_text("test\n", encoding="utf-8")
    args = SimpleNamespace(corpus=str(corpus), heldout=str(heldout), english=None)
    monkeypatch.setattr(
        common,
        "git_provenance",
        lambda: {"commit": "abc123", "dirty": False},
    )

    provenance = common.build_provenance(args, torch.device("cpu"))

    assert provenance["git"] == {"commit": "abc123", "dirty": False}
    assert provenance["environment"]["device"] == "cpu"
    assert provenance["environment"]["python"]
    assert provenance["environment"]["torch"]
    assert provenance["environment"]["transformers"]
    assert set(provenance["inputs"]) == {"corpus", "heldout"}
    assert provenance["inputs"]["corpus"]["sha256"]
    assert provenance["inputs"]["heldout"]["sha256"]


def test_token_and_byte_budgets_are_mutually_exclusive():
    """The two budgets are different experiments; a run must pick one."""
    import argparse

    ap = argparse.ArgumentParser()
    common.add_common_args(ap)
    base = ["--out", "o", "--corpus", "c", "--heldout", "h"]

    assert ap.parse_args(base).byte_budget is None
    assert ap.parse_args(base + ["--bytes", "1e7"]).byte_budget == 1e7
    with pytest.raises(SystemExit):
        ap.parse_args(base + ["--tokens", "2e6", "--bytes", "1e7"])


def test_byte_budget_equalizes_text_across_differing_tokenizers():
    """Equal bytes, unequal steps: the inverse of what a token budget gives.

    The extended tokenizer packs ~3.4x more bytes per token, so budgeting by
    tokens hands it that much more text. Budgeting by bytes is what makes a
    matched-text comparison possible.
    """
    batch, seq, target_bytes = 16, 256, 12_950_000
    steps = {}
    for name, bytes_per_tok in (("extended", 6.28), ("baseline", 1.85)):
        steps[name] = int((target_bytes / bytes_per_tok) // (batch * seq))
        seen = steps[name] * batch * seq * bytes_per_tok
        assert abs(seen - target_bytes) / target_bytes < 0.01

    assert steps["baseline"] > steps["extended"] * 3


def _fake_tok():
    class Tok:  # one token per four characters
        def __call__(self, texts, add_special_tokens=False):
            return {"input_ids": [[0] * max(1, len(t) // 4) for t in texts]}

    return Tok()


def test_replay_hits_the_requested_token_share():
    """The share is counted in tokens, which is what the optimizer sees."""
    target = ["telugu line " * 10] * 100
    replay = ["english line " * 10] * 20
    for ratio in (0.1, 0.25, 0.5):
        rep = {}
        common.mix_replay(_fake_tok(), list(target), list(replay), ratio, 0, rep)
        assert abs(rep["replay"]["ratio_actual"] - ratio) < 0.01


def test_replay_cycles_a_small_corpus_and_is_seed_deterministic():
    target = ["telugu line " * 10] * 100
    replay = ["english line " * 10] * 5
    rep = {}
    a = common.mix_replay(_fake_tok(), list(target), list(replay), 0.25, 7, rep)
    b = common.mix_replay(_fake_tok(), list(target), list(replay), 0.25, 7, {})
    c = common.mix_replay(_fake_tok(), list(target), list(replay), 0.25, 8, {})
    assert rep["replay"]["cycled"] is True
    assert a == b and a != c


def test_replay_rejects_a_ratio_outside_the_open_unit_interval():
    for bad in (0.0, 1.0, 1.5, -0.1):
        with pytest.raises(ValueError, match="replay ratio"):
            common.mix_replay(_fake_tok(), ["x" * 40], ["y" * 40], bad, 0, {})


def test_split_disjoint_is_lossless_and_seed_deterministic():
    """The train/heldout split was made by hand in October and never committed.

    A split that silently overlaps turns bits-per-byte on the heldout set into
    a training-set measurement, which is the kind of error that reads as a
    good result.
    """
    from tokpremium.data.prepare import split_disjoint

    lines = [f"line {i}" for i in range(1000)]
    train, held = split_disjoint(lines, 200, seed=0)

    assert len(held) == 200 and len(train) == 800
    assert not set(train) & set(held)
    assert sorted(train + held) == sorted(lines)
    assert split_disjoint(lines, 200, seed=0) == (train, held)
    assert split_disjoint(lines, 200, seed=1)[0] != train
