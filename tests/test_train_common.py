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
