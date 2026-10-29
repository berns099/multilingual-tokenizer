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
