"""Resume must reproduce an uninterrupted run, on the device training uses.

Regression: after resuming, the old-rows snapshot loaded on the CPU while the
parameters lived on the GPU, so the final torch.equal crashed -- after all the
training had finished. A CPU-only test cannot see that, hence the mps case.
"""

import argparse

import pytest

torch = pytest.importorskip("torch")
tf = pytest.importorskip("transformers")

from tokpremium.train import common  # noqa: E402

DEVICES = ["cpu"] + (["mps"] if torch.backends.mps.is_available() else [])


class Boom(Exception):
    pass


def _setup(tmp_path):
    try:
        tok = tf.PreTrainedTokenizerFast.from_pretrained("Qwen/Qwen3-0.6B")
        cfg = tf.AutoConfig.from_pretrained("Qwen/Qwen3-0.6B")
    except Exception:
        pytest.skip("Qwen3-0.6B tokenizer/config not cached")
    (tmp_path / "t.txt").write_text("\n".join(["తెలుగు భాష చాలా అందమైనది " * 8] * 40))
    cfg.hidden_size, cfg.intermediate_size, cfg.num_hidden_layers = 16, 32, 1
    cfg.num_attention_heads = cfg.num_key_value_heads = 2
    cfg.head_dim = 8
    cfg.layer_types = cfg.layer_types[:1]
    return tok, cfg


def _args(tmp_path, out):
    return argparse.Namespace(
        out=str(out), corpus=str(tmp_path / "t.txt"), heldout=str(tmp_path / "t.txt"),
        english=None, tokens=8 * 2 * 16, seq_len=16, batch=2, lr=1e-2, eval_docs=3,
        seed=0, save_every=2,
    )


@pytest.mark.parametrize("dev", DEVICES)
def test_resume_matches_uninterrupted(tmp_path, dev):
    tok, cfg = _setup(tmp_path)
    first_new = 151000

    def make():
        torch.manual_seed(1)
        m = tf.AutoModelForCausalLM.from_config(cfg).to(dev)
        for p in m.parameters():
            p.requires_grad_(False)
        w = m.get_input_embeddings().weight
        w.requires_grad_(True)
        common.freeze_old_rows(w, first_new)
        return m, w

    m, w = make()
    common.run_training(m, tok, _args(tmp_path, tmp_path / "a"), [w], old_rows=first_new)
    ref = w.detach().cpu().clone()

    m, w = make()
    calls = [0]

    def boom(*_):
        calls[0] += 1
        # eval_docs forwards come first (pre-training eval); then one per training
        # step. Crash on the 5th training step, after the step-4 checkpoint.
        if calls[0] == 3 + 5:
            raise Boom

    m.register_forward_pre_hook(boom)
    with pytest.raises(Boom):
        common.run_training(m, tok, _args(tmp_path, tmp_path / "b"), [w], old_rows=first_new)
    assert (tmp_path / "b" / "ckpt.pt").exists(), "crash fired before any checkpoint"

    m, w = make()
    rep = common.run_training(m, tok, _args(tmp_path, tmp_path / "b"), [w], old_rows=first_new)
    assert rep["old_rows_unchanged"] is True
    assert torch.allclose(w.detach().cpu().float(), ref.float(), atol=1e-5)
    assert not (tmp_path / "b" / "ckpt.pt").exists()
