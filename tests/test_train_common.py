import pytest

torch = pytest.importorskip("torch")

from tokpremium.train.common import freeze_old_rows  # noqa: E402


def test_old_rows_stay_bit_identical_under_adamw():
    emb = torch.nn.Parameter(torch.randn(8, 4))
    before = emb.detach().clone()
    freeze_old_rows(emb, first_new=5)
    opt = torch.optim.AdamW([emb], lr=0.1, weight_decay=0.0)
    for _ in range(5):
        (emb * torch.randn(8, 4)).sum().backward()
        opt.step()
        opt.zero_grad()
    assert torch.equal(emb[:5].detach(), before[:5])
    assert not torch.equal(emb[5:].detach(), before[5:])
