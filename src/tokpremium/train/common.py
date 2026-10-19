"""Shared pieces for stage-1/2 and the matched baseline: data, eval, device."""

from __future__ import annotations

import math
from pathlib import Path

import torch

from ..survey.metrics import normalize

EOT = 151643  # <|endoftext|>, Qwen's document separator; used as context-start


def device() -> torch.device:
    return torch.device("mps" if torch.backends.mps.is_available() else "cpu")


def read_lines(path: str | Path) -> list[str]:
    return [normalize(x) for x in Path(path).read_text(encoding="utf-8").splitlines() if x.strip()]


def pack(tok, texts: list[str], seq_len: int) -> torch.Tensor:
    """Concatenate docs with EOT between, cut into [N, seq_len] rows.

    Returns exactly the tokens a run will see, so budget = N * seq_len and
    "data exposure" is a number we can log for the matched-compute comparison.
    """
    stream: list[int] = []
    for ids in tok(texts, add_special_tokens=False)["input_ids"]:
        stream += ids + [EOT]
    n = len(stream) // seq_len
    return torch.tensor(stream[: n * seq_len]).view(n, seq_len)


@torch.no_grad()
def bits_per_byte(model, tok, texts: list[str], max_chars: int = 600) -> float:
    """Tokenizer-independent LM quality: total NLL in bits / UTF-8 bytes scored.

    Perplexity is not comparable across tokenizers (different denominators);
    this is. Each doc is truncated by characters, so every tokenizer scores the
    same text, and is prefixed with EOT so the first real token is scored too.
    """
    dev = next(model.parameters()).device
    model.eval()
    nll, n_bytes = 0.0, 0
    for t in texts:
        t = t[:max_chars]
        ids = torch.tensor([[EOT] + tok(t, add_special_tokens=False)["input_ids"]], device=dev)
        logits = model(ids).logits[:, :-1].float()
        nll += torch.nn.functional.cross_entropy(
            logits[0], ids[0, 1:], reduction="sum"
        ).item()
        n_bytes += len(t.encode("utf-8"))
    return nll / math.log(2) / n_bytes


def freeze_old_rows(weight: torch.nn.Parameter, first_new: int) -> None:
    """Zero the gradient of rows < first_new, so only new rows can move.

    With AdamW at weight_decay=0, a row whose gradient is always zero has zero
    moments and a zero update, so old rows stay bit-identical. Verified by a
    test and again at the end of every run.
    """
    weight.register_hook(lambda g: torch.cat([torch.zeros_like(g[:first_new]), g[first_new:]]))


def run_training(model, tok, args, params, old_rows: int | None = None) -> dict:
    """Shared loop for every arm, so arms differ only in tokenizer/trainables.

    Same packing, batch, step count, LR schedule and seed => matched compute.
    Reports bytes_seen (text exposure), which is what differs between arms.
    """
    import json
    import time

    dev = next(model.parameters()).device
    torch.manual_seed(args.seed)
    held = read_lines(args.heldout)[: args.eval_docs]
    eng = read_lines(args.english) if args.english else []
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    ckpt_path = out / "ckpt.pt"
    resume = torch.load(ckpt_path, map_location="cpu") if ckpt_path.exists() else None
    if resume:
        rep = resume["rep"]
        print(f"resuming from step {resume['step']}", flush=True)
    else:
        rep = {"args": vars(args), "tokens_seen": 0}
        rep["bpb_tel_before"] = bits_per_byte(model, tok, held)
        if eng:
            rep["bpb_eng_before"] = bits_per_byte(model, tok, eng)
    print(rep, flush=True)

    texts = read_lines(args.corpus)
    data = pack(tok, texts, args.seq_len)
    tot_bytes = sum(len(t.encode()) for t in texts)
    n_tok_corpus = sum(len(x) for x in tok(texts, add_special_tokens=False)["input_ids"])
    bytes_per_tok = tot_bytes / n_tok_corpus
    steps = int(args.tokens // (args.batch * args.seq_len))
    opt = torch.optim.AdamW(params, lr=args.lr, weight_decay=0.0)
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: min(1.0, (s + 1) / 20) * max(0.05, 1 - s / max(steps, 1))
    )
    snap = params[0][:old_rows].detach().clone() if old_rows else None
    perm = torch.randperm(len(data))
    start = 0
    if resume:
        with torch.no_grad():
            for p, v in zip(params, resume["params"]):
                p.copy_(v.to(p.device))
        opt.load_state_dict(resume["opt"])
        sched.load_state_dict(resume["sched"])
        start = resume["step"]
        # torch.load(map_location="cpu") put the snapshot on the CPU; the live
        # parameters are on the training device, and torch.equal needs both on one.
        snap = resume["snap"].to(dev) if resume["snap"] is not None else None
        del resume
    model.train()
    t0 = time.time()
    seen0 = rep["tokens_seen"]
    for step in range(start, steps):
        i = (step * args.batch) % max(len(data) - args.batch, 1)
        batch = data[perm[i : i + args.batch]].to(dev)
        loss = model(batch, labels=batch).loss
        loss.backward()
        opt.step()
        sched.step()
        opt.zero_grad(set_to_none=True)
        rep["tokens_seen"] += batch.numel()
        if step % 25 == 0 or step == steps - 1:
            rate = (rep["tokens_seen"] - seen0) / (time.time() - t0)
            print(f"step {step + 1}/{steps} loss {loss.item():.3f} {rate:,.0f} tok/s", flush=True)
        if args.save_every and (step + 1) % args.save_every == 0 and step + 1 < steps:
            tmp = ckpt_path.with_suffix(".tmp")
            torch.save(
                {
                    "step": step + 1,
                    "rep": rep,
                    "params": [p.detach().cpu() for p in params],
                    "opt": opt.state_dict(),
                    "sched": sched.state_dict(),
                    "snap": snap,
                },
                tmp,
            )
            tmp.replace(ckpt_path)  # atomic: a crash mid-write never corrupts ckpt.pt

    rep["bytes_seen"] = int(rep["tokens_seen"] * bytes_per_tok)
    rep["train_bytes_per_token"] = bytes_per_tok
    rep["bpb_tel_after"] = bits_per_byte(model, tok, held)
    if eng:
        rep["bpb_eng_after"] = bits_per_byte(model, tok, eng)
    if snap is not None:
        rep["old_rows_unchanged"] = bool(torch.equal(params[0][:old_rows].detach(), snap))
    rep["seconds"] = time.time() - t0
    model.to(torch.bfloat16).save_pretrained(out)
    tok.save_pretrained(out)
    (out / "report.json").write_text(json.dumps(rep, indent=1))
    ckpt_path.unlink(missing_ok=True)  # run finished; the ~2 GB resume state is not needed
    print(json.dumps({k: v for k, v in rep.items() if k != "args"}, indent=1))
    return rep


def add_common_args(ap) -> None:
    ap.add_argument("--out", required=True)
    ap.add_argument("--corpus", required=True)
    ap.add_argument("--heldout", required=True)
    ap.add_argument("--english")
    ap.add_argument("--tokens", type=float, default=2e6, help="training token budget (compute)")
    ap.add_argument("--seq-len", type=int, default=256)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--eval-docs", type=int, default=100)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--save-every", type=int, default=50, help="steps between resumable checkpoints; 0=off")
