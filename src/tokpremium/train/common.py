"""Shared pieces for stage-1/2 and the matched baseline: data, eval, device."""

from __future__ import annotations

import hashlib
import math
import platform
import random
import subprocess
from argparse import Namespace
from datetime import datetime, timezone
from importlib.metadata import version
from pathlib import Path

import torch

from ..survey.metrics import normalize

EOT = 151643  # <|endoftext|>, Qwen's document separator; used as context-start


def device() -> torch.device:
    return torch.device("mps" if torch.backends.mps.is_available() else "cpu")


def read_lines(path: str | Path) -> list[str]:
    return [normalize(x) for x in Path(path).read_text(encoding="utf-8").splitlines() if x.strip()]


def file_fingerprint(path: str | Path) -> dict[str, str | int]:
    """Return a stable identity for an input file without loading it into memory."""
    path = Path(path)
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return {
        "path": str(path),
        "bytes": path.stat().st_size,
        "sha256": digest.hexdigest(),
    }


def git_provenance() -> dict[str, str | bool | None]:
    """Return the checked-out revision and whether tracked or untracked files differ."""
    cwd = Path(__file__).resolve().parent
    try:
        revision = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=cwd,
            check=True,
            capture_output=True,
            text=True,
            timeout=5,
        ).stdout.strip()
        status = subprocess.run(
            ["git", "status", "--porcelain=v1", "--untracked-files=all"],
            cwd=cwd,
            check=True,
            capture_output=True,
            text=True,
            timeout=5,
        ).stdout
    except (OSError, subprocess.SubprocessError):
        return {"commit": None, "dirty": None}
    return {"commit": revision, "dirty": bool(status)}


def build_provenance(args: Namespace, dev: torch.device) -> dict:
    """Describe the code, runtime, and exact input files used by a training run."""
    inputs = {
        name: file_fingerprint(path)
        for name in ("corpus", "heldout", "english", "replay")
        if (path := getattr(args, name, None))
    }
    return {
        "started_at_utc": datetime.now(timezone.utc).isoformat(),
        "git": git_provenance(),
        "environment": {
            "python": platform.python_version(),
            "torch": str(torch.__version__),
            "transformers": version("transformers"),
            "device": str(dev),
        },
        "inputs": inputs,
    }


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
        nll += torch.nn.functional.cross_entropy(logits[0], ids[0, 1:], reduction="sum").item()
        n_bytes += len(t.encode("utf-8"))
    return nll / math.log(2) / n_bytes


def freeze_old_rows(weight: torch.nn.Parameter, first_new: int) -> None:
    """Zero the gradient of rows < first_new, so only new rows can move.

    With AdamW at weight_decay=0, a row whose gradient is always zero has zero
    moments and a zero update, so old rows stay bit-identical. Verified by a
    test and again at the end of every run.
    """
    weight.register_hook(lambda g: torch.cat([torch.zeros_like(g[:first_new]), g[first_new:]]))


def check_replay_not_probe(replay: str | Path, english: str | Path | None) -> None:
    """Refuse to train on the file held out for measuring English retention.

    The probe and the replay come from one stream cut in half, so the two paths
    are easy to swap by accident, and the failure is silent: the English number
    would simply look good, because the model had been trained on the very text
    the number is computed over. Compared after resolving, so a relative path
    and an absolute one naming the same file are still caught.
    """
    if english is None:
        return
    if Path(replay).resolve() == Path(english).resolve():
        raise ValueError(
            "--replay and --english are the same file. The probe is held out "
            "for measuring English retention; training on it makes that "
            "measurement meaningless."
        )


def mix_replay(
    tok,
    target: list[str],
    replay: list[str],
    ratio: float,
    seed: int,
    rep: dict,
) -> list[str]:
    """Mix English replay lines into the target corpus at a token share.

    The share is counted in TOKENS, not lines or bytes, because tokens are what
    the optimizer consumes and what the budget is denominated in. Counting by
    lines would make the share depend on line length, which differs sharply
    between a web-scraped Armenian corpus and FineWeb English.

    `ratio` is the replay share of the mixture, so 0.25 means one replay token
    for every three target tokens. A short replay file is cycled rather than
    truncated, because the alternative is silently training on less English
    than the ratio claims. Cycling is recorded so a run that leaned on a thin
    replay corpus is visible in the report instead of inferred.

    The mixture is shuffled under `seed`, which both interleaves the two
    languages and keeps the result reproducible: resuming a run re-derives a
    byte-identical mixture, so the permutation over packed rows still lines up.

    Returns the shuffled mixture and writes a `replay` block into `rep`.
    """
    if not 0 < ratio < 1:
        raise ValueError(f"replay ratio must be in (0, 1), got {ratio}")
    if not replay:
        raise ValueError("replay corpus is empty")

    n_target = sum(len(x) for x in tok(target, add_special_tokens=False)["input_ids"])
    counts = [len(x) for x in tok(replay, add_special_tokens=False)["input_ids"]]

    # ratio = want / (n_target + want), solved for want.
    want = ratio / (1 - ratio) * n_target

    chosen: list[str] = []
    taken = 0
    cycled = False
    i = 0
    while taken < want:
        if i >= len(replay):
            i, cycled = 0, True
        chosen.append(replay[i])
        taken += counts[i]
        i += 1

    mixture = target + chosen
    random.Random(seed).shuffle(mixture)

    rep["replay"] = {
        "lines": len(chosen),
        "tokens": taken,
        "target_tokens": n_target,
        "ratio_requested": ratio,
        "ratio_actual": taken / (n_target + taken),
        "cycled": cycled,
    }
    return mixture


def load_eval_docs(args) -> tuple[list[str], list[str]]:
    """Heldout and English probe documents, both capped at --eval-docs.

    Capping both matters for two separate reasons.

    Cost: bits_per_byte runs one forward pass per document with no batching, so
    an uncapped probe is not merely slow. prepare.py writes an English probe of
    around 55,000 lines, which costs roughly forty minutes to score, twice per
    arm, across three arms of a run. An uncapped probe consumed a whole job.

    Comparability: the target-language number and the English number are read
    side by side. Measuring one over 100 documents and the other over 55,000
    gives them very different precision while presenting them as a pair.

    --eval-docs is cheap now that it binds, so raising it buys tighter numbers:
    500 documents costs seconds rather than the better part of an hour.
    """
    held = read_lines(args.heldout)[: args.eval_docs]
    eng = read_lines(args.english)[: args.eval_docs] if args.english else []
    return held, eng


def run_training(model, tok, args, params, old_rows: int | None = None) -> dict:
    """Shared loop for every arm, so arms differ only in tokenizer/trainables.

    Same packing, batch, step count, LR schedule and seed => matched compute.
    Reports bytes_seen (text exposure), which is what differs between arms.
    """
    import json
    import time

    dev = next(model.parameters()).device
    torch.manual_seed(args.seed)
    held, eng = load_eval_docs(args)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    ckpt_path = out / "ckpt.pt"
    resume = (
        torch.load(ckpt_path, map_location="cpu", weights_only=True) if ckpt_path.exists() else None
    )
    if resume:
        rep = resume["rep"]
        print(f"resuming from step {resume['step']}", flush=True)
    else:
        rep = {
            "args": vars(args),
            "provenance": build_provenance(args, dev),
            "tokens_seen": 0,
        }
        rep["bpb_tel_before"] = bits_per_byte(model, tok, held)
        if eng:
            rep["bpb_eng_before"] = bits_per_byte(model, tok, eng)
    print(rep, flush=True)

    texts = read_lines(args.corpus)
    rep["target_bytes"] = sum(len(t.encode()) for t in texts)
    if getattr(args, "replay", None):
        check_replay_not_probe(args.replay, args.english)
        texts = mix_replay(
            tok, texts, read_lines(args.replay), args.replay_ratio, args.seed, rep
        )
        print(f"replay: {rep['replay']}", flush=True)
    data = pack(tok, texts, args.seq_len)
    tot_bytes = sum(len(t.encode()) for t in texts)
    n_tok_corpus = sum(len(x) for x in tok(texts, add_special_tokens=False)["input_ids"])
    bytes_per_tok = tot_bytes / n_tok_corpus
    # Matched compute and matched text are different experiments and cannot be
    # satisfied at once. An extended tokenizer packs more bytes into every
    # token, so an equal token budget hands it roughly 3.5x the Armenian text
    # that the original-tokenizer arm sees (12.95 MB against 3.72 MB in our
    # October run). Budgeting by bytes equalizes the text and leaves the step
    # counts unequal instead. Which one a run chose is recorded in the report
    # so a result is never read as the other kind of comparison.
    if getattr(args, "byte_budget", None):
        token_budget = args.byte_budget / bytes_per_tok
        rep["budget"] = {"kind": "bytes", "requested_bytes": float(args.byte_budget)}
    else:
        token_budget = args.tokens
        rep["budget"] = {"kind": "tokens", "requested_tokens": float(args.tokens)}

    steps = int(token_budget // (args.batch * args.seq_len))
    rep["budget"].update(
        {
            "steps": steps,
            "bytes_per_token": bytes_per_tok,
            "planned_tokens": steps * args.batch * args.seq_len,
            "planned_bytes": int(steps * args.batch * args.seq_len * bytes_per_tok),
        }
    )
    opt = torch.optim.AdamW(params, lr=args.lr, weight_decay=0.0)
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: min(1.0, (s + 1) / 20) * max(0.05, 1 - s / max(steps, 1))
    )
    snap = params[0][:old_rows].detach().clone() if old_rows else None
    perm = torch.randperm(len(data))
    start = 0
    if resume:
        with torch.no_grad():
            for p, v in zip(params, resume["params"], strict=True):
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
    budget = ap.add_mutually_exclusive_group()
    budget.add_argument(
        "--tokens",
        type=float,
        default=2e6,
        help="token budget: MATCHED COMPUTE. Both arms do the same number of "
        "optimizer steps on the same batch shape.",
    )
    budget.add_argument(
        "--bytes",
        type=float,
        default=None,
        dest="byte_budget",
        help="byte budget: MATCHED TEXT. Converted to steps using this corpus's "
        "own bytes-per-token, so both arms read the same amount of the language.",
    )
    ap.add_argument("--seq-len", type=int, default=256)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument(
        "--replay",
        help="English text mixed into training to limit forgetting. MUST NOT "
        "be the --english probe, which is held out to measure retention; "
        "run_training refuses if the two resolve to the same file. Both arms "
        "of a comparison must replay identically, or the English result "
        "measures the replay instead of the tokenizer.",
    )
    ap.add_argument(
        "--replay-ratio",
        type=float,
        default=0.25,
        help="replay share of the mixture, counted in tokens. configs/vocab.yaml "
        "names target-plus-English for stage 2 but fixes no number, so this is "
        "a starting point rather than a finding.",
    )
    ap.add_argument("--eval-docs", type=int, default=100)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument(
        "--save-every",
        type=int,
        default=50,
        help="steps between resumable checkpoints; 0=off",
    )
