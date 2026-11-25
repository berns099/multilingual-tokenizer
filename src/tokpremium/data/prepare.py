"""Build the training, heldout, probe and replay files, reproducibly.

    python -m tokpremium.data.prepare --lang hye_Armn --out data/train

Writes four files under --out:

    <lang>_train.txt      target-language training text
    <lang>_heldout.txt    target-language heldout, for bits/byte during training
    eng_probe.txt         English retention probe: MEASURED, never trained on
    eng_replay.txt        English replay: TRAINED ON, never measured

Until now this split was made by hand and never committed, which is half of
why the October runs could not be reproduced. Everything here is deterministic
given --seed, so two people running this command get byte-identical files, and
the hashes recorded in each training run's provenance block will agree.

Three separations are enforced rather than assumed, because each one silently
invalidates a different number if it breaks:

  train vs heldout   a shared line makes bits/byte a training-set measurement
  probe vs replay    training on the probe makes English retention meaningless
  train vs FLORES    FLORES dev is the evaluation set for every reported result

The FLORES check is the one most easily missed: the target corpus comes from
FineWeb-2, which is web text, and FLORES passages are drawn from Wikipedia and
Wikinews, so overlap is possible rather than hypothetical. Lines are compared
after the same NFC normalization the rest of the pipeline applies, so a
difference in Unicode form cannot hide a real duplicate.
"""

from __future__ import annotations

import argparse
import hashlib
import random
import sys
from pathlib import Path

import yaml

from ..survey.metrics import normalize
from ..vocab import corpus
from . import flores

# English source for the probe and the replay. The project config names a
# source for every target language but none for English, because English was
# only ever a measurement until stage 2 introduced replay.
ENGLISH_SOURCE = {"repo": "HuggingFaceFW/fineweb", "config": "sample-10BT"}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write(path: Path, lines: list[str]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def split_disjoint(lines: list[str], n_second: int, seed: int) -> tuple[list[str], list[str]]:
    """Shuffle under `seed`, then cut. Returns (first, second)."""
    shuffled = list(lines)
    random.Random(seed).shuffle(shuffled)
    return shuffled[n_second:], shuffled[:n_second]


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--lang", default="hye_Armn")
    ap.add_argument("--config", default="configs/vocab.yaml")
    ap.add_argument("--out", default="data/train")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--heldout-lines", type=int, default=2000)
    ap.add_argument(
        "--english-chars",
        type=int,
        default=20_000_000,
        help="English to stream. Split in half: probe and replay, disjoint.",
    )
    args = ap.parse_args(argv)

    cfg = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    out = Path(args.out)

    # ---- target language -------------------------------------------------
    source = cfg["corpus"]["sources"][args.lang]
    cache = cfg["corpus"]["cache_dir"]
    target_path = corpus.fetch(args.lang, source, cfg["corpus"]["max_chars"], cache)
    target = [ln for ln in corpus.read_lines(target_path) if ln.strip()]

    # ---- remove anything that appears in the evaluation set ---------------
    # Both sides are already NFC-normalized, by corpus.fetch and by flores.
    evaluation = {normalize(s.strip()) for s in flores.load_sentences(args.lang, split="dev")}
    evaluation |= {normalize(s.strip()) for s in flores.load_sentences(args.lang, split="devtest")}
    kept = [ln for ln in target if normalize(ln.strip()) not in evaluation]
    dropped = len(target) - len(kept)

    if len(kept) <= args.heldout_lines:
        ap.error(f"only {len(kept)} usable lines; need more than --heldout-lines")

    train, heldout = split_disjoint(kept, args.heldout_lines, args.seed)

    # ---- English: probe and replay, from one stream, cut in half ----------
    eng_path = corpus.fetch("eng_Latn_train", ENGLISH_SOURCE, args.english_chars, cache)
    english = [ln for ln in corpus.read_lines(eng_path) if ln.strip()]
    replay, probe = split_disjoint(english, len(english) // 2, args.seed)

    files = {
        f"{args.lang}_train.txt": train,
        f"{args.lang}_heldout.txt": heldout,
        "eng_probe.txt": probe,
        "eng_replay.txt": replay,
    }
    written = {name: write(out / name, lines) for name, lines in files.items()}

    # ---- assert the separations actually hold ----------------------------
    assert not (set(train) & set(heldout)), "train and heldout overlap"
    assert not (set(probe) & set(replay)), "probe and replay overlap"
    assert not ({normalize(x.strip()) for x in train} & evaluation), "train contains FLORES"

    print(f"dropped {dropped} line(s) of {args.lang} found in FLORES dev/devtest")
    for name, path in written.items():
        n = len(files[name])
        size = path.stat().st_size
        print(f"  {name:24s} {n:>8,} lines  {size / 1e6:>7.2f} MB  sha256={sha256(path)[:16]}")
    print("\nrecord these hashes: each training run's provenance block stores the same ones.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
