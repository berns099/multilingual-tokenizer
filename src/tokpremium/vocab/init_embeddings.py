"""Mean-init the new embedding rows and save a checkpoint. Phase C, no training.

    E_new[t] = mean(E_old[t1], ..., E_old[tN])

where (t1..tN) is the token's decomposition under the ORIGINAL tokenizer
(decompose.py). Mean, not sum, so row norms stay comparable; N=1 is a copy.
Old rows are untouched. If the model ties input and output embeddings, the one
matrix is the lm-head too; if it does not, the same rule is applied to both.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch


def mean_init_rows(weight: torch.Tensor, decomp: dict[int, list[int]], old_rows: int) -> None:
    """Overwrite rows `decomp.keys()` of `weight` in place with source means.

    `old_rows` guards the invariant: sources must all be original ids. Computed
    in float32 and cast back, since bf16 accumulation is lossy for long means.
    """
    with torch.no_grad():
        for new_id, srcs in decomp.items():
            if not srcs or max(srcs) >= old_rows or new_id < old_rows:
                raise ValueError(f"bad decomposition for id {new_id}: {srcs}")
            rows = weight[torch.tensor(srcs)].float()
            weight[new_id] = rows.mean(dim=0).to(weight.dtype)


def load_decomposition(
    merges_json: str | Path, n: int
) -> tuple[dict[int, list[int]], list[tuple[str, str]]]:
    """First `n` merges from continue_bpe's merges.json: {new id: source ids}, merges.

    Merges are greedy, so the first n entries are exactly the n-token extension.
    """
    entries = json.loads(Path(merges_json).read_text(encoding="utf-8"))["new_tokens"][:n]
    if len(entries) < n:
        raise ValueError(f"{merges_json} has only {len(entries)} merges, asked for {n}")
    return (
        {e["id"]: e["source_ids"] for e in entries},
        [tuple(e["merge"]) for e in entries],
    )


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", default="Qwen/Qwen3-0.6B")
    ap.add_argument("--lang", required=True, help="e.g. hye_Armn")
    ap.add_argument("--pretokenizer", default="shipped", help="as in configs/vocab.yaml")
    ap.add_argument("--n", type=int, required=True, help="number of new tokens (a sweep size)")
    ap.add_argument("--results-dir", default="results/vocab")
    ap.add_argument("--out", required=True, help="checkpoint dir (gitignored)")
    args = ap.parse_args(argv)

    from transformers import AutoModelForCausalLM

    from . import continue_bpe

    merges_json = continue_bpe.merges_path(args.results_dir, args.lang, args.pretokenizer)
    decomp, merges = load_decomposition(merges_json, args.n)
    out = Path(args.out)
    continue_bpe.build_tokenizer(
        continue_bpe.load_base_spec(args.model), args.pretokenizer, args.lang, merges, out
    )
    from transformers import PreTrainedTokenizerFast

    tok = PreTrainedTokenizerFast.from_pretrained(out)

    model = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype=torch.bfloat16)
    old_rows = model.get_input_embeddings().weight.shape[0]
    # Ids in the stock tokenizer's range (incl. padding rows) are original.
    first_new = min(decomp)
    needed = max(max(decomp) + 1, len(tok))
    if needed > old_rows:
        model.resize_token_embeddings(needed)
    inp = model.get_input_embeddings().weight
    # Sources are always <= the stock tokenizer's top id; rows below first_new are "old".
    mean_init_rows(inp, decomp, old_rows=first_new)
    tied = model.config.tie_word_embeddings
    if not tied:
        mean_init_rows(model.get_output_embeddings().weight, decomp, old_rows=first_new)

    model.save_pretrained(out)
    tok.save_pretrained(out)
    print(f"saved {out}: rows {old_rows} -> {inp.shape[0]}, tied={tied}, "
          f"new={len(decomp)}, hidden={inp.shape[1]}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
