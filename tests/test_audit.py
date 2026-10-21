"""Tests for the unreachable-token audit (vocab/audit.py), offline.

Uses the tiny Qwen-shaped tokenizer from test_continue_bpe and plants tokens
whose reachability is known by construction.
"""

from test_continue_bpe import make_base
from tokpremium.vocab.audit import classify, script_of
from tokpremium.vocab.continue_bpe import extend, piece_bytes


def byte_pieces(spec, text):
    """The single-byte vocab pieces spelling `text`."""
    vocab = spec["model"]["vocab"]
    return [next(t for t in vocab if piece_bytes(t) == bytes([b])) for b in text.encode()]


def plant(spec, text):
    """Add `text` as one token via a left-to-right chain of merges; return its id."""
    pieces = byte_pieces(spec, text)
    merges, acc = [], pieces[0]
    for p in pieces[1:]:
        if acc + p not in spec["model"]["vocab"]:
            merges.append((acc, p))
        acc += p
    spec, ids = extend(spec, merges)
    return spec, ids[-1]


def category(df, token_id):
    return df.loc[df["id"] == token_id, "category"].item()


def test_ordinary_tokens_are_reachable():
    df = classify(make_base())
    ascii_text = df["text"].map(lambda t: t != "" and t.isascii())
    assert (df.loc[ascii_text, "category"] == "reachable").all()


def test_token_spanning_a_vowel_sign_is_split_by_the_pretokenizer():
    # త + ె: the vowel sign is \p{M}, which Qwen's regex will not keep in a letter run.
    spec, tid = plant(make_base(), "తె")
    df = classify(spec)
    assert category(df, tid) == "split_by_pretokenizer"
    assert df.loc[df["id"] == tid, "unlocked_by_keep_marks"].item()


def test_token_the_normalizer_rewrites_is_unreachable():
    # U+F900 is a CJK compatibility ideograph; NFC maps it to U+8C48.
    spec, tid = plant(make_base(), "豈")
    df = classify(spec)
    assert category(df, tid) == "changed_by_normalizer"


def test_normalizer_takes_precedence_over_the_split_regex():
    # U+FB35 (Hebrew vav with dagesh) NFC-decomposes to vav + U+05BC, a mark the
    # split regex then cuts off. The normalizer is the reason it is unreachable.
    spec, tid = plant(make_base(), "וּ")
    df = classify(spec)
    assert category(df, tid) == "changed_by_normalizer"


def test_partial_character_is_a_fragment():
    spec, tid = plant(make_base(), "త")  # plant the whole char, then check its 2-byte prefix
    pieces = byte_pieces(spec, "త")
    df = classify(spec)
    two_bytes = spec["model"]["vocab"][pieces[0] + pieces[1]]
    assert category(df, two_bytes) == "fragment"
    assert category(df, tid) == "reachable"


def test_script_of():
    assert script_of("ขณะนี้") == "Thai"
    assert script_of(" hello") == "Latin"
    assert script_of("中文") == "Han"
    assert script_of(" 123 ") == "none"
