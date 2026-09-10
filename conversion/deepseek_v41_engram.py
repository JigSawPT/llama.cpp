"""Engram (conditional memory) tables for DeepSeek-V4.1.

The lookup tables are a deterministic function of the tokenizer and of a few hparams. Building
them needs Unicode normalization and a primality test, so they are computed here and written to
the GGUF as tensors. The C++ side then only does integer arithmetic.

Layout, from inference/engram.py:
  n_hash_cols = (max_ngram_size - 1) * n_heads          24 for V4.1
  every (ngram size, head) pair owns a disjoint prime-sized bucket range in the layer table
  primes are drawn in order from engram_vocab_size - 1 and never reused across layers
"""

from __future__ import annotations

import numpy as np


def build_compressed_token_map(tokenizer) -> tuple[list[int], int]:
    """Token ids collapsed so that tokens which normalize alike hash alike.

    Kept identical to inference/engram.py: a mismatch in the compressed vocab size silently
    rehashes the whole table, because every multiplier derives from that size.
    """
    from tokenizers import Regex, normalizers

    sentinel = "\ue000"  # keeps a single-space token from collapsing to the empty string
    normalizer = normalizers.Sequence([
        normalizers.NFKC(),
        normalizers.NFD(),
        normalizers.StripAccents(),
        normalizers.Lowercase(),
        normalizers.Replace(Regex(r"[ \t\r\n]+"), " "),
        normalizers.Replace(Regex(r"^ $"), sentinel),
        normalizers.Strip(),
        normalizers.Replace(sentinel, " "),
    ])

    backend = tokenizer.backend_tokenizer
    key_to_new: dict[str, int] = {}
    lookup = [0] * len(tokenizer)
    for token_id in range(len(tokenizer)):
        text = backend.decode([token_id], skip_special_tokens=False)
        if "\ufffd" in text:
            key = backend.id_to_token(token_id)
        else:
            normalized = normalizer.normalize_str(text)
            key = normalized if normalized else text

        new_id = key_to_new.get(key)
        if new_id is None:
            new_id = len(key_to_new)
            key_to_new[key] = new_id
        lookup[token_id] = new_id

    return lookup, len(key_to_new)


def _next_prime(start: int, seen: set[int]) -> int:
    from sympy import isprime

    candidate = start + 1
    while not isprime(candidate) or candidate in seen:
        candidate += 1
    return candidate


def build_primes(layer_ids, max_ngram_size: int, n_heads: int, vocab_size: int) -> np.ndarray:
    """[n_layers, max_ngram_size - 1, n_heads] bucket moduli, drawn in order and never reused."""
    primes, seen = [], set()
    for _ in layer_ids:
        per_ngram = []
        for _ in range(max_ngram_size - 1):
            sizes, current = [], vocab_size - 1
            for _ in range(n_heads):
                current = _next_prime(current, seen)
                seen.add(current)
                sizes.append(current)
            per_ngram.append(sizes)
        primes.append(per_ngram)
    return np.array(primes, dtype=np.int64)


def build_offsets(primes: np.ndarray) -> np.ndarray:
    """[n_layers, n_hash_cols] start of each bucket range inside the layer table."""
    flat = primes.reshape(primes.shape[0], -1)
    out = np.zeros_like(flat)
    out[:, 1:] = np.cumsum(flat[:, :-1], axis=1)
    return out


def build_multipliers(layer_ids, max_ngram_size: int, compressed_vocab_size: int) -> np.ndarray:
    """[n_layers, max_ngram_size] odd multipliers, one per lookback, from a per-layer RNG."""
    bound = max(1, (np.iinfo(np.int64).max // compressed_vocab_size) // 2)
    rows = []
    for layer_id in layer_ids:
        rng = np.random.default_rng(10007 * int(layer_id))
        rows.append(rng.integers(low=0, high=bound, size=(max_ngram_size,), dtype=np.int64) * 2 + 1)
    return np.stack(rows)


def hash_ids_for_tokens(compressed: list[int], pad_id: int, primes: np.ndarray,
                        offsets: np.ndarray, multipliers: np.ndarray,
                        max_ngram_size: int) -> np.ndarray:
    """Reference implementation of the C++ side, used only to test it.

    `compressed` is the whole sequence of compressed ids; -1 marks a dead token. Returns
    [n_tokens, n_layers, n_hash_cols].
    """
    n = len(compressed)
    n_layers = primes.shape[0]
    tokens = np.empty((n, max_ngram_size), dtype=np.int64)
    for i in range(n):
        blocked = False
        for shift in range(max_ngram_size):
            src = compressed[max(i - shift, 0)]
            blocked = blocked or i < shift or src == -1
            tokens[i, shift] = pad_id if blocked else src

    products = tokens[:, None, :] * multipliers[None, :, :]
    rolling = products[:, :, 0].copy()
    flat_primes = primes.reshape(n_layers, -1)
    out = np.empty((n, n_layers, primes.shape[1] * primes.shape[2]), dtype=np.int64)
    for i in range(1, max_ngram_size):
        rolling = np.bitwise_xor(rolling, products[:, :, i])
        cols = slice((i - 1) * primes.shape[2], i * primes.shape[2])
        out[:, :, cols] = rolling[:, :, None] % primes[None, :, i - 1, :]
    return out + offsets[None, :, :]
