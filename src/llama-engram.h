#pragma once

#include <cstdint>

// DeepSeek-V4.1 engram: conditional memory read from a hashed n-gram table.
//
// A position is hashed as (max_ngram - 1) n-grams, each split over n_head heads, so every
// (n-gram size, head) pair owns its own prime-sized bucket range inside the layer table.
// The tables come from the GGUF: building them needs the tokenizer, Unicode normalization and
// a primality test, so the C++ side only does integer arithmetic.
//
// The table is ~91 GiB per layer and only 24 rows are read per token, so the gather and the
// dequantization run on the host and the graph gets the result as a plain F32 input.

struct llama_engram_tables {
    const int32_t * token_map   = nullptr; // [n_vocab]                  token id -> compressed id
    const int64_t * primes      = nullptr; // [n_hash_cols, n_layer]     bucket modulus
    const int64_t * offsets     = nullptr; // [n_hash_cols, n_layer]     start of each bucket range
    const int64_t * multipliers = nullptr; // [max_ngram,   n_layer]     odd, one per lookback

    uint32_t n_layer      = 0;
    uint32_t n_head       = 0;
    uint32_t max_ngram    = 0;
    uint32_t n_hash_cols  = 0; // (max_ngram - 1) * n_head
    uint32_t head_dim     = 0;
    int32_t  pad_id       = 0; // already mapped through token_map
};

// hist[0] is the current token, hist[k] the one k positions back, already padded and blocked.
// Writes n_hash_cols row indices.
void llama_engram_hash_row(const llama_engram_tables & t, uint32_t il, const int32_t * hist, int64_t * rows);

// One table row: head_dim values in FP8 E4M3 with one E8M0 scale per 32.
void llama_engram_dequant_row(const uint8_t * w, const uint8_t * s, uint32_t head_dim, float * dst);
