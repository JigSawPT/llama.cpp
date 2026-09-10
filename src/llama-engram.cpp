#include "llama-engram.h"

#include <cmath>

// FP8 E4M3: 1 sign, 4 exponent (bias 7), 3 mantissa. 0xFF and 0x7F are NaN.
static float e4m3_to_fp32(uint8_t v) {
    const int sign = (v >> 7) & 1;
    const int exp  = (v >> 3) & 0xF;
    const int mant = v & 0x7;

    float mag;
    if (exp == 0) {
        mag = ldexpf((float) mant / 8.0f, -6);
    } else if (exp == 0xF && mant == 0x7) {
        mag = NAN;
    } else {
        mag = ldexpf(1.0f + (float) mant / 8.0f, exp - 7);
    }

    return sign ? -mag : mag;
}

// E8M0: a bare exponent with bias 127. 0xFF is NaN.
static float e8m0_to_fp32(uint8_t v) {
    return v == 0xFF ? NAN : ldexpf(1.0f, (int) v - 127);
}

static const float * e4m3_table() {
    static float tbl[256];
    static bool built = false;
    if (!built) {
        for (int i = 0; i < 256; ++i) {
            tbl[i] = e4m3_to_fp32((uint8_t) i);
        }
        built = true;
    }
    return tbl;
}

void llama_engram_hash_row(const llama_engram_tables & t, uint32_t il, const int32_t * hist, int64_t * rows) {
    const int64_t * mul = t.multipliers + (size_t) il*t.max_ngram;
    const int64_t * prm = t.primes      + (size_t) il*t.n_hash_cols;
    const int64_t * off = t.offsets     + (size_t) il*t.n_hash_cols;

    int64_t rolling = (int64_t) hist[0] * mul[0];

    for (uint32_t g = 1; g < t.max_ngram; ++g) {
        rolling ^= (int64_t) hist[g] * mul[g];

        // the running value after step g is the hash of the (g+1)-gram; each head takes it
        // modulo its own prime, so the heads land in disjoint ranges
        for (uint32_t h = 0; h < t.n_head; ++h) {
            const uint32_t col = (g - 1)*t.n_head + h;

            // rolling is an XOR and can be negative; C++ % keeps the sign of the dividend while
            // the reference uses Python semantics, where the result follows the divisor
            int64_t r = rolling % prm[col];
            if (r < 0) {
                r += prm[col];
            }

            rows[col] = r + off[col];
        }
    }
}

void llama_engram_dequant_row(const uint8_t * w, const uint8_t * s, uint32_t head_dim, float * dst) {
    const float * tbl = e4m3_table();

    for (uint32_t b = 0; b < head_dim/32; ++b) {
        const float scale = e8m0_to_fp32(s[b]);

        for (uint32_t j = 0; j < 32; ++j) {
            dst[b*32 + j] = tbl[w[b*32 + j]] * scale;
        }
    }
}
