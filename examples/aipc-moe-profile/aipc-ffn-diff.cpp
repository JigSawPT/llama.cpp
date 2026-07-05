// aipc-ffn-diff: capture selected FFN tensors for ONE designated MoE layer during a
// single fixed forward pass, and dump them to a binary file for sign-sensitive diffing.
//
// Purpose (SWIGLU_OAI dual-chain verification, decisive test #1): prove the dual-chain
// FFN output equals the stock FFN output for one gpt-oss layer on a FIXED input,
// separating a real systematic bias (structured residual) from benign atomic-order noise.
//
// Run twice with the SAME prompt/config:
//   cache OFF:  (no AIPC_MOE_HOT_* env)                    -> stock build_moe_ffn
//   cache ON :  AIPC_MOE_HOT_LIST=... AIPC_MOE_HOT_N=20    -> build_moe_ffn_split
// then diff the two dumps element-wise (analyze externally).
//
// Env:
//   AIPC_DIFF_LAYER : target layer il (default 10, the first CPU-resident gpt-oss layer at ncmoe26)
//   AIPC_DIFF_OUT   : output prefix (writes <prefix>.<tensorname>.bin + <prefix>.meta.txt)
//
// Tensors captured (all for the target layer; whichever appear in the graph):
//   ffn_moe_out        - FINAL aggregated MoE output [n_embd, n_tokens] (same name both paths) -- PRIMARY
//   ffn_moe_down       - per-expert down output [n_embd, n_expert_used, n_tokens]
//                        (stock: BEFORE down bias ; split: merged output INCLUDING down bias)
//   ffn_moe_down_biased- stock only: per-expert down output WITH bias (== split's ffn_moe_down)
//   ffn_moe_weighted   - input control: hidden fed to the experts (identical OFF vs ON => valid diff)
//   ffn_inp            - control: FFN input residual (identical OFF vs ON on CPU)
//
// Usage: llama-aipc-ffn-diff -m model.gguf -p "..." -ngl 0 -t 1 --n-cpu-moe 26 [-b 1 -ub 1]
#include "arg.h"
#include "common.h"
#include "log.h"
#include "llama.h"
#include "ggml.h"

#include <cinttypes>
#include <clocale>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

struct diff_cfg {
    int         target_il = 10;
    std::string out_prefix = "aipc_ffn_diff";
    FILE *      meta = nullptr;
    int         n_captured = 0;
    bool        armed = false;   // only capture once armed (i.e. on the DECODE pass, n_tk=1,
                                 // where the AIPC split is active; prefill n_tk>8 bails to stock)
};

// tensor basenames we care about (name is "<base>-<il>")
static const char * kWanted[] = {
    "ffn_moe_out",
    "ffn_moe_down",
    "ffn_moe_down_biased",
    "ffn_moe_weighted",
    "ffn_inp",
};

// return the basename if t->name is "<wanted>-<target_il>", else nullptr
static const char * match_wanted(const char * name, int target_il) {
    for (const char * w : kWanted) {
        const size_t wl = strlen(w);
        if (strncmp(name, w, wl) == 0 && name[wl] == '-') {
            const int il = atoi(name + wl + 1);
            if (il == target_il) {
                return w;
            }
        }
    }
    return nullptr;
}

static bool cb_capture(struct ggml_tensor * t, bool ask, void * user_data) {
    auto * cfg = (diff_cfg *) user_data;
    const char * base = match_wanted(t->name, cfg->target_il);
    if (ask) {
        return cfg->armed && base != nullptr;   // only request contents on the armed (decode) pass
    }
    if (!base || !cfg->armed) {
        return true;
    }
    if (t->type != GGML_TYPE_F32) {
        LOG_INF("  [capture] %s: skipped (type %d != F32)\n", t->name, (int) t->type);
        return true;
    }
    const int64_t n = ggml_nelements(t);
    std::vector<float> data(n);
    ggml_backend_tensor_get(t, data.data(), 0, n * sizeof(float));

    // stats
    double sum = 0.0, sumsq = 0.0;
    float mn = INFINITY, mx = -INFINITY;
    int64_t n_nan = 0, n_inf = 0;
    for (int64_t i = 0; i < n; i++) {
        const float v = data[i];
        if (std::isnan(v)) { n_nan++; continue; }
        if (std::isinf(v)) { n_inf++; continue; }
        sum += v; sumsq += (double) v * v;
        if (v < mn) mn = v;
        if (v > mx) mx = v;
    }
    const double mean = sum / (double) n;
    const double l2   = std::sqrt(sumsq);

    // write raw f32 (little-endian, native) to <prefix>.<base>.bin
    char path[1024];
    snprintf(path, sizeof(path), "%s.%s.bin", cfg->out_prefix.c_str(), base);
    FILE * bf = fopen(path, "wb");
    if (bf) {
        fwrite(data.data(), sizeof(float), (size_t) n, bf);
        fclose(bf);
    } else {
        LOG_ERR("  [capture] could not write %s\n", path);
    }

    LOG_INF("  [capture] %-22s il=%d  ne=[%" PRId64 ",%" PRId64 ",%" PRId64 ",%" PRId64 "]  n=%" PRId64
            "  mean=%.9g  l2=%.9g  min=%.6g  max=%.6g  nan=%" PRId64 " inf=%" PRId64 "\n",
            base, cfg->target_il, t->ne[0], t->ne[1], t->ne[2], t->ne[3], n, mean, l2, mn, mx, n_nan, n_inf);
    if (cfg->meta) {
        fprintf(cfg->meta, "%s il=%d ne=%" PRId64 ",%" PRId64 ",%" PRId64 ",%" PRId64 " n=%" PRId64
                " mean=%.9g l2=%.9g min=%.6g max=%.6g nan=%" PRId64 " inf=%" PRId64 "\n",
                base, cfg->target_il, t->ne[0], t->ne[1], t->ne[2], t->ne[3], n, mean, l2, mn, mx, n_nan, n_inf);
        fflush(cfg->meta);
    }
    cfg->n_captured++;
    return true;
}

int main(int argc, char ** argv) {
    std::setlocale(LC_NUMERIC, "C");

    diff_cfg cfg;
    if (const char * e = std::getenv("AIPC_DIFF_LAYER")) cfg.target_il = atoi(e);
    if (const char * e = std::getenv("AIPC_DIFF_OUT"))   cfg.out_prefix = e;

    common_params params;
    common_init();

    if (!common_params_parse(argc, argv, params, LLAMA_EXAMPLE_COMMON)) {
        return 1;
    }

    llama_backend_init();
    llama_numa_init(params.numa);

    params.cb_eval = cb_capture;
    params.cb_eval_user_data = &cfg;
    params.warmup = false;

    auto llama_init = common_init_from_params(params);
    auto * model = llama_init->model();
    auto * ctx   = llama_init->context();
    if (model == nullptr || ctx == nullptr) {
        LOG_ERR("initialization failed\n");
        return 1;
    }

    char meta_path[1024];
    snprintf(meta_path, sizeof(meta_path), "%s.meta.txt", cfg.out_prefix.c_str());
    cfg.meta = fopen(meta_path, "wb");

    const llama_vocab * vocab = llama_model_get_vocab(model);
    std::vector<llama_token> tokens = common_tokenize(ctx, params.prompt, llama_vocab_get_add_bos(vocab), true);
    if (tokens.empty()) {
        LOG_ERR("no input tokens (use -p)\n");
        return 1;
    }
    LOG_INF("aipc-ffn-diff: target layer=%d, out prefix='%s', %zu input tokens\n",
            cfg.target_il, cfg.out_prefix.c_str(), tokens.size());

    // 1) PREFILL the prompt (n_tk = prompt length > 8 -> AIPC split bails to stock; NOT armed,
    //    so nothing is captured here). This populates the KV cache identically for OFF and ON.
    if (llama_decode(ctx, llama_batch_get_one(tokens.data(), (int32_t) tokens.size()))) {
        LOG_ERR("prefill llama_decode failed\n");
        return 1;
    }

    // 2) Arm, then do ONE single-token DECODE pass (n_tk = 1 <= 8 -> AIPC split ACTIVE).
    //    Feed a FIXED token (the last prompt token) so the decode input is identical across
    //    OFF and ON runs -> a valid sign-sensitive stock-vs-split diff of the FFN output.
    cfg.armed = true;
    llama_token dec_tok = tokens.back();
    if (const char * e = std::getenv("AIPC_DIFF_DECODE_TOK")) dec_tok = (llama_token) atoi(e);
    LOG_INF("aipc-ffn-diff: ARMED; single decode step with fixed token %d (n_tk=1, split active)\n", dec_tok);
    if (llama_decode(ctx, llama_batch_get_one(&dec_tok, 1))) {
        LOG_ERR("decode llama_decode failed\n");
        return 1;
    }

    LOG_INF("aipc-ffn-diff: captured %d tensors for layer %d -> %s.*.bin\n",
            cfg.n_captured, cfg.target_il, cfg.out_prefix.c_str());

    // 3) OPTIONAL multi-token NaN stress: greedily decode AIPC_DIFF_NGEN more tokens (n_tk=1,
    //    split active every step) and assert the logits stay finite. Prints the decoded text so
    //    coherence can be eyeballed. Used for the degenerate-hot-list stress (#5).
    int n_gen = 0;
    if (const char * e = std::getenv("AIPC_DIFF_NGEN")) n_gen = atoi(e);
    if (n_gen > 0) {
        cfg.armed = false;   // stop tensor dumping; we only check finiteness now
        const int n_vocab = llama_vocab_n_tokens(vocab);
        llama_token cur = dec_tok;
        std::string text;
        int n_nonfinite = 0;
        for (int g = 0; g < n_gen; g++) {
            if (llama_decode(ctx, llama_batch_get_one(&cur, 1))) {
                LOG_ERR("gen decode failed at step %d\n", g);
                return 1;
            }
            const float * logits = llama_get_logits_ith(ctx, -1);
            // argmax + finiteness scan
            int best = 0; float bestv = -INFINITY;
            for (int v = 0; v < n_vocab; v++) {
                const float lv = logits[v];
                if (std::isnan(lv) || std::isinf(lv)) { n_nonfinite++; continue; }
                if (lv > bestv) { bestv = lv; best = v; }
            }
            cur = best;
            char piece[256];
            int np = llama_token_to_piece(vocab, cur, piece, sizeof(piece), 0, true);
            if (np > 0) text.append(piece, np);
        }
        LOG_INF("aipc-ffn-diff: NaN-stress gen %d tokens, non-finite logits=%d\n", n_gen, n_nonfinite);
        LOG_INF("aipc-ffn-diff: GEN TEXT >>>%s<<<\n", text.c_str());
        if (cfg.meta) fprintf(cfg.meta, "nan_stress_ngen=%d nonfinite_logits=%d text=%s\n", n_gen, n_nonfinite, text.c_str());
    }

    if (cfg.meta) fclose(cfg.meta);

    llama_backend_free();
    return 0;
}
