// aipc-moe-profile: accumulates activation counts per (layer, expert) from the
// ffn_moe_topk callback and dumps a hot-list JSON for the hot/cold split (AI_PC V2).
// Usage: llama-aipc-moe-profile -m model.gguf -f corpus.txt -ngl 999 [--n-cpu-moe N] [-c 32768]
// Output: aipc_moe_profile.json in the current directory.
#include "arg.h"
#include "common.h"
#include "log.h"
#include "llama.h"

#include <cinttypes>
#include <clocale>
#include <cstdio>
#include <cstring>
#include <map>
#include <string>
#include <vector>

struct topk_stats {
    std::map<int, std::vector<int64_t>> counts; // layer -> per-expert count
    int64_t n_calls = 0;
};

static bool cb_topk(struct ggml_tensor * t, bool ask, void * user_data) {
    const bool is_topk = strncmp(t->name, "ffn_moe_topk-", 13) == 0;
    if (ask) {
        return is_topk;
    }
    if (!is_topk) {
        return true;
    }
    auto * st = (topk_stats *) user_data;
    const int il = atoi(t->name + 13);

    GGML_ASSERT(t->type == GGML_TYPE_I32);
    std::vector<uint8_t> data(ggml_nbytes(t));
    ggml_backend_tensor_get(t, data.data(), 0, ggml_nbytes(t));

    auto & c = st->counts[il];
    for (int64_t i1 = 0; i1 < t->ne[1]; i1++) {         // tokens
        for (int64_t i0 = 0; i0 < t->ne[0]; i0++) {     // used experts
            const int32_t id = *(const int32_t *) (data.data() + i1 * t->nb[1] + i0 * t->nb[0]);
            if (id >= 0) {
                if ((size_t) id >= c.size()) {
                    c.resize(id + 1, 0);
                }
                c[id]++;
            }
        }
    }
    st->n_calls++;
    return true;
}

static bool run(llama_context * ctx, const common_params & params) {
    const llama_model * model = llama_get_model(ctx);
    const llama_vocab * vocab = llama_model_get_vocab(model);

    std::vector<llama_token> tokens = common_tokenize(ctx, params.prompt, llama_vocab_get_add_bos(vocab), true);
    if (tokens.empty()) {
        LOG_ERR("no input tokens (use -p or -f)\n");
        return false;
    }
    const uint32_t n_ctx = llama_n_ctx(ctx);
    if (tokens.size() > n_ctx) {
        LOG_INF("corpus (%zu tokens) truncated to context (%u)\n", tokens.size(), n_ctx);
        tokens.resize(n_ctx);
    }
    LOG_INF("processing %zu tokens in blocks of %d\n", tokens.size(), params.n_batch);
    for (size_t i = 0; i < tokens.size(); i += params.n_batch) {
        const size_t n = std::min((size_t) params.n_batch, tokens.size() - i);
        if (llama_decode(ctx, llama_batch_get_one(tokens.data() + i, n))) {
            LOG_ERR("llama_decode failed at block offset %zu\n", i);
            return false;
        }
    }
    return true;
}

static void dump_json(const topk_stats & st, const char * path) {
    FILE * f = fopen(path, "wb");
    if (!f) {
        LOG_ERR("could not write %s\n", path);
        return;
    }
    fprintf(f, "{\n  \"layers\": {\n");
    bool first_l = true;
    for (const auto & [il, c] : st.counts) {
        if (!first_l) fprintf(f, ",\n");
        first_l = false;
        fprintf(f, "    \"%d\": {\"counts\": [", il);
        for (size_t e = 0; e < c.size(); e++) {
            fprintf(f, "%s%" PRId64, e ? "," : "", c[e]);
        }
        fprintf(f, "], \"hot\": [");
        // ids sorted by descending count
        std::vector<int> order(c.size());
        for (size_t e = 0; e < c.size(); e++) order[e] = (int) e;
        std::sort(order.begin(), order.end(), [&](int a, int b) { return c[a] > c[b]; });
        for (size_t e = 0; e < order.size(); e++) {
            fprintf(f, "%s%d", e ? "," : "", order[e]);
        }
        fprintf(f, "]}");
    }
    fprintf(f, "\n  }\n}\n");
    fclose(f);
}

int main(int argc, char ** argv) {
    std::setlocale(LC_NUMERIC, "C");

    topk_stats stats;
    common_params params;
    common_init();

    if (!common_params_parse(argc, argv, params, LLAMA_EXAMPLE_COMMON)) {
        return 1;
    }

    llama_backend_init();
    llama_numa_init(params.numa);

    params.cb_eval = cb_topk;
    params.cb_eval_user_data = &stats;
    params.warmup = false;

    auto llama_init = common_init_from_params(params);
    auto * model = llama_init->model();
    auto * ctx   = llama_init->context();
    if (model == nullptr || ctx == nullptr) {
        LOG_ERR("initialization failed\n");
        return 1;
    }

    if (!run(ctx, params)) {
        return 1;
    }

    const char * out = "aipc_moe_profile.json";
    dump_json(stats, out);

    // plain-text format for the loader (env AIPC_MOE_HOT_LIST): "il id id id ..."
    {
        FILE * f = fopen("aipc_moe_profile.hotlist", "wb");
        if (f) {
            for (const auto & [il, c] : stats.counts) {
                std::vector<int> order(c.size());
                for (size_t e = 0; e < c.size(); e++) order[e] = (int) e;
                std::sort(order.begin(), order.end(), [&](int a, int b) { return c[a] > c[b]; });
                fprintf(f, "%d", il);
                for (int id : order) fprintf(f, " %d", id);
                fprintf(f, "\n");
            }
            fclose(f);
            LOG_INF("hot-list (text) written to aipc_moe_profile.hotlist\n");
        }
    }

    // summary: top-N coverage per layer
    LOG_INF("\nlayers with data: %zu | callback calls: %" PRId64 "\n", stats.counts.size(), stats.n_calls);
    for (const auto & [il, c] : stats.counts) {
        int64_t total = 0, mx = 0;
        for (auto v : c) { total += v; mx = std::max(mx, v); }
        std::vector<int64_t> s(c.begin(), c.end());
        std::sort(s.rbegin(), s.rend());
        int64_t acc = 0; size_t n80 = 0;
        for (; n80 < s.size() && acc < (int64_t)(0.80 * total); n80++) acc += s[n80];
        if (il % 8 == 0) {
            LOG_INF("  layer %2d: %zu experts seen, top-%zu cover 80%%, max %.1f%%\n",
                    il, c.size(), n80, 100.0 * mx / std::max<int64_t>(total, 1));
        }
    }
    LOG_INF("hot-list written to %s\n", out);

    llama_backend_free();
    return 0;
}
