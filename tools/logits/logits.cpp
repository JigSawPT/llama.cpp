// Dump the logits of one forward pass, so a port can be compared against a reference
// implementation on numbers instead of on generated text.
//
//   llama-logits -m model.gguf -p "prompt" -o logits.bin [--tokens 1,2,3] [-ngl N] [--moe-stream]
//
// Writes n_vocab float32 values for the LAST position of the prompt, and prints the top-10
// so a mismatch is visible without opening the file. Token ids can be given directly with
// --tokens, which is what a cross-implementation comparison needs: the two tokenizers must
// not be part of what is being compared.

#include "llama.h"

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

static void uso(const char * exe) {
    printf("usage: %s -m model.gguf [-p prompt | --tokens 1,2,3] -o out.bin\n", exe);
    printf("  -ngl N            layers on the GPU\n");
    printf("  -c N              context size (default: prompt length rounded up)\n");
    printf("  --moe-stream      stream routed experts from disk\n");
    printf("  --moe-stream-cache N   expert cache budget in GiB\n");
    printf("  --moe-stream-l2 N      host RAM tier in GiB\n");
    printf("  --no-bos          do not prepend BOS when tokenizing -p\n");
}

int main(int argc, char ** argv) {
    std::string modelo, prompt, saida, lista_tokens;
    int   ngl = 0, n_ctx = 0, moe_cache_gib = 0, moe_l2_gib = 0;
    bool  moe_stream = false, add_bos = true;

    for (int i = 1; i < argc; i++) {
        const std::string a = argv[i];
        auto proximo = [&](const char * nome) -> std::string {
            if (i + 1 >= argc) { fprintf(stderr, "%s needs a value\n", nome); exit(1); }
            return argv[++i];
        };
        if      (a == "-m")                    modelo        = proximo("-m");
        else if (a == "-p")                    prompt        = proximo("-p");
        else if (a == "-o")                    saida         = proximo("-o");
        else if (a == "--tokens")              lista_tokens  = proximo("--tokens");
        else if (a == "-ngl")                  ngl           = atoi(proximo("-ngl").c_str());
        else if (a == "-c")                    n_ctx         = atoi(proximo("-c").c_str());
        else if (a == "--moe-stream")          moe_stream    = true;
        else if (a == "--moe-stream-cache")  { moe_cache_gib = atoi(proximo("--moe-stream-cache").c_str()); moe_stream = true; }
        else if (a == "--moe-stream-l2")     { moe_l2_gib    = atoi(proximo("--moe-stream-l2").c_str());    moe_stream = true; }
        else if (a == "--no-bos")              add_bos       = false;
        else { uso(argv[0]); return 1; }
    }

    if (modelo.empty() || saida.empty() || (prompt.empty() && lista_tokens.empty())) {
        uso(argv[0]);
        return 1;
    }

    ggml_backend_load_all();

    llama_model_params mp = llama_model_default_params();
    mp.n_gpu_layers = ngl;
    // expert streaming is a load-time property of the model, not of the context
    mp.moe_stream        = moe_stream;
    mp.moe_stream_budget = (uint64_t) moe_cache_gib * 1024ull*1024ull*1024ull;
    mp.moe_stream_l2_gib = (uint32_t) moe_l2_gib;

    llama_model * model = llama_model_load_from_file(modelo.c_str(), mp);
    if (!model) {
        fprintf(stderr, "failed to load %s\n", modelo.c_str());
        return 1;
    }
    const llama_vocab * vocab = llama_model_get_vocab(model);

    std::vector<llama_token> tokens;
    if (!lista_tokens.empty()) {
        // ids given directly: the tokenizer stays out of the comparison
        size_t pos = 0;
        while (pos < lista_tokens.size()) {
            size_t virgula = lista_tokens.find(',', pos);
            if (virgula == std::string::npos) virgula = lista_tokens.size();
            tokens.push_back((llama_token) atoi(lista_tokens.substr(pos, virgula - pos).c_str()));
            pos = virgula + 1;
        }
    } else {
        const int n = -llama_tokenize(vocab, prompt.c_str(), prompt.size(), NULL, 0, add_bos, true);
        tokens.resize(n);
        if (llama_tokenize(vocab, prompt.c_str(), prompt.size(), tokens.data(), tokens.size(), add_bos, true) < 0) {
            fprintf(stderr, "tokenize failed\n");
            return 1;
        }
    }
    if (tokens.empty()) {
        fprintf(stderr, "no tokens\n");
        return 1;
    }

    llama_context_params cp = llama_context_default_params();
    cp.n_ctx   = n_ctx > 0 ? n_ctx : (uint32_t) tokens.size() + 8;
    cp.n_batch = (uint32_t) tokens.size();

    llama_context * ctx = llama_init_from_model(model, cp);
    if (!ctx) {
        fprintf(stderr, "failed to create the context\n");
        return 1;
    }

    printf("logits: %zu tokens:", tokens.size());
    for (const llama_token t : tokens) {
        printf(" %d", t);
    }
    printf("\n");

    llama_batch batch = llama_batch_get_one(tokens.data(), tokens.size());
    if (llama_decode(ctx, batch)) {
        fprintf(stderr, "decode failed\n");
        return 1;
    }

    const int n_vocab = llama_vocab_n_tokens(vocab);
    const float * lg = llama_get_logits_ith(ctx, -1);
    if (!lg) {
        fprintf(stderr, "no logits\n");
        return 1;
    }

    FILE * f = fopen(saida.c_str(), "wb");
    if (!f) {
        fprintf(stderr, "cannot write %s\n", saida.c_str());
        return 1;
    }
    fwrite(lg, sizeof(float), n_vocab, f);
    fclose(f);

    std::vector<int> ord(n_vocab);
    for (int i = 0; i < n_vocab; i++) ord[i] = i;
    std::partial_sort(ord.begin(), ord.begin() + 10, ord.end(),
            [&](int a, int b) { return lg[a] > lg[b]; });

    printf("logits: %d values written to %s\n", n_vocab, saida.c_str());
    printf("logits: top 10\n");
    for (int i = 0; i < 10; i++) {
        char buf[128];
        const int n = llama_token_to_piece(vocab, ord[i], buf, sizeof(buf) - 1, 0, true);
        buf[n > 0 ? n : 0] = 0;
        printf("  %2d. id %6d  %12.6f  %s\n", i + 1, ord[i], lg[ord[i]], buf);
    }

    llama_free(ctx);
    llama_model_free(model);
    return 0;
}
