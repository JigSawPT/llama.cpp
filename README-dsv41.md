# DeepSeek-V4.1-Flash on llama.cpp: the `dsv41-porte` branch

This branch runs [DeepSeek-V4.1-Flash](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash)
(552B MoE, 40 layers, 384 routed experts, a 189 GiB n-gram memory) on a single consumer GPU by
streaming experts from NVMe through a two-tier cache (VRAM + pinned RAM). Measured on an RTX 5090
with 31.8 GiB of VRAM and 125.7 GiB of RAM: 5.1 tokens/s on new content, 21 tokens/s on resident
content, logits within the reference's own fp8 rounding floor.

The full report, the measurement tools and the raw results live in
[deepseek-v41-flash-on-5090](https://github.com/JigSawPT/deepseek-v41-flash-on-5090). The models are on
Hugging Face: [DeepSeek-V4.1-Flash-GGUF](https://huggingface.co/JigSawPT/DeepSeek-V4.1-Flash-GGUF)
(target, 11 shards) and
[DeepSeek-V4.1-Flash-DSpark-GGUF](https://huggingface.co/JigSawPT/DeepSeek-V4.1-Flash-DSpark-GGUF)
(draft head).

## What is in the branch

Base: upstream `1c3c967` (b10269). On top of it, in order:

| piece | commits | origin |
|---|---|---|
| `moe-stream`: expert streaming from disk with a VRAM cache and a pinned host tier (CLOCK eviction), JigSaw expert pinning | `4aa9d62`, `dd94447` | [Crow 0.3.2](https://github.com/nibor1896/Crow) by nibor1896, MIT, applied as a patch series; the CLOCK eviction and the pinning are additions made here |
| `deepseek41` architecture: converter, V4.1 compressor path, shared compressed caches, hierarchical indexer (first level), engram tables kept on disk (`TENSOR_HOST`), the hyper-connection mix threaded one sub-layer ahead, no per-head q norm | `9da1cbf` .. `c1b783c` | this branch |
| `tools/logits`: dump one forward pass, per-layer states and any named graph node, to compare a port on numbers | `eff0ca1`, `301a086`, `7b70bcc`, `30ac511` | this branch |
| engram prefetch: hash every row first, then `PrefetchVirtualMemory` / `posix_madvise` the whole list (11.4x on the engram cost per token) | `f53b5dc` | this branch |
| speculative host-tier queue with a trace oracle, off by default (an instrument: it measured that perfect knowledge of future routing pays +30 % and that no layer predictor reaches it) | `b8e2958`, `aeda6df` | this branch |
| recurrent-state rollback for every speculative type, not only model drafts (`need_n_rs_seq`) | `caab38d` | this branch |
| DSpark draft head of V4.1: `--dspark` export for `DeepseekV41ForCausalLM`, `dflash.dsv41_semantics` | `30848ae` | this branch |

`LLAMA_DSV41_NO_HC_THREAD=1` applies each sub-layer's own hyper-connection mix instead of the
carried one (the V4 rule). It exists for A/B tests only and will not be part of an upstream PR.

## Build (Windows, CUDA)

CUDA 13.0. The 13.2 toolkit miscompiles `mul_mat_id` for the iq quantization types on Blackwell,
so the toolset is pinned:

```
cmake -B build -G "Visual Studio 17 2022" -T cuda=13.0 -DGGML_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=120 -DLLAMA_BUILD_EXAMPLES=ON -DLLAMA_CURL=OFF
cmake --build build --config Release --target llama-server llama-cli llama-gguf-split -j 8
```

Linux builds are untested on this branch; the streaming code has POSIX paths (`pread`,
`posix_madvise`) but nobody has run them.

## Run

The target GGUF is 502 GB, of which 189 GiB are the two engram tables. They are never loaded:
the loader maps them and the host reads 48 rows per token. Everything else that does not fit in
the VRAM cache streams from disk through the host tier. Put the file on an NVMe.

```
llama-server -m DeepSeek-V4.1-Flash-MXFP4-engram-00001-of-00011.gguf -ngl 99 -c 8192 ^
  --moe-stream --moe-stream-cache 18 --moe-stream-l2 72 ^
  --reasoning off --host 127.0.0.1 --port 8080
```

| flag | meaning | measured |
|---|---|---|
| `--moe-stream-cache 18` | VRAM expert cache, GiB; minimum is 18 slots per layer (13 GiB) | 13..22 GiB: throughput flat at 4.3 |
| `--moe-stream-l2 72` | pinned host tier, GiB | 72 is the optimum on 125.7 GiB; 88 is slower (it steals page cache from the engram tables) |
| `--moe-stream-io-threads 1` | one I/O thread | bit-for-bit reproducible runs, 3.6 instead of 4.3 tokens/s |
| `--reasoning off` | chat mode | thinking mode at temperature 0 loops on vague prompts |
| `-md DeepSeek-V4.1-Flash-DSpark.gguf --spec-type draft-dspark --spec-draft-n-max 2 --spec-draft-n-cpu-moe 3` | the model's own draft head | neutral on mixed content (-4 % cold, 0 % resident); +12-15 % on verbatim repetition only |

## Convert

From the Hugging Face checkpoint (476 GB, fp8 + MXFP4 experts):

```
python convert_hf_to_gguf.py <DeepSeek-V4.1-Flash> --outtype bf16 --engram --outfile DeepSeek-V4.1-Flash-MXFP4-engram.gguf
python convert_hf_to_gguf.py <DeepSeek-V4.1-Flash> --dspark --target-model-dir <DeepSeek-V4.1-Flash> --outtype bf16 --outfile DeepSeek-V4.1-Flash-DSpark.gguf
llama-gguf-split --split --split-max-size 48G DeepSeek-V4.1-Flash-MXFP4-engram.gguf DeepSeek-V4.1-Flash-MXFP4-engram
```

The routed experts are a lossless repack of the released MXFP4 blocks (verified block by block);
attention and dense weights are dequantized from fp8 and stored as Q8_0/BF16; the engram tables
travel as their raw fp8 bytes with their scales. Without `--engram` the tables are left out and the
model is not the released model: zeroing the memory is the exact identity of the module, but the
outputs differ.

Upstream has an open conversion PR for V4.1 ([#28696](https://github.com/ggml-org/llama.cpp/pull/28696))
with the same architecture name and the same tensor names for the dense parts, but a different
storage for the engram tables and their hash constants (quantized table, metadata arrays). Files
from one converter do not load in the other's runtime; reconciling the two is part of the plan to
upstream this branch.

## Exactness, in one paragraph

Against the reference implementation at 1 401 tokens the logits correlate at 0.9967, the same as
the port against itself across two runs (0.9959): the divergence is indistinguishable from zero.
The remaining gap sits at the rounding floor of the reference's own fp8 linear layers (0.9999),
and the MoE gate amplifies it because it is a switch (86.7 % of the top-6 choices agree, the
6th-to-7th margin is about 1 %). The port is not reproducible across runs above ~1 024 tokens
because expert-cache slot assignment depends on I/O timing; one I/O thread makes it bit-exact.

## Credits

- DeepSeek for the model and the reference implementation (MIT).
- nibor1896 for Crow, whose `moe-stream` patch series is the base of the streaming path (MIT).
- ggml-org/llama.cpp, whose `deepseek4`, `dflash` and speculative code this branch builds on.
- Engineering on this branch was assisted by Claude (Anthropic); every commit says so.
