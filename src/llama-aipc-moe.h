#pragma once

#include "ggml.h"

// AIPC V2.0 (experimental): hot/cold split of MoE experts.
// The original expert tensor (CPU/RAM-resident via -ot/--n-cpu-moe) stays
// complete; the most-activated experts (hot-list from aipc-moe-profile) get a
// VRAM copy plus remapping tables for the dual path in build_moe_ffn.
// Global registry guarded by a mutex; cleared on every model load (concurrent
// multi-model in one process remains out of scope for V2.0).
struct aipc_moe_split {
    ggml_tensor * hot;    // VRAM copy of the hot experts [ne0, ne1, n_hot + n_pad]
                          //   slots [n_hot, n_hot+n_pad) are ZEROED dummies, one per
                          //   k position, to keep ids unique per token in mul_mat_id
    ggml_tensor * hot_b;  // VRAM copy of the per-expert bias in the SAME hot-slot order
                          //   [ne_b0, n_hot + n_pad], dummy slots ZEROED; nullptr when the
                          //   original expert projection has no bias (plain SwiGLU). Indexed
                          //   by sel_hot exactly like `hot`, so the dummy-slot ids that carry
                          //   the unique-ids invariant also select a zeroed (finite) bias row
                          //   -> no 0*Inf=NaN leak and correct bias at hot positions.
    ggml_tensor * lookup; // F32 [n_expert]: global id -> hot slot (0 if cold; see iota)
    ggml_tensor * mask;   // F32 [n_expert]: 1.0 hot, 0.0 cold
    ggml_tensor * iota;   // F32 [n_pad]: values n_hot..n_hot+n_pad-1 (per-position dummy slots)
    int n_hot;
    int n_pad;            // = the model's n_expert_used
};

const aipc_moe_split * aipc_moe_split_lookup(const ggml_tensor * exps);
void aipc_moe_split_register(const ggml_tensor * exps, const aipc_moe_split & split);
void aipc_moe_split_clear();
