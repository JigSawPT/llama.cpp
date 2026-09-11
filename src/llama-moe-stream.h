#pragma once

#include "llama-mmap.h"

#include "ggml-cpp.h"

#include <algorithm>
#include <condition_variable>
#include <cstdint>
#include <deque>
#include <memory>
#include <mutex>
#include <string>
#include <thread>
#include <unordered_map>
#include <utility>
#include <vector>

// SSD streaming of MoE routed expert weights
//
// Streamed layers do not materialize their ffn_*_exps tensors; instead each weight gets a
// device-side cache tensor of n_slots expert slabs, filled on demand from the GGUF file by an
// id-remapping custom op that runs on the CPU right after the router top-k. The remap only
// changes which cache slot an expert id resolves to - it never changes which experts the router
// selected, so streaming affects latency, not outputs.
//
// Missing experts are loaded by a pool of I/O threads while the remap op waits; eviction is by
// decaying route hotness with an LRU tiebreak. Reads are buffered by default, or O_DIRECT with
// LLAMA_MOE_STREAM_DIRECT=1 (bypasses the page cache; recommended when the model far exceeds RAM).
//
// note: multiple contexts decoding the same streamed model concurrently are not supported -
// one context can evict slots referenced by the other's in-flight graph.

struct llama_moe_stream;

enum llama_moe_stream_slot_state : uint8_t {
    LLAMA_MOE_STREAM_SLOT_EMPTY    = 0,
    LLAMA_MOE_STREAM_SLOT_LOADING  = 1, // reserved, load queued or in flight
    LLAMA_MOE_STREAM_SLOT_RESIDENT = 2,
};

// one streamed weight tensor (gate/up/down or fused gate_up) of one layer
struct llama_moe_stream_weight {
    ggml_tensor * cache = nullptr; // cache tensor {ne0, ne1, n_slots}

    uint16_t file_idx  = 0; // GGUF split file index
    size_t   offs      = 0; // file offset of the full exps tensor data
    size_t   nb_expert = 0; // bytes per expert slab
};

struct llama_moe_stream_layer;

// userdata of one wave's custom ops (multi-pass prefill): identifies which pass this is
struct llama_moe_stream_wave {
    llama_moe_stream_layer * sl   = nullptr;
    int32_t                  wave = -1;
};

// per-layer streaming state - also the userdata of the id-remapping custom op
struct llama_moe_stream_layer {
    llama_moe_stream * mgr = nullptr;

    int32_t  il       = -1;
    uint32_t n_expert = 0;
    uint32_t n_slots  = 0;

    std::vector<llama_moe_stream_weight> weights; // 2 (fused gate_up + down) or 3 entries

    // residency state, guarded by mgr->mtx
    std::vector<int32_t>                 slot_expert;   // [n_slots] expert id or -1
    std::vector<uint8_t>                 slot_state;    // [n_slots] llama_moe_stream_slot_state
    // [n_slots] BITMASK over weights: bit w set means a worker already owns the read
    // of tensor w for this slot. It was a plain flag while one item carried the whole
    // expert; with one item per tensor several workers legitimately share a slot, and
    // a flag would let the first of them shut the others out.
    std::vector<uint8_t>                 slot_claimed;
    // [n_slots] tensors of this slot still to arrive. Set to weights.size() when the
    // slot is reserved and decremented by each finishing part; the slot turns RESIDENT
    // only at zero. With one item per expert this was implicit in the for loop that
    // read them; with one item per tensor it has to be counted, or a slot would be
    // published while some of its tensors are still unwritten.
    std::vector<uint16_t>                slot_parts_left;
    std::vector<uint64_t>                slot_gen;      // [n_slots] reservation generation
    std::vector<int64_t>                 slot_last_use; // [n_slots] LRU stamps
    std::unordered_map<int32_t, int32_t> expert_slot;   // RESIDENT and LOADING entries

    std::vector<uint32_t> route_hotness; // [n_expert] decayed selection counts, for eviction

    // [n_expert] undecayed selection counts, for measuring expert locality only. route_hotness
    // cannot answer that question: it is halved periodically so eviction can forget a formerly hot
    // expert, which is right for eviction and wrong for a histogram.
    std::vector<uint64_t> route_total;
    std::vector<uint8_t>  seen;          // [n_expert] for cold-miss attribution
    int64_t use_counter = 0;

    // scratch for the remap callback
    std::vector<int32_t> uniq;
    std::vector<uint8_t> touched;
    std::vector<uint8_t> keep;         // [n_slots] slots the current call must not evict
    // JigSaw (21/08): experts PINADOS - imunes ao despejo. A cache dinamica (route_hotness
    // + LRU) adapta-se, mas um expert quente pode ser despejado num desvio de carga e cada
    // regresso paga um miss. O pin da estabilidade ao nucleo que o perfil offline diz ser
    // estavel, e deixa a dinamica so para a cauda.
    //   LLAMA_MOE_STREAM_PIN_LIST = ficheiro "il id id ..." (formato do aipc-moe-profile)
    //   LLAMA_MOE_STREAM_PIN_N    = quantos por camada pinar (0 = desligado)
    std::vector<uint8_t> expert_pinned; // [n_expert] 1 = pinar quando carregar
    std::vector<uint8_t> slot_pinned;   // [n_slots]  1 = nunca despejar
    std::vector<int32_t> demand_slots; // slots the current call waits on

    // wave plan for multi-pass prefill (guarded by mgr->mtx): the touched experts are split into
    // plan_n_waves passes of at most plan_capacity experts each, run one pass at a time
    uint32_t plan_capacity  = 0;  // experts per wave, set at graph build
    uint32_t plan_n_waves   = 0;  // waves of the current call
    int32_t  plan_next_wave = -1; // wave expected to run next (ordering guard)
    std::vector<uint8_t> expert_wave; // [n_expert] wave each touched expert belongs to, 0xff = untouched
    std::vector<int32_t> plan_pool;   // resident slots the masked-out pairs of this wave park on
    std::vector<int32_t> pool_used;   // scratch: pool slots already used in the current token row

    std::vector<std::unique_ptr<llama_moe_stream_wave>> wave_ud; // stable per-wave op userdata

    // stable userdata for wave w (grows lazily); called at graph build time only
    llama_moe_stream_wave * wave_userdata(int32_t wave, uint32_t capacity);

    // whether the exps tensors passed to build_moe_ffn are this layer's cache tensors
    // (e.g. grovemoe evaluates a second, unstreamed expert group on the same layer index)
    bool matches(const ggml_tensor * gate, const ggml_tensor * up,
                 const ggml_tensor * down, const ggml_tensor * gate_up) const;
};

// One queued read. ONE PER WEIGHT TENSOR, not one per expert.
//
// An expert is 2 or 3 tensors (fused gate_up + down, or gate/up/down). Until
// 2026-08-07 a single work item carried the whole expert and one worker read
// its tensors in a for loop - strictly one read in flight at a time, which is
// why the drive saw a queue depth of 1.6 while it needs 8 to reach its rate.
//
// Splitting the item is what puts several reads in flight for a SINGLE token
// stream, with no extra bytes, no extra VRAM and no change to routing.
// Measured 2026-08-07 (runs/2026-08-07/striping): reads confined to one 9.2 MiB
// expert reach 9850 MB/s at depth 8, byte for byte the same as reads scattered
// over the whole 46 GiB file - so adjacency costs nothing and the drive really
// does serve the parts of one expert in parallel.
//
// `parts_left` lives on the layer, not here: every part of one slot has to
// decrement the same counter, and the slot only becomes RESIDENT at zero. A
// slot published before its last tensor lands would be read as valid weights
// while part of it is still garbage.
struct llama_moe_stream_work {
    llama_moe_stream_layer * sl = nullptr;

    int32_t  expert = -1;
    int32_t  slot   = -1;
    uint64_t gen    = 0; // stale unless it matches slot_gen[slot]
    int32_t  weight = -1; // index into sl->weights; -1 means "every tensor" (legacy/whole expert)

    // Speculative work fills the HOST tier only: no cache slot, no upload, no generation to
    // go stale. It exists because the graph cannot issue more than one layer of demand at a
    // time -- the router of layer L needs the output of L-1 -- so the drive never sees a queue
    // deeper than ~4 and gives 4,33 GB/s of the 10,04 it reaches at depth 6.
    bool     spec   = false;
};

// The smallest expert cache the multi-pass path can work with.
//
// The cache must hold three sets at once: this wave's experts, the next wave's preloaded experts
// (so its loads overlap this wave's compute), and n_expert_used parking slots the masked-out pairs
// GEMM against (Metal needs a slot at most once per token row). That gives
// cap + cap + n_expert_used = n_slots, and a wave must fit at least n_expert_used experts, so
// n_slots >= 3*n_expert_used.
//
// This is a PRECONDITION on the configuration, checked where n_slots is resolved. It used to be
// discovered inside graph building instead, on the first batch that touched more experts than the
// cache holds - measured 2026-08-03 in runs/2026-08-03/stream-a1-on/run-01.raw.txt as
// "have 16, need 18", where 16 was what the DEFAULT resolved to.
static inline uint32_t llama_moe_stream_min_slots(uint32_t n_expert_used) {
    return 3*n_expert_used;
}

// Worst-case wave plan for one ubatch. cap is experts per wave, n_waves passes over the expert
// GEMMs; split is false when the ubatch cannot touch more experts than the cache already holds.
struct llama_moe_stream_wave_budget {
    uint32_t cap     = 0;     // experts per wave; 0 when no split is needed
    uint32_t n_waves = 1;     // passes over the expert GEMMs
    bool     split   = false;
};

// The one place this is computed.
//
// It stood twice - in build_moe_ffn, whose cap becomes the runtime plan_capacity, and in
// graph_max_nodes, which sizes a node budget - and the two copies had already drifted in four
// ways: the budget copy left out the "does it need splitting at all" gate, clamped cap to 1
// instead of refusing an impossible capacity, computed n_touch_max in 32 bit, and multiplied by
// every layer instead of the streamed ones. Only the graph copy feeds the runtime, so none of
// that could make a run WRONG - but the two disagreed, measured for n_tokens between 5 and 10 at
// the 64-slot operating point, where the graph built one wave and the budget assumed three.
//
// UNIT: experts. PRECONDITION: n_slots >= llama_moe_stream_min_slots(n_expert_used), enforced at
// resolve time. The assert below is an invariant, not the error path for a user's value.
static inline llama_moe_stream_wave_budget llama_moe_stream_wave_plan(
        uint32_t n_slots, uint32_t n_expert, uint32_t n_expert_used, uint32_t n_tokens) {
    llama_moe_stream_wave_budget b;

    // worst-case distinct experts: n_expert_used per token, but never more than n_expert total.
    // 64 bit on purpose - one of the two old copies multiplied in 32.
    const uint64_t n_touch_max = std::min<uint64_t>((uint64_t) n_expert, (uint64_t) n_tokens*n_expert_used);
    if (n_touch_max <= (uint64_t) n_slots) {
        return b; // everything this ubatch can touch is already resident-able: one pass
    }

    GGML_ASSERT(n_slots >= llama_moe_stream_min_slots(n_expert_used));

    b.split   = true;
    b.cap     = (n_slots - n_expert_used)/2;
    b.n_waves = (uint32_t) ((n_touch_max + b.cap - 1)/b.cap); // ceil(n_touch_max/cap)
    return b;
}

// Second cache level in host RAM, below the VRAM slots and above the SSD.
//
// WHY IT CAN PAY AT ALL, measured on this machine 2026-08-09 at the shipped block sizes
// (2,686,976 B for gate/up, 3,211,264 B for down):
//
//     SSD pooled read      10,592.7 MB/s      pinned host -> device   47,357.4 MB/s
//
// Today a miss costs an SSD read into a staging buffer plus a pageable upload: 401.5 us for a
// gate/up slab. Out of pinned host memory the same slab uploads in 56.7 us - 7.08x per transfer.
//
// WHY IT COSTS NOTHING TO FILL, which is the part that makes it worth building. The worker
// already reads every missing slab into a staging buffer and throws that buffer away. Here the
// read goes STRAIGHT INTO an L2 slot and the upload sources from there, so filling the tier is
// not an extra copy - it is the same copy, kept. A design that memcpy'd into the tier after the
// read would spend more per fill than a later hit returns, and would be worse than no tier.
//
// WHAT IT CANNOT DO: catch a cold first touch. Those bytes have never been read, so no cache of
// any size holds them. Over the ten-task gate of 2026-08-06, cold misses were 10,363 of 301,864 -
// the rest are evictions, and only those are addressable here.
//
// PINNED, and it asks ggml rather than CUDA: ggml_backend_dev_host_buffer_type() is the backend's
// own page-locked allocator, so this file stays free of any one backend. If the device has none,
// the tier falls back to ordinary aligned memory and SAYS SO - it still saves the SSD read, it
// just uploads at pageable rates, and the difference between those was measured at 2.6x.
struct llama_moe_stream_l2 {
    enum state : uint8_t { FREE = 0, LOADING = 1, RESIDENT = 2 };

    struct entry {
        uint64_t key  = UINT64_MAX; // file index and offset of the slab, unique per weight
        uint32_t head = 0;          // payload start inside the slot (direct-I/O head padding)
        uint32_t len  = 0;
        uint32_t pins = 0;          // readers currently uploading OUT of this slot
        uint8_t  st   = FREE;
        uint8_t  ref  = 0;          // CLOCK reference bit: set on a hit, cleared by the hand
    };

    uint8_t * base        = nullptr; // n_entries * slot_stride bytes
    size_t    slot_stride = 0;       // max_nb_expert + 2*align, so any slab fits any slot
    size_t    n_entries   = 0;
    bool      pinned      = false;

    std::vector<entry>                   entries;
    std::unordered_map<uint64_t, size_t> index; // key -> entry; only LOADING/RESIDENT are in it
    size_t                               next_victim = 0; // CLOCK hand
    bool                                 clock       = true; // off = plain FIFO, for A/B

    ggml_backend_buffer_ptr buf; // holds the pinned allocation when there is one

    mutable std::mutex mtx;

    int64_t n_hit   = 0; // slab served out of host RAM, no SSD read
    int64_t n_fill  = 0; // slab read from SSD straight into a slot
    int64_t n_evict = 0;

    // Time spent WAITING to enter this tier's mutex, and the number of entries it is spread over.
    // Measured 2026-08-09: with the tier on, load stall fell from 346,389 to 150,083 ms - 196
    // seconds less waiting on the drive - and total wall time did not move. So the saved time is
    // being spent somewhere else, and this counter exists to say whether "somewhere else" is here.
    // Every find/reserve/commit/release of eight I/O workers passes through one lock, which is the
    // same serialisation the handle pool paid 2.22x against 1.01x to remove.
    //
    // It is a lower bound on the cost: it counts the wait, not the work done while holding, and
    // not the cache-line traffic the shared counters cause. If it comes back near zero, the theory
    // is wrong and the time is somewhere this counter cannot see.
    int64_t t_lock_us  = 0;
    int64_t n_lock_ops = 0;

    ~llama_moe_stream_l2();

    static uint64_t make_key(uint16_t file_idx, size_t offs) {
        return ((uint64_t) file_idx << 48) ^ (uint64_t) offs;
    }

    // Resident payload for this slab, or nullptr. A LOADING entry returns nullptr on purpose: the
    // caller then reads it itself rather than waiting, which keeps this tier off the critical path
    // of a second worker that happens to want the same slab.
    //
    // A HIT PINS THE SLOT and hands back its index. The caller MUST release() it once the upload
    // out of that memory has finished. Without the pin the slot is an ordinary eviction candidate
    // while it is still being read, a second worker reserves it, reads a different expert into it
    // from the drive, and the first worker uploads half of each. Measured 2026-08-09: the model
    // emitted 8,191 characters of "<<<<<<<<" instead of an answer. Nothing about that failure is
    // visible in a throughput number - it looks like a fast run.
    const uint8_t * find(uint16_t file_idx, size_t offs, size_t len, size_t * out_slot);
    // Is this slab resident? Unlike find(), this counts nothing and marks nothing: a
    // speculative probe that bumped n_hit would inflate the tier's own hit rate, and one
    // that set the reference bit would protect a slab from eviction on the strength of a
    // guess. Both corrupt the measurement this path exists to produce.
    bool has(uint16_t file_idx, size_t offs, size_t len);
    void            release(size_t slot);

    // A slot to read INTO, marked LOADING, or nullptr when the tier is off, already owns this
    // slab, or every slot is in flight. LOADING is itself the protection while the drive read and
    // the upload run; commit() is what ENDS that protection, so it must be called AFTER the
    // upload, never before.
    uint8_t * reserve(uint16_t file_idx, size_t offs, size_t len, size_t * out_slot);
    void      commit(size_t slot, size_t head);
    void      abandon(size_t slot);
};

struct llama_moe_stream {
    uint32_t n_slots      = 0; // expert cache slots per streamed layer
    int32_t  n_io_threads = 0;

    std::vector<std::unique_ptr<llama_moe_stream_layer>> layers; // [n_layer], null = not streamed

    llama_moe_stream(uint32_t n_layer, uint32_t n_slots, int32_t n_io_threads, bool direct);
    ~llama_moe_stream();

    llama_moe_stream_layer * layer(int32_t il) const {
        return il >= 0 && (size_t) il < layers.size() ? layers[il].get() : nullptr;
    }

    // registers a streamed weight of layer il and returns its cache tensor
    ggml_tensor * create_cache_tensor(
            int32_t il, ggml_backend_buffer_type_t buft, const ggml_tensor * meta,
            uint16_t file_idx, size_t offs);

    // allocate the cache tensor buffers (after all create_cache_tensor calls)
    void alloc_bufs(bool no_alloc);

    // reopen the GGUF files for streaming reads
    void open_files(const std::vector<std::string> & paths);

    size_t size_bufs() const;

    // role names the model instance this manager belongs to and is printed in front of every
    // line ("target: moe stream: ..."). One process holds one manager per model, so without it
    // two blocks are separable only by their order - which is not a property a parser can rely
    // on. nullptr prints no prefix at all and reproduces the pre-role output byte for byte.
    // JigSaw: quantos slots por camada podem ficar pinados (0 = pinning desligado).
    // Clampado a n_slots - 3*n_expert_used no resolve (llama-model.cpp) - os pins sao
    // slots EXTRA acima do minimo dinamico, senao o plano de vagas encrava (22/08).
    uint32_t pin_budget = 0;

    void print_stats(const char * role) const;
    void print_locality(const char * pfx) const; // expert selection concentration; called from print_stats

    bool use_direct_io = false; // O_DIRECT streaming reads (LLAMA_MOE_STREAM_DIRECT), no page cache

    // Router bias toward cache-resident experts: added to the selection logits before top-k
    // (LLAMA_MOE_STREAM_ROUTE_BIAS, a float; 0 disables the hook entirely). Every miss begins as a
    // selection, so this trades routing fidelity for residency - it deliberately changes the output
    // and is a measurement instrument, not a default.
    float route_bias = 0.0f;

    llama_files files; // privately reopened GGUF files, same indices as the loader's

    // Host-RAM tier below the VRAM slots. Off unless LLAMA_MOE_STREAM_L2_GIB is set; sized in
    // whole GiB because the useful range on a consumer box is tens of gigabytes and a finer knob
    // would suggest a precision the allocation does not have.
    std::unique_ptr<llama_moe_stream_l2> l2;
    size_t l2_gib = 0;

    // allocates the tier once max_nb_expert is known. no-op when l2_gib is 0.
    void alloc_l2(ggml_backend_buffer_type_t host_buft);

    size_t  max_nb_expert      = 0;
    int64_t hot_decay_interval = 0; // remap calls between route-hotness halvings (0 = no decay)

    // A4, prefetch. How many remap calls ahead the oracle looks, and the ceiling on the
    // speculative queue. ahead = 0 turns the whole thing off, which is the default: this path
    // is opt-in because it changes what the drive is asked for and nothing else should inherit
    // that by accident.
    int32_t spec_ahead     = 0;
    size_t  spec_queue_max = 512;
    // Own counter, not stats.n_calls: that one is bumped from two places (the remap and the
    // multi-pass planner) and only one of them can be the trace index without leaving holes
    // the size of the prefill.
    int64_t spec_calls     = 0;
    // Highest call index already queued. Without it, call c queues c+1..c+k and call c+1 queues
    // c+2..c+k+1, so every future call is queued k times over -- which is what put 487 000
    // entries through the queue at ahead=120 and made the sweep measure duplication, not depth.
    int64_t spec_frontier  = -1;
    // workers with id below this never take speculative work, so demand always has a pool
    int32_t n_spec_workers_from = 0;
    // One entry per remap call, in call order: the layer it ran on and the experts it touched.
    // The index IS the call number, so replay needs no notion of token boundary -- two runs of
    // the same prompt issue the same calls in the same order.
    std::vector<std::pair<int32_t, std::vector<int32_t>>> spec_oracle;
    std::vector<std::pair<int32_t, std::vector<int32_t>>> spec_trace;   // being recorded
    std::string                                           spec_trace_path;

    std::vector<std::pair<ggml_backend_buffer_type_t, ggml_context_ptr>> ctxs; // one per buft
    std::vector<ggml_backend_buffer_ptr> bufs;

    // load pool (queue and all layer residency state guarded by mtx)
    mutable std::mutex      mtx;
    std::condition_variable cv_work; // queued work or shutdown
    std::condition_variable cv_done; // a load committed or failed

    // Serialises ggml_backend_tensor_set ONLY - never the read.
    //
    // Measured 2026-08-07: splitting an expert into one work item per weight
    // tensor raised queue depth 1.60 -> 4.31 and decode 9.89 -> 12.76 tok/s,
    // but binary-search stopped being reproducible - byte-identical in 4 of 4
    // runs on the old path, two different outputs across two runs on the new
    // one. Four other deterministic tasks still matched the reference exactly,
    // so the slot counter was not publishing early; what changed is that
    // several workers now call ggml_backend_tensor_set concurrently, which the
    // old one-worker-per-expert loop never did.
    //
    // Kept separate from mtx on purpose: mtx guards residency state and is held
    // while workers sleep on the condition variables. Uploading under it would
    // put the whole pool back in single file and give up the gain. This lock is
    // held only for the copy itself, which is short next to the disk read.
    std::mutex              upload_mtx;

    std::deque<llama_moe_stream_work> q_demand;

    // Drained only when q_demand is empty, so speculation can never delay a load the graph is
    // waiting on. Bounded: a queue that grows faster than it drains is prefetching the past.
    std::deque<llama_moe_stream_work> q_spec;

    std::vector<std::thread> workers;
    bool workers_started = false;
    bool shutting_down   = false;
    bool load_failed     = false;

    bool debug = false;

    struct {
        int64_t n_calls     = 0; // remap invocations
        int64_t n_hit       = 0; // touched experts already resident or loading
        int64_t n_hit_pin   = 0; // JigSaw: hits servidos por slots pinados
        int64_t n_miss      = 0; // demand loads issued
        int64_t n_miss_cold = 0; // first-ever touch of an expert
        int64_t n_spec_queued = 0; // experts guessed ahead (A4)
        int64_t n_spec_filled = 0; // of those, actually read into the host tier
        int64_t t_stall_us  = 0; // wait time in miss handling, from the point every miss is issued

        // wait for a free slot, which t_stall_us cannot see: it starts only once all demand loads
        // are issued, while this one happens before, when every allowed slot is already loading and
        // pick_victim_locked has nothing to hand out. under a tight cache the thread waits here and
        // no counter used to notice, which made every wait-share figure a lower bound.
        int64_t t_victim_us = 0; // wait time for a slot to free up
        int64_t n_victim_waits = 0; // how often that wait was entered at all

        int64_t n_wave_calls     = 0; // wave-ids invocations (>= n_calls under multi-pass prefill)
        int64_t n_waves_run      = 0; // non-empty waves
        int64_t n_preload_issued = 0; // next-wave loads started during a wave's compute
        int64_t n_preload_ready  = 0; // wave experts already resident from the previous preload
        int64_t t_stall_wave_us  = 0; // wait time in wave miss handling
    } stats;

    // internals
    void start_workers_locked();
    // worker_id is the worker's dense index, 0..n_io_threads-1, and is passed all the way down to
    // llama_file::read_raw_at - on Windows it selects that worker's private handle, which is what
    // makes the pool parallel rather than merely correct
    void worker_loop(int worker_id);
    int32_t pick_victim_locked(llama_moe_stream_layer & sl, const uint8_t * keep) const;
    void reserve_slot_locked(llama_moe_stream_layer & sl, int32_t expert, int32_t slot);
    // queues one read per weight tensor and wakes the pool; see the .cpp for why
    // the split is what raises queue depth on a single stream
    void enqueue_slot_locked(llama_moe_stream_layer & sl, int32_t expert, int32_t slot);

    // queue a host-tier-only fill for an expert not yet asked for
    void speculate_locked(llama_moe_stream_layer & sl, int32_t expert);

    void load_oracle(const char * path);
    void save_trace() const;

    // record this call and, if an oracle is loaded, queue the next spec_ahead calls
    void trace_and_speculate_locked(llama_moe_stream_layer & sl);

    // multi-pass prefill helpers (called by llama_moe_stream_wave_ids, all under mtx)
    void plan_waves_locked(llama_moe_stream_layer & sl, const int32_t * ids, int64_t n); // wave 0: build the plan
    void stage_wave_locked(std::unique_lock<std::mutex> & lk, llama_moe_stream_layer & sl, int32_t w, uint32_t n_ids); // make wave w resident + preload next
    void emit_wave_slots(llama_moe_stream_layer & sl, const int32_t * ids, int32_t * out, int32_t w, uint32_t n_ids, int64_t n_tok); // write the slot ids
};

// callback of the id-remapping custom op inserted by build_moe_ffn
void llama_moe_stream_remap(ggml_tensor * dst, const ggml_tensor * a, int ith, int nth, void * userdata);

// Adds mgr->route_bias to the selection logits of every expert this layer currently holds
// (resident or loading), leaving masked-out entries at -inf untouched. Runs before top-k, so it
// changes which experts are selected. A bias of 0 copies the input unchanged.
void llama_moe_stream_route_bias(ggml_tensor * dst, const ggml_tensor * a, int ith, int nth, void * userdata);

// callbacks of the multi-pass prefill custom ops inserted by build_moe_ffn when a ubatch touches
// more experts than the cache holds; each src[0] is the contiguous selected ids
//   wave_ids:  makes wave w's expert slice resident and emits slot ids (masked pairs park on a pool)
//   wave_mask: emits 1.0 for pairs belonging to wave w, 0.0 otherwise
void llama_moe_stream_wave_ids (ggml_tensor * dst, int ith, int nth, void * userdata);
void llama_moe_stream_wave_mask(ggml_tensor * dst, int ith, int nth, void * userdata);
