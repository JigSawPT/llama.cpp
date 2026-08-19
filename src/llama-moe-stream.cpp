#include "llama-moe-stream.h"

#include "llama-impl.h"

#include "ggml-backend.h"

#include <algorithm>
#include <cinttypes>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <stdexcept>

#ifdef _WIN32
#include <malloc.h>
#else
#include <fcntl.h>
#include <unistd.h>
#endif

static const uint32_t MOE_STREAM_IO_THREADS_DEFAULT = 9;
static const uint32_t MOE_STREAM_IO_THREADS_MAX     = 18;
static const int64_t  MOE_STREAM_HOT_DECAY_TOKENS   = 64;

// O_DIRECT alignment: 4096 is a multiple of any device logical block size (512/4096), so it is
// universally valid, and reading a few extra KB of head/tail padding per slab is negligible
static const size_t MOE_STREAM_DIRECT_ALIGN = 4096;

// saturating increment - route-hotness counters accumulate over a whole run and must not wrap
static uint32_t sat_inc(uint32_t & c) {
    if (c < UINT32_MAX - 1) {
        c++;
    }
    return c;
}

// page-aligned allocation, required both for O_DIRECT reads and for Metal private-buffer uploads
static void * moe_aligned_alloc(size_t n) {
#ifdef _WIN32
    return _aligned_malloc(n, MOE_STREAM_DIRECT_ALIGN);
#else
    void * p = nullptr;
    if (posix_memalign(&p, MOE_STREAM_DIRECT_ALIGN, n) != 0) {
        p = nullptr;
    }
    return p;
#endif
}

static void moe_aligned_free(void * p) {
#ifdef _WIN32
    _aligned_free(p);
#else
    free(p);
#endif
}

// read len bytes at file offset offs into staging (thread-safe positional read); staging must have
// room for len (+ 2*MOE_STREAM_DIRECT_ALIGN when direct). returns a pointer to the len bytes
// within staging, or nullptr on failure
static const uint8_t * llama_moe_stream_pread(llama_file & file, uint8_t * staging, size_t len, size_t offs, bool direct, int worker_id) {
#ifdef _WIN32
    GGML_UNUSED(direct);
    // Windows has a positional read now, out of a pool of private handles - one per worker_id.
    // The mutex this replaces bought correctness by serialising every reader, and that is not a
    // small price: measured 2026-08-03, a shared handle stays at 1.01x at queue depth 8 while a
    // handle per thread reaches 2.22x. Windows serialises on the file OBJECT, not the file pointer,
    // so removing the seek was never going to be enough on its own.
    //
    // The alignment is asked of the file rather than derived from `direct`, because read_raw_at is
    // the RAW form and the file is the only one who knows what it actually opened: 1 when buffered,
    // which degenerates the arithmetic below into a plain read, and the device sector size under
    // direct I/O - including the case where a direct open fell back to buffered on its own.
    const size_t a = file.read_alignment();
    if (a == 0 || a > MOE_STREAM_DIRECT_ALIGN) {
        // staging carries 2*MOE_STREAM_DIRECT_ALIGN of slack and not one byte more, so a larger
        // sector would run off the end of it. A named failure, not a heap overrun.
        LLAMA_LOG_WARN("%s: read alignment %zu is outside the staging slack of %zu\n",
                __func__, a, MOE_STREAM_DIRECT_ALIGN);
        return nullptr;
    }

    // same head/tail dance as the O_DIRECT branch below; a == 1 makes it a no-op
    const size_t aoffs = offs & ~(a - 1);
    const size_t head  = offs - aoffs;
    const size_t total = ((head + len + a - 1)/a)*a;

    try {
        const size_t got = file.read_raw_at(staging, total, aoffs, worker_id);
        // short only at end of file, and the aligned tail may legitimately run past it - what has
        // to arrive is the payload, not the padding
        if (got < head + len) {
            return nullptr;
        }
        return staging + head;
    } catch (...) {
        return nullptr;
    }
#else
    const int fd = file.file_id();

    if (direct) {
        // O_DIRECT requires the offset, length, and buffer all block-aligned
        const size_t a     = MOE_STREAM_DIRECT_ALIGN;
        const size_t aoffs = offs & ~(a - 1);
        const size_t head  = offs - aoffs;
        const size_t total = ((head + len + a - 1)/a)*a;
        ssize_t r;
        do {
            r = pread(fd, staging, total, aoffs);
        } while (r < 0 && errno == EINTR);
        if (r < 0 || (size_t) r < head + len) {
            return nullptr;
        }
        return staging + head;
    }

    uint8_t * p    = staging;
    size_t    left = len;
    while (left > 0) {
        const ssize_t r = pread(fd, p, left, offs);
        if (r < 0) {
            if (errno == EINTR) {
                continue;
            }
            return nullptr;
        }
        if (r == 0) {
            return nullptr; // unexpected EOF
        }
        p    += r;
        offs += (size_t) r;
        left -= (size_t) r;
    }
    return staging;
#endif
}

// true iff all of the given exps tensors are this layer's cache tensors - guards against a second,
// non-streamed expert group on the same layer index (e.g. grovemoe chexps)
// ---------------------------------------------------------------------------------------------
// L2: the host-RAM tier. See the header for why it exists and why filling it is free.
// ---------------------------------------------------------------------------------------------

// Takes the tier's lock and charges the wait to t_lock_us. Every entry point uses it, so the
// counter has the same denominator as the operations themselves - a partial instrumentation would
// under-report exactly the contention it exists to find.
//
// The clock is read BEFORE the lock and the counter written after, under the lock, so no atomic is
// needed for the counter itself. ggml_time_us() costs something and that cost lands inside the
// measurement; at this call rate that is a reason to read the result as an upper bound on cheap
// waits, not a reason to leave the question open.
#define L2_LOCK(tier)                                                  \
    const int64_t _t_before = ggml_time_us();                          \
    std::unique_lock<std::mutex> lk((tier).mtx);                       \
    (tier).t_lock_us += ggml_time_us() - _t_before;                    \
    (tier).n_lock_ops++;

// The pinned allocation belongs to `buf` and frees itself; the fallback came from
// moe_aligned_alloc and does not.
llama_moe_stream_l2::~llama_moe_stream_l2() {
    if (base != nullptr && !buf) {
        moe_aligned_free(base);
    }
}

const uint8_t * llama_moe_stream_l2::find(uint16_t file_idx, size_t offs, size_t len, size_t * out_slot) {
    if (base == nullptr) {
        return nullptr;
    }
    const uint64_t key = make_key(file_idx, offs);

    L2_LOCK(*this);
    auto it = index.find(key);
    if (it == index.end()) {
        return nullptr;
    }
    entry & e = entries[it->second];
    // A LOADING entry is NOT a hit. Its bytes are not there yet, and handing them out would upload
    // whatever the slot last held - the one defect this tier could introduce that no throughput
    // number would ever reveal, because a wrong expert is still a fast expert.
    if (e.st != RESIDENT || e.len != len) {
        return nullptr;
    }
    // Pin it for the duration of the caller's upload. See the header for what happened without it.
    e.pins++;
    e.ref = 1;
    n_hit++;
    *out_slot = it->second;
    return base + it->second*slot_stride + e.head;
}

void llama_moe_stream_l2::release(size_t slot) {
    L2_LOCK(*this);
    entry & e = entries[slot];
    if (e.pins > 0) {
        e.pins--;
    }
}

uint8_t * llama_moe_stream_l2::reserve(uint16_t file_idx, size_t offs, size_t len, size_t * out_slot) {
    if (base == nullptr || len + 2*MOE_STREAM_DIRECT_ALIGN > slot_stride) {
        return nullptr;
    }
    const uint64_t key = make_key(file_idx, offs);

    L2_LOCK(*this);
    if (index.find(key) != index.end()) {
        return nullptr; // resident, or another worker is already filling it
    }

    // CLOCK (second chance): a slab hit since the hand last passed it survives one sweep. Two
    // sweeps at most, because the first clears every reference bit it walks over.
    const size_t start = next_victim;
    const size_t sweep = clock ? 2*n_entries : n_entries;
    for (size_t i = 0; i < sweep; i++) {
        const size_t s = (start + i) % n_entries;
        entry & e = entries[s];
        if (e.st == LOADING || e.pins > 0) {
            // LOADING: a worker is filling it. pins > 0: a worker is uploading OUT of it. Both are
            // reads in flight against this memory, and taking it would corrupt one of them.
            continue;
        }
        if (e.st == RESIDENT) {
            if (clock && e.ref) {
                e.ref = 0;
                continue;
            }
            index.erase(e.key);
            n_evict++;
        }
        e.key  = key;
        e.len  = (uint32_t) len;
        e.head = 0;
        e.pins = 0;
        e.ref  = 0;
        e.st   = LOADING;
        index[key]  = s;
        next_victim = (s + 1) % n_entries;
        *out_slot   = s;
        return base + s*slot_stride;
    }
    return nullptr; // every slot is in flight
}

void llama_moe_stream_l2::commit(size_t slot, size_t head) {
    L2_LOCK(*this);
    entry & e = entries[slot];
    e.head = (uint32_t) head;
    e.st   = RESIDENT;
    n_fill++;
}

void llama_moe_stream_l2::abandon(size_t slot) {
    L2_LOCK(*this);
    entry & e = entries[slot];
    index.erase(e.key);
    e.key = UINT64_MAX;
    e.st  = FREE;
}

void llama_moe_stream::alloc_l2(ggml_backend_buffer_type_t host_buft) {
    if (l2_gib == 0 || max_nb_expert == 0) {
        return;
    }
    // Every slot takes the largest slab plus the direct-I/O head/tail slack, so any weight fits
    // any slot. Uniform slots waste the difference between the widest and the narrowest weight
    // (0.5 MiB of 3.06 on the shipped model) and buy an allocator that cannot fragment.
    const size_t stride = max_nb_expert + 2*MOE_STREAM_DIRECT_ALIGN;
    const size_t want   = l2_gib * 1024ull * 1024ull * 1024ull;
    const size_t n_l2   = want / stride;
    if (n_l2 == 0) {
        LLAMA_LOG_WARN("%s: L2 of %zu GiB is smaller than one %zu-byte slot - tier not created\n",
                __func__, l2_gib, stride);
        return;
    }

    auto tier = std::make_unique<llama_moe_stream_l2>();
    tier->slot_stride = stride;
    tier->n_entries   = n_l2;

    if (const char * s = std::getenv("LLAMA_MOE_STREAM_L2_CLOCK")) {
        tier->clock = std::strtol(s, nullptr, 10) != 0;
    }

    const size_t bytes = n_l2 * stride;

    if (host_buft != nullptr) {
        ggml_backend_buffer_t b = ggml_backend_buft_alloc_buffer(host_buft, bytes);
        if (b != nullptr) {
            tier->buf.reset(b);
            tier->base   = (uint8_t *) ggml_backend_buffer_get_base(b);
            tier->pinned = true;
        }
    }
    if (tier->base == nullptr) {
        // Not a failure: the tier still removes the SSD read, it just uploads at pageable rates.
        // Which one is in force is printed, because a run that silently fell back would report the
        // slower number as this tier's.
        tier->base = (uint8_t *) moe_aligned_alloc(bytes);
        if (tier->base == nullptr) {
            LLAMA_LOG_WARN("%s: could not allocate %.2f GiB for the L2 tier - tier not created\n",
                    __func__, bytes/1024.0/1024.0/1024.0);
            return;
        }
    }

    tier->entries.resize(n_l2);
    tier->index.reserve(n_l2*2);

    // WARN and not INFO, and that is a deliberate misuse of the level. At the default verbosity
    // llama-server prints no INFO from this library at all - measured 2026-08-09, the whole load
    // block is absent while WARN lines come through. This message reports that GiB-scale memory
    // has been PAGE-LOCKED and is gone from the rest of the machine until the process exits.
    // A user who passes --moe-stream-l2 and sees nothing cannot tell it took effect from a typo,
    // and silently holding 32 GiB is not something to find out from the task manager.
    LLAMA_LOG_WARN("%s: MoE L2 host tier = %.2f GiB %s, %zu slots of %zu bytes, eviction %s\n",
            __func__, bytes/1024.0/1024.0/1024.0,
            tier->pinned ? "PINNED (page-locked, unavailable to the rest of the system)"
                         : "PAGEABLE (no host buffer type - uploads stay at pageable rates)",
            n_l2, stride, tier->clock ? "CLOCK" : "FIFO");

    l2 = std::move(tier);
}

bool llama_moe_stream_layer::matches(const ggml_tensor * gate, const ggml_tensor * up,
                                     const ggml_tensor * down, const ggml_tensor * gate_up) const {
    auto is_cache = [this](const ggml_tensor * t) {
        for (const auto & w : weights) {
            if (w.cache == t) {
                return true;
            }
        }
        return false;
    };

    size_t n = 0;
    for (const ggml_tensor * t : { gate, up, down, gate_up }) {
        if (t == nullptr) {
            continue;
        }
        if (!is_cache(t)) {
            return false;
        }
        n++;
    }

    return n > 0 && n == weights.size();
}

// sizes the per-layer table and clamps the I/O thread count; workers are spawned lazily on first use
llama_moe_stream::llama_moe_stream(uint32_t n_layer, uint32_t n_slots, int32_t n_io_threads, bool direct) : n_slots(n_slots) {
    layers.resize(n_layer);

    this->n_io_threads = n_io_threads <= 0 ? MOE_STREAM_IO_THREADS_DEFAULT : n_io_threads;
    this->n_io_threads = std::min<int32_t>(this->n_io_threads, MOE_STREAM_IO_THREADS_MAX);

    debug         = std::getenv("LLAMA_MOE_STREAM_DEBUG") != nullptr;
    use_direct_io = direct;

    // Router bias toward resident experts. Read once here rather than per call so a run cannot
    // change policy halfway through and leave counters that describe two different experiments.
    // Host-RAM tier size in whole GiB. Read once here, same reasoning as the bias below: a run
    // must not change shape halfway through and leave counters describing two experiments. The
    // tier itself is allocated in alloc_bufs(), where max_nb_expert is finally known.
    if (const char * s = std::getenv("LLAMA_MOE_STREAM_L2_GIB")) {
        const long v = std::strtol(s, nullptr, 10);
        l2_gib = v > 0 ? (size_t) v : 0;
    }

    if (const char * s = std::getenv("LLAMA_MOE_STREAM_ROUTE_BIAS")) {
        route_bias = std::strtof(s, nullptr);
        if (route_bias != 0.0f) {
            LLAMA_LOG_INFO("%s: MoE expert streaming: router biased toward resident experts by %+.4f\n",
                    __func__, route_bias);
        }
    }
}

// stop and join the I/O workers before the cache buffers and files they use are destroyed
llama_moe_stream::~llama_moe_stream() {
    {
        std::lock_guard<std::mutex> lock(mtx);
        shutting_down = true;
        q_demand.clear();
    }
    cv_work.notify_all();
    for (auto & w : workers) {
        w.join();
    }
}

ggml_tensor * llama_moe_stream::create_cache_tensor(
        int32_t il, ggml_backend_buffer_type_t buft, const ggml_tensor * meta,
        uint16_t file_idx, size_t offs) {
    GGML_ASSERT(il >= 0 && (size_t) il < layers.size());
    GGML_ASSERT(ggml_is_contiguous(meta));
    GGML_ASSERT(meta->ne[2] > 0 && meta->ne[3] == 1);

    const uint32_t n_expert  = meta->ne[2];
    const size_t   nb_expert = ggml_nbytes(meta) / n_expert;
    GGML_ASSERT(nb_expert * n_expert == ggml_nbytes(meta));
    GGML_ASSERT(n_slots > 0 && n_slots < n_expert);

    ggml_context * ctx = nullptr;
    for (auto & [cur_buft, cur_ctx] : ctxs) {
        if (cur_buft == buft) {
            ctx = cur_ctx.get();
            break;
        }
    }
    if (ctx == nullptr) {
        ggml_init_params params = {
            /*.mem_size   =*/ ggml_tensor_overhead()*(layers.size()*4 + 1),
            /*.mem_buffer =*/ NULL,
            /*.no_alloc   =*/ true,
        };
        ctx = ggml_init(params);
        if (ctx == nullptr) {
            throw std::runtime_error("failed to create ggml context for MoE expert streaming");
        }
        ctxs.emplace_back(buft, ctx);
    }

    ggml_tensor * cache = ggml_new_tensor_3d(ctx, meta->type, meta->ne[0], meta->ne[1], n_slots);
    ggml_format_name(cache, "%s.stream_cache", meta->name);
    GGML_ASSERT(ggml_nbytes(cache) == nb_expert * n_slots);

    auto & sl = layers[il];
    if (!sl) {
        sl = std::make_unique<llama_moe_stream_layer>();
        sl->mgr      = this;
        sl->il       = il;
        sl->n_expert = n_expert;
        sl->n_slots  = n_slots;
        sl->slot_expert  .resize(n_slots, -1);
        sl->slot_state   .resize(n_slots, LLAMA_MOE_STREAM_SLOT_EMPTY);
        sl->slot_claimed .resize(n_slots, 0);
        sl->slot_parts_left.resize(n_slots, 0);
        sl->slot_gen     .resize(n_slots, 0);
        sl->slot_last_use.resize(n_slots, 0);
        sl->route_hotness.resize(n_expert, 0);
        sl->route_total.resize(n_expert, 0);
        sl->seen         .resize(n_expert, 0);
        sl->keep         .resize(n_slots, 0);
    }
    GGML_ASSERT(sl->n_expert == n_expert);

    sl->weights.push_back({ cache, file_idx, offs, nb_expert });

    max_nb_expert = std::max(max_nb_expert, nb_expert);

    return cache;
}

void llama_moe_stream::alloc_bufs(bool no_alloc) {
    for (auto & [buft, ctx_ptr] : ctxs) {
        ggml_context * ctx = ctx_ptr.get();
        if (ggml_get_first_tensor(ctx) == nullptr) {
            continue;
        }

        ggml_backend_buffer_t buf;
        if (no_alloc) {
            buf = ggml_backend_buft_alloc_buffer(buft, /*size =*/ 0); // dummy buffer
            for (ggml_tensor * t = ggml_get_first_tensor(ctx); t != nullptr; t = ggml_get_next_tensor(ctx, t)) {
                t->buffer = buf;
            }
        } else {
            buf = ggml_backend_alloc_ctx_tensors_from_buft(ctx, buft);
        }
        if (buf == nullptr) {
            throw std::runtime_error(format("unable to allocate %s buffer for MoE expert streaming", ggml_backend_buft_name(buft)));
        }
        ggml_backend_buffer_set_usage(buf, GGML_BACKEND_BUFFER_USAGE_WEIGHTS);
        bufs.emplace_back(buf);

        LLAMA_LOG_INFO("%s: %12s expert cache size = %8.2f MiB (%u slots per layer)\n",
                __func__, ggml_backend_buffer_name(buf), ggml_backend_buffer_get_size(buf) / 1024.0 / 1024.0, n_slots);
    }

    // The host tier is allocated here and not in the constructor: max_nb_expert only exists once
    // every weight has been registered, and the slot size is derived from it.
    if (!no_alloc && l2_gib > 0) {
        // Page-locked memory belongs to whichever backend owns the cache tensors, so the device
        // comes from their own buffer type rather than being assumed to be device 0.
        ggml_backend_buffer_type_t host_buft = nullptr;
        for (auto & [buft, ctx_ptr] : ctxs) {
            ggml_backend_dev_t dev = ggml_backend_buft_get_device(buft);
            if (dev != nullptr) {
                host_buft = ggml_backend_dev_host_buffer_type(dev);
                if (host_buft != nullptr) {
                    break;
                }
            }
        }
        alloc_l2(host_buft);
    }
}

void llama_moe_stream::open_files(const std::vector<std::string> & paths) {
    for (const auto & path : paths) {
        if (path.empty()) {
            throw std::runtime_error("MoE expert streaming requires a file-based model (not a stream/file descriptor)");
        }
    }

    auto open_all = [&](bool direct) {
        files.clear();
        for (const auto & path : paths) {
            files.emplace_back(new llama_file(path.c_str(), "rb", direct));
        }
    };

    open_all(use_direct_io);

    // fall back to buffered when O_DIRECT is unusable: either the open did not honor it (macOS,
    // Windows, unsupported filesystems), or it opened but a probe read fails (some network/overlay
    // filesystems accept the flag then reject aligned reads). reopening is needed because O_DIRECT
    // is a property of the fd. done here, single-threaded, before any worker starts.
    if (use_direct_io) {
        bool ok = !files.empty() && files.front()->has_direct_io();
        if (ok) {
            uint8_t * probe = (uint8_t *) moe_aligned_alloc(MOE_STREAM_DIRECT_ALIGN);
            GGML_ASSERT(probe != nullptr);
            // -1: single-threaded, before any worker starts, so it reads through the shared handle
            ok = llama_moe_stream_pread(*files.front(), probe, MOE_STREAM_DIRECT_ALIGN, 0, /*direct =*/ true, /*worker_id =*/ -1) != nullptr;
            moe_aligned_free(probe);
        }
        if (!ok) {
            LLAMA_LOG_WARN("%s: O_DIRECT not usable, falling back to buffered streaming reads\n", __func__);
            use_direct_io = false;
            open_all(false);
        }
    }

    if (use_direct_io) {
        LLAMA_LOG_INFO("%s: MoE expert streaming uses O_DIRECT (page cache bypassed)\n", __func__);
    }

    // one token drives ~one remap per streamed layer, so decaying every 64 tokens is
    //   64 * n_streamed_layers remap calls (computed once here, off the hot path)
    int64_t n_streamed = 0;
    for (const auto & sl : layers) {
        n_streamed += sl != nullptr;
    }
    hot_decay_interval = MOE_STREAM_HOT_DECAY_TOKENS * n_streamed;
}

// spawn the I/O thread pool on first use (from the remap callback, under mtx)
void llama_moe_stream::start_workers_locked() {
    if (workers_started) {
        return;
    }
    workers_started = true;
    workers.reserve(n_io_threads);
    for (int32_t i = 0; i < n_io_threads; i++) {
        workers.emplace_back([this, i]() { worker_loop((int) i); });
    }
}

// I/O worker: pops a reserved load, reads its expert slab(s) from the GGUF file into the cache
// slot, and marks the slot RESIDENT (or flags load_failed); stale/duplicate items are skipped
void llama_moe_stream::worker_loop(int worker_id) {
    // page-aligned staging (Metal private buffers require page-aligned source + page-multiple
    // length; O_DIRECT needs the extra head/tail slack for its aligned reads)
    uint8_t * staging = (uint8_t *) moe_aligned_alloc(max_nb_expert + 2*MOE_STREAM_DIRECT_ALIGN);
    GGML_ASSERT(staging != nullptr);

    std::unique_lock<std::mutex> lk(mtx);
    while (true) {
        cv_work.wait(lk, [&]{ return shutting_down || !q_demand.empty(); });
        if (shutting_down) {
            break;
        }

        llama_moe_stream_work w = q_demand.front();
        q_demand.pop_front();

        auto & sl = *w.sl;
        const uint8_t bit = (w.weight >= 0 && w.weight < 8) ? (uint8_t) (1u << w.weight) : 0u;
        if (w.gen != sl.slot_gen[w.slot] ||
            sl.slot_state[w.slot] != LLAMA_MOE_STREAM_SLOT_LOADING ||
            sl.slot_expert[w.slot] != w.expert ||
            (bit && (sl.slot_claimed[w.slot] & bit))) {
            // Stale (the slot was re-reserved under us) or a duplicate item for a
            // tensor another worker already owns. The claim is per TENSOR now; a
            // per-slot flag would make the first worker lock its colleagues out of
            // the very parallelism this split exists to create.
            continue;
        }
        sl.slot_claimed[w.slot] |= bit;

        lk.unlock();

        bool ok = true;
        {
            const auto & wt = sl.weights[w.weight];
            const size_t offs = wt.offs + (size_t) w.expert*wt.nb_expert;

            // Two ways this slab can reach the upload, and BOTH have to keep their memory alive
            // until the upload is done. A slot that is an eviction candidate while it is being
            // read gets a different expert written into it mid-upload, and the model receives
            // half of each. That is not a hypothetical: it is what this code did on 2026-08-09.
            size_t pinned_slot = SIZE_MAX; // hit  -> released after the upload
            size_t filled_slot = SIZE_MAX; // fill -> committed after the upload
            size_t filled_head = 0;        // payload offset inside that slot

            // 1. host tier. A hit skips the drive entirely and uploads from page-locked memory.
            const uint8_t * data = l2 ? l2->find(wt.file_idx, offs, wt.nb_expert, &pinned_slot) : nullptr;

            if (data == nullptr) {
                // 2. drive. The read lands DIRECTLY in an L2 slot when one is free, so the tier
                //    fills for nothing: same read, same bytes, kept instead of discarded. Falls
                //    back to this worker's own staging buffer when the tier is off, already owns
                //    the slab, or has every slot in flight.
                size_t    slot_l2 = 0;
                uint8_t * dst     = l2 ? l2->reserve(wt.file_idx, offs, wt.nb_expert, &slot_l2) : nullptr;

                data = llama_moe_stream_pread(*files[wt.file_idx], dst ? dst : staging,
                        wt.nb_expert, offs, use_direct_io, worker_id);

                if (dst != nullptr) {
                    if (data == nullptr) {
                        // A failed read must not leave a slot claiming to hold this slab, or the
                        // next hit uploads uninitialised memory and nothing ever says so.
                        l2->abandon(slot_l2);
                    } else {
                        // NOT committed yet. The slot stays LOADING - and therefore off the
                        // eviction list - until its bytes have actually left for the GPU.
                        filled_slot = slot_l2;
                        filled_head = (size_t) (data - dst);
                    }
                }
            }

            if (data == nullptr) {
                ok = false;
            } else {
                // The READ above is the parallel part and stays outside the lock;
                // only the upload is serialised. See upload_mtx in the header for
                // the measurement that made this necessary.
                {
                    std::lock_guard<std::mutex> up(upload_mtx);
                    ggml_backend_tensor_set(wt.cache, data, (size_t) w.slot*wt.nb_expert, wt.nb_expert);
                }
                // Only now is the memory free to be reused by someone else.
                if (filled_slot != SIZE_MAX) {
                    l2->commit(filled_slot, filled_head);
                }
            }
            if (pinned_slot != SIZE_MAX) {
                l2->release(pinned_slot);
            }
        }

        lk.lock();

        // Re-check the generation: this worker held no lock while reading, and the
        // slot may have been evicted and re-reserved for another expert meanwhile.
        // Decrementing that generation's counter would publish a slot whose tensors
        // were never all written.
        if (w.gen != sl.slot_gen[w.slot]) {
            cv_done.notify_all();
            continue;
        }

        sl.slot_claimed[w.slot] &= (uint8_t) ~bit;
        if (!ok) {
            load_failed = true;
        } else if (sl.slot_parts_left[w.slot] > 0 && --sl.slot_parts_left[w.slot] == 0) {
            // RESIDENT only when the LAST tensor of this slot has landed. Publishing
            // at the first one would hand out a slot that is part garbage.
            sl.slot_state[w.slot] = LLAMA_MOE_STREAM_SLOT_RESIDENT;
        }
        cv_done.notify_all();
    }
    lk.unlock();

    moe_aligned_free(staging);
}

// least valuable evictable slot: empty first, then coldest resident (min route hotness, oldest use
// as tiebreak); LOADING and keep slots are never candidates. returns -1 when no candidate exists
int32_t llama_moe_stream::pick_victim_locked(llama_moe_stream_layer & sl, const uint8_t * keep) const {
    int32_t v = -1;

    for (uint32_t s = 0; s < sl.n_slots; s++) {
        if ((keep && keep[s]) || sl.slot_state[s] == LLAMA_MOE_STREAM_SLOT_LOADING) {
            continue;
        }
        if (sl.slot_state[s] == LLAMA_MOE_STREAM_SLOT_EMPTY) {
            return s;
        }
        if (v < 0) {
            v = s;
            continue;
        }
        const uint32_t hs = sl.route_hotness[sl.slot_expert[s]];
        const uint32_t hv = sl.route_hotness[sl.slot_expert[v]];
        if (hs < hv || (hs == hv && sl.slot_last_use[s] < sl.slot_last_use[v])) {
            v = s;
        }
    }

    return v;
}

// bind expert -> slot and mark it LOADING: evict the slot's prior occupant, bump slot_gen (so any
// in-flight load for the old occupant is recognized as stale), and update the expert_slot index
void llama_moe_stream::reserve_slot_locked(llama_moe_stream_layer & sl, int32_t expert, int32_t slot) {
    if (sl.slot_expert[slot] >= 0) {
        if (debug) {
            LLAMA_LOG_DEBUG("%s: layer %d: evict expert %d from slot %d\n", __func__, sl.il, sl.slot_expert[slot], slot);
        }
        sl.expert_slot.erase(sl.slot_expert[slot]);
    }

    sl.slot_expert[slot] = expert;
    sl.slot_state[slot]  = LLAMA_MOE_STREAM_SLOT_LOADING;
    sl.slot_gen[slot]++;
    sl.slot_last_use[slot] = ++sl.use_counter;
    sl.expert_slot[expert] = slot;
    sl.seen[expert] = 1;
    // A reservation starts owing one read per weight tensor, and owns none of them
    // yet. Both are reset HERE rather than at enqueue: bumping slot_gen above has
    // just invalidated every in-flight part of the previous occupant, and a stale
    // worker finishing afterwards must not decrement this generation's counter.
    sl.slot_parts_left[slot] = (uint16_t) sl.weights.size();
    sl.slot_claimed[slot]    = 0;
}

// Queue one read per weight tensor of this slot. THE POINT OF THE SPLIT: a single
// item per expert meant one worker read 2-3 tensors in a for loop, strictly one at
// a time, which is the whole reason the drive saw a queue depth of 1.6. One item
// per tensor lets that many workers pull in parallel out of the same pool.
void llama_moe_stream::enqueue_slot_locked(llama_moe_stream_layer & sl, int32_t expert, int32_t slot) {
    const int32_t n = (int32_t) sl.weights.size();
    for (int32_t w = 0; w < n; w++) {
        q_demand.push_back({ &sl, expert, slot, sl.slot_gen[slot], w });
    }
    // notify_all, not notify_one: several items just arrived and one woken worker
    // would take one and leave the rest queued behind a sleeping pool - exactly the
    // serialisation this change exists to remove.
    cv_work.notify_all();
}

size_t llama_moe_stream::size_bufs() const {
    size_t size = 0;
    for (const auto & buf : bufs) {
        size += ggml_backend_buffer_get_size(buf.get());
    }
    return size;
}

// EVERY NUMBER HERE IS CUMULATIVE over the lifetime of this manager, i.e. over the model
// instance, i.e. over the server process. There is no reset and none is wanted: a reset is a
// second piece of state that two callers can disagree about, and the request-local figure is a
// difference between two consecutive blocks, which the reader can form and the writer cannot.
// A restarted server starts the counts at zero because the manager is new, not because anything
// was cleared.
void llama_moe_stream::print_stats(const char * role) const {
    std::lock_guard<std::mutex> lock(mtx);

    // "target: " / "drafter: ", or nothing at all. Built once and passed down so that every line
    // of one block - including the locality lines below - carries the same role token; a block
    // whose first line is labelled and whose rest is not cannot be attributed when two models
    // print into one log.
    char pfx[32];
    if (role && role[0]) {
        snprintf(pfx, sizeof(pfx), "%s: ", role);
    } else {
        pfx[0] = '\0';
    }

    const int64_t n_touched = stats.n_hit + stats.n_miss;
    LLAMA_LOG_WARN("%s: %smoe stream: remap calls = %" PRId64 ", expert hits = %" PRId64 ", misses = %" PRId64 " (%" PRId64 " cold), hit rate = %.2f%%\n",
            __func__, pfx, stats.n_calls, stats.n_hit, stats.n_miss, stats.n_miss_cold,
            n_touched > 0 ? 100.0*stats.n_hit/n_touched : 0.0);
    LLAMA_LOG_WARN("%s: %smoe stream: load stall = %.2f ms total (%.3f ms per remap call)\n",
            __func__, pfx, stats.t_stall_us/1000.0, stats.n_calls > 0 ? stats.t_stall_us/1000.0/stats.n_calls : 0.0);
    LLAMA_LOG_INFO("%s: %smoe stream: slot wait = %.2f ms total over %" PRId64 " waits (%.1f%% of the two stalls)\n",
            __func__, pfx, stats.t_victim_us/1000.0, stats.n_victim_waits,
            (stats.t_victim_us + stats.t_stall_us) > 0 ? 100.0*stats.t_victim_us/(stats.t_victim_us + stats.t_stall_us) : 0.0);
    if (l2) {
        // Counted in SLABS, not experts: one expert is two or three weight tensors and each is
        // looked up on its own, so this denominator is NOT stats.n_miss and must not be read as
        // if it were. An L2 hit is a VRAM-tier miss that never reached the drive.
        std::lock_guard<std::mutex> l2lock(l2->mtx);
        const int64_t n_slab = l2->n_hit + l2->n_fill;
        LLAMA_LOG_INFO("%s: moe stream: L2 host tier = %.2f GiB %s, %zu slots\n",
                __func__, l2->n_entries*l2->slot_stride/1024.0/1024.0/1024.0,
                l2->pinned ? "pinned" : "PAGEABLE", l2->n_entries);
        LLAMA_LOG_WARN("%s: moe stream: L2 slab hits = %" PRId64 ", fills = %" PRId64 ", evictions = %" PRId64 ", hit rate = %.2f%%\n",
                __func__, l2->n_hit, l2->n_fill, l2->n_evict,
                n_slab > 0 ? 100.0*l2->n_hit/n_slab : 0.0);
        // The question this run exists to answer: the tier removed 196 s of drive wait and the
        // wall clock did not move, so where did the time go? This is the wait at THIS lock. If it
        // is small, the answer is elsewhere and the mutex theory is wrong.
        LLAMA_LOG_INFO("%s: moe stream: L2 lock wait = %.2f ms total over %" PRId64 " ops (%.3f us each)\n",
                __func__, l2->t_lock_us/1000.0, l2->n_lock_ops,
                l2->n_lock_ops > 0 ? (double) l2->t_lock_us/l2->n_lock_ops : 0.0);
    }
    if (stats.n_wave_calls > 0) {
        LLAMA_LOG_INFO("%s: %smoe stream: waves = %" PRId64 " (%" PRId64 " non-empty), preloads issued = %" PRId64 " (ready on arrival = %" PRId64 "), wave stall = %.2f ms\n",
                __func__, pfx, stats.n_wave_calls, stats.n_waves_run, stats.n_preload_issued, stats.n_preload_ready, stats.t_stall_wave_us/1000.0);
    }

    print_locality(pfx);
}

// Expert locality: how concentrated the routing is. A cache only pays if a minority of experts
// carries a majority of the selections; if routing is flat, no cache size helps and the whole
// streaming lever is bounded by disk bandwidth alone.
//
// Reported per layer as the share of experts needed to cover 50/80/95 % of that layer's selections,
// then aggregated. The Gini coefficient is printed beside it because the coverage figures alone
// cannot tell a mildly skewed distribution from a bimodal one.
void llama_moe_stream::print_locality(const char * pfx) const {
    std::vector<double> cov50, cov80, cov95, ginis;
    uint64_t n_layers_seen = 0;

    for (const auto & sl : layers) {
        if (!sl || sl->route_total.empty()) {
            continue;
        }
        std::vector<uint64_t> c = sl->route_total;
        uint64_t total = 0;
        for (const uint64_t v : c) {
            total += v;
        }
        if (total == 0) {
            continue;
        }
        n_layers_seen++;

        std::sort(c.begin(), c.end(), std::greater<uint64_t>());
        const double n = (double) c.size();

        uint64_t acc = 0;
        double c50 = 0, c80 = 0, c95 = 0;
        for (size_t i = 0; i < c.size(); i++) {
            acc += c[i];
            const double share = (double) acc/total;
            const double frac  = (double) (i + 1)/n;
            if (c50 == 0 && share >= 0.50) { c50 = frac; }
            if (c80 == 0 && share >= 0.80) { c80 = frac; }
            if (c95 == 0 && share >= 0.95) { c95 = frac; }
        }
        cov50.push_back(c50);
        cov80.push_back(c80);
        cov95.push_back(c95);

        // Gini; 0 = every expert used equally, 1 = one expert takes all.
        //
        // The textbook form (2*sum(i*x_i))/(n*sum(x)) - (n+1)/n assumes ASCENDING order. c is sorted
        // descending for the coverage figures above, which flips the sign, so the terms are swapped
        // here rather than sorting the vector twice.
        double weighted = 0;
        for (size_t i = 0; i < c.size(); i++) {
            weighted += (double) (i + 1)*c[i];
        }
        ginis.push_back((n + 1.0)/n - (2.0*weighted)/(n*total));
    }

    if (n_layers_seen == 0) {
        return;
    }

    auto mean = [](const std::vector<double> & v) {
        double s = 0;
        for (const double x : v) { s += x; }
        return s/v.size();
    };

    LLAMA_LOG_INFO("%s: %smoe stream: expert locality over %" PRIu64 " layers, %u experts each\n",
            __func__, pfx, n_layers_seen, layers.empty() || !layers[0] ? 0 : layers[0]->n_expert);
    LLAMA_LOG_INFO("%s: %smoe stream:   50%% of selections covered by %.1f%% of experts, 80%% by %.1f%%, 95%% by %.1f%%\n",
            __func__, pfx, 100.0*mean(cov50), 100.0*mean(cov80), 100.0*mean(cov95));
    LLAMA_LOG_INFO("%s: %smoe stream:   Gini = %.3f (0 = flat, 1 = one expert takes everything)\n",
            __func__, pfx, mean(ginis));
}

// custom-op callback (single-threaded on ith 0): given the router's expert ids, ensure every touched
// expert is resident - reserving cache slots and demand-loading misses, stalling until they commit -
// then rewrite each id to its cache slot. this only relabels ids, so the same experts are computed
// in the same order; the result matches a non-streamed run (bit-exact when both paths use the same
// kernels, as on CUDA; a CPU build that repacks the non-streamed weights can differ in the last bits).
// Adds mgr->route_bias to the selection logits of the experts this layer currently holds. Runs on
// the CPU right before top-k, under the same lock the remap takes, so the residency it reads is the
// one the remap acts on a moment later.
//
// Masked-out entries stay masked: group masking writes -inf and -inf + bias is still -inf, but the
// isfinite guard states that instead of relying on it. A bias of 0 copies the input unchanged, which
// is what makes the off-case verifiable against a run without the hook at all.
void llama_moe_stream_route_bias(ggml_tensor * dst, const ggml_tensor * a, int ith, int nth, void * userdata) {
    GGML_UNUSED(nth);
    if (ith != 0) {
        return;
    }

    auto * sl  = (llama_moe_stream_layer *) userdata;
    auto * mgr = sl->mgr;

    GGML_ASSERT(a->type   == GGML_TYPE_F32);
    GGML_ASSERT(dst->type == GGML_TYPE_F32);
    GGML_ASSERT(ggml_are_same_shape(a, dst));
    GGML_ASSERT(ggml_is_contiguous(a));
    GGML_ASSERT(ggml_is_contiguous(dst));

    memcpy(dst->data, a->data, ggml_nbytes(a));

    const float bias = mgr->route_bias;
    if (bias == 0.0f) {
        return;
    }

    const int64_t n_expert = a->ne[0];
    const int64_t n_tokens = ggml_nelements(a) / n_expert;

    float * out = (float *) dst->data;

    std::unique_lock<std::mutex> lk(mgr->mtx);

    for (int64_t t = 0; t < n_tokens; t++) {
        float * row = out + t*n_expert;
        for (const auto & kv : sl->expert_slot) {
            const int32_t e = kv.first;
            if (e >= 0 && e < n_expert && std::isfinite(row[e])) {
                row[e] += bias;
            }
        }
    }
}

void llama_moe_stream_remap(ggml_tensor * dst, const ggml_tensor * a, int ith, int nth, void * userdata) {
    GGML_UNUSED(nth);
    if (ith != 0) {
        return;
    }

    auto * sl  = (llama_moe_stream_layer *) userdata;
    auto * mgr = sl->mgr;

    GGML_ASSERT(a->type == GGML_TYPE_I32);
    GGML_ASSERT(ggml_is_contiguous(a));
    GGML_ASSERT(ggml_are_same_shape(a, dst));

    const int64_t n = ggml_nelements(a);

    const int32_t * ids = (const int32_t *) a->data;
          int32_t * out = (int32_t *) dst->data;

    std::unique_lock<std::mutex> lk(mgr->mtx);

    if (mgr->load_failed) {
        GGML_ABORT("MoE expert streaming: expert load failed (I/O error)");
    }

    mgr->stats.n_calls++;
    mgr->start_workers_locked();

    // distinct experts touched by this ubatch, in first-use order
    sl->touched.assign(sl->n_expert, 0);
    sl->uniq.clear();
    for (int64_t i = 0; i < n; i++) {
        const int32_t e = ids[i];
        GGML_ASSERT(e >= 0 && (uint32_t) e < sl->n_expert);
        if (!sl->touched[e]) {
            sl->touched[e] = 1;
            sl->uniq.push_back(e);
        }
    }

    if (sl->uniq.size() > sl->n_slots) {
        GGML_ABORT("MoE expert streaming: layer %d needs %zu distinct experts but the cache has only %u slots; "
                   "increase --moe-stream-cache or reduce the ubatch size (-ub)",
                sl->il, sl->uniq.size(), sl->n_slots);
    }

    // route hotness for eviction; halved periodically so a formerly-hot expert ages out
    for (const int32_t e : sl->uniq) {
        sat_inc(sl->route_hotness[e]);
        sl->route_total[e]++;
    }
    if (mgr->hot_decay_interval > 0 && mgr->stats.n_calls % mgr->hot_decay_interval == 0) {
        for (auto & sl2 : mgr->layers) {
            if (sl2) {
                for (auto & h : sl2->route_hotness) {
                    h >>= 1;
                }
            }
        }
    }

    // classify the touched experts; reserve and enqueue demand loads in deterministic order
    std::fill(sl->keep.begin(), sl->keep.end(), 0);
    sl->demand_slots.clear();

    bool waited = false;
    for (const int32_t e : sl->uniq) {
        const auto it = sl->expert_slot.find(e);
        if (it != sl->expert_slot.end()) {
            const int32_t s = it->second;
            if (sl->slot_state[s] == LLAMA_MOE_STREAM_SLOT_LOADING) {
                mgr->enqueue_slot_locked(*sl, e, s);
                waited = true;
            }
            mgr->stats.n_hit++;
            sl->keep[s] = 1;
            sl->demand_slots.push_back(s);
        } else {
            int32_t v = mgr->pick_victim_locked(*sl, sl->keep.data());
            if (v < 0) {
                // every allowed slot is loading; wait for a commit and retry.
                //
                // timed separately from t_stall_us on purpose: this wait happens BEFORE the demand
                // loads are issued, so the stall timer below cannot cover it. the two answer
                // different questions - t_stall_us is time spent waiting on the disk, this is time
                // spent waiting for the cache to give a slot back. they call for opposite fixes.
                const int64_t t0 = ggml_time_us();
                do {
                    mgr->cv_done.wait(lk);
                    if (mgr->load_failed) {
                        GGML_ABORT("MoE expert streaming: expert load failed (I/O error)");
                    }
                } while ((v = mgr->pick_victim_locked(*sl, sl->keep.data())) < 0);
                mgr->stats.t_victim_us += ggml_time_us() - t0;
                mgr->stats.n_victim_waits++;
            }
            if (!sl->seen[e]) {
                mgr->stats.n_miss_cold++;
            }
            mgr->reserve_slot_locked(*sl, e, v);
            mgr->enqueue_slot_locked(*sl, e, v);
            mgr->stats.n_miss++;
            waited = true;
            sl->keep[v] = 1;
            sl->demand_slots.push_back(v);
        }
    }

    if (waited) {
        const int64_t t0 = ggml_time_us();
        mgr->cv_done.wait(lk, [&]{
            if (mgr->load_failed) {
                return true;
            }
            for (const int32_t s : sl->demand_slots) {
                if (sl->slot_state[s] != LLAMA_MOE_STREAM_SLOT_RESIDENT) {
                    return false;
                }
            }
            return true;
        });
        if (mgr->load_failed) {
            GGML_ABORT("MoE expert streaming: expert load failed (I/O error)");
        }
        mgr->stats.t_stall_us += ggml_time_us() - t0;
    }

    for (int64_t i = 0; i < n; i++) {
        const int32_t s = sl->expert_slot.at(ids[i]);
        sl->slot_last_use[s] = ++sl->use_counter;
        out[i] = s;
    }
}

// stable per-wave userdata; grows lazily and records the per-wave expert capacity (set at build)
llama_moe_stream_wave * llama_moe_stream_layer::wave_userdata(int32_t wave, uint32_t capacity) {
    GGML_ASSERT(capacity >= 1 && capacity <= n_slots);
    plan_capacity = capacity;
    while ((size_t) wave >= wave_ud.size()) {
        auto ud = std::make_unique<llama_moe_stream_wave>();
        ud->sl   = this;
        ud->wave = (int32_t) wave_ud.size();
        wave_ud.push_back(std::move(ud));
    }
    return wave_ud[wave].get();
}

// wave 0 of a ubatch: record the distinct touched experts (sl.uniq, first-use order) and split them
// into consecutive groups of plan_capacity, one group per wave (sl.expert_wave[e] = e's wave)
void llama_moe_stream::plan_waves_locked(llama_moe_stream_layer & sl, const int32_t * ids, int64_t n) {
    stats.n_calls++;
    start_workers_locked();

    sl.touched.assign(sl.n_expert, 0);
    sl.uniq.clear();
    for (int64_t i = 0; i < n; i++) {
        const int32_t e = ids[i];
        GGML_ASSERT(e >= 0 && (uint32_t) e < sl.n_expert);
        if (!sl.touched[e]) {
            sl.touched[e] = 1;
            sl.uniq.push_back(e);
        }
    }

    GGML_ASSERT(sl.plan_capacity > 0);
    sl.expert_wave.assign(sl.n_expert, 0xff);
    for (size_t i = 0; i < sl.uniq.size(); i++) {
        GGML_ASSERT(i/sl.plan_capacity < 0xff);
        sl.expert_wave[sl.uniq[i]] = (uint8_t) (i/sl.plan_capacity);
    }
    sl.plan_n_waves   = (uint32_t) ((sl.uniq.size() + sl.plan_capacity - 1)/sl.plan_capacity);
    sl.plan_next_wave = 0;
}

// make wave w's expert slice (uniq[w*cap .. +count)) resident, waiting for its loads, and best-effort
// preload the next wave so its loads overlap this wave's compute. leaves sl.demand_slots = this wave's
// slots and sl.plan_pool = the resident parking pool (>= n_ids slots) the emit draws masked pairs from
void llama_moe_stream::stage_wave_locked(std::unique_lock<std::mutex> & lk, llama_moe_stream_layer & sl, int32_t w, uint32_t n_ids) {
    const size_t first = (size_t) w*sl.plan_capacity;
    const size_t count = first < sl.uniq.size() ? std::min<size_t>(sl.plan_capacity, sl.uniq.size() - first) : 0;

    std::fill(sl.keep.begin(), sl.keep.end(), 0);
    sl.demand_slots.clear();

    // a small final wave has fewer than n_ids own slots; borrow the rest from the previous wave's
    //   pool so every token row has n_ids distinct resident parking slots for its masked pairs
    std::vector<int32_t> borrowed;
    if (count < n_ids) {
        GGML_ASSERT(sl.plan_pool.size() >= n_ids - count);
        for (size_t i = 0; i < n_ids - count; i++) {
            borrowed.push_back(sl.plan_pool[i]);
            sl.keep[sl.plan_pool[i]] = 1; // parking slots must survive this wave's loads
        }
    }

    // protect the next wave's already-resident experts so this wave's victims do not evict them
    const size_t nfirst = first + sl.plan_capacity;
    const size_t ncount = nfirst < sl.uniq.size() ? std::min<size_t>(sl.plan_capacity, sl.uniq.size() - nfirst) : 0;
    for (size_t i = nfirst; i < nfirst + ncount; i++) {
        const auto it = sl.expert_slot.find(sl.uniq[i]);
        if (it != sl.expert_slot.end()) {
            sl.keep[it->second] = 1;
        }
    }

    // reserve and demand-load this wave's experts (per-expert, same path as the decode remap)
    bool waited = false;
    if (count > 0) {
        stats.n_waves_run++;
        for (size_t i = first; i < first + count; i++) {
            const int32_t e  = sl.uniq[i];
            const auto    it = sl.expert_slot.find(e);
            if (it != sl.expert_slot.end()) {
                // already in the cache (resident, or still loading from the previous wave's preload)
                const int32_t s = it->second;
                if (sl.slot_state[s] == LLAMA_MOE_STREAM_SLOT_LOADING) {
                    enqueue_slot_locked(sl, e, s); // promote to demand, wait for it
                    waited = true;
                } else {
                    stats.n_preload_ready++; // resident from the previous wave's preload
                }
                stats.n_hit++;
                sl.keep[s] = 1;
                sl.demand_slots.push_back(s);
            } else {
                // miss: evict a non-kept slot and queue the load
                int32_t v;
                while ((v = pick_victim_locked(sl, sl.keep.data())) < 0) {
                    cv_done.wait(lk);
                    if (load_failed) {
                        GGML_ABORT("MoE expert streaming: expert load failed (I/O error)");
                    }
                }
                if (!sl.seen[e]) {
                    stats.n_miss_cold++;
                }
                reserve_slot_locked(sl, e, v);
                enqueue_slot_locked(sl, e, v);
                stats.n_miss++;
                waited = true;
                sl.keep[v] = 1;
                sl.demand_slots.push_back(v);
            }
        }
    }

    // best-effort preload of the next wave so its loads overlap this wave's compute; never waits,
    //   whatever cannot be reserved now simply becomes the next wave's demand load
    if (std::getenv("LLAMA_MOE_STREAM_NO_PRELOAD") == nullptr) {
        for (size_t i = nfirst; i < nfirst + ncount; i++) {
            const int32_t e = sl.uniq[i];
            if (sl.expert_slot.find(e) != sl.expert_slot.end()) {
                continue;
            }
            const int32_t v = pick_victim_locked(sl, sl.keep.data());
            if (v < 0) {
                continue;
            }
            if (!sl.seen[e]) {
                stats.n_miss_cold++;
            }
            reserve_slot_locked(sl, e, v);
            sl.keep[v] = 1;
            enqueue_slot_locked(sl, e, v);
            stats.n_preload_issued++;
        }
    }

    if (waited) {
        const int64_t t0 = ggml_time_us();
        cv_done.wait(lk, [&]{
            if (load_failed) {
                return true;
            }
            for (const int32_t s : sl.demand_slots) {
                if (sl.slot_state[s] != LLAMA_MOE_STREAM_SLOT_RESIDENT) {
                    return false;
                }
            }
            return true;
        });
        if (load_failed) {
            GGML_ABORT("MoE expert streaming: expert load failed (I/O error)");
        }
        stats.t_stall_wave_us += ggml_time_us() - t0;
    }

    // parking pool: this wave's own resident slots plus the borrowed ones (all keep-protected;
    //   the next same-layer reservation is ordered after this wave's GEMMs by the graph)
    sl.plan_pool = sl.demand_slots;
    sl.plan_pool.insert(sl.plan_pool.end(), borrowed.begin(), borrowed.end());
    GGML_ASSERT(sl.plan_pool.size() >= n_ids);
}

// write out[i] = the cache slot the GEMM should index for each (token, expert) pair of wave w, one
// token row at a time: pairs whose expert is in this wave get its real slot; the rest park on distinct
// resident pool slots (pool_used prevents a repeat within the row, required by the Metal kernel)
void llama_moe_stream::emit_wave_slots(llama_moe_stream_layer & sl, const int32_t * ids, int32_t * out,
        int32_t w, uint32_t n_ids, int64_t n_tok) {
    for (int64_t t = 0; t < n_tok; t++) {
        sl.pool_used.clear();

        // pass 1: pairs whose expert belongs to this wave -> that expert's real (resident) slot
        for (uint32_t kk = 0; kk < n_ids; kk++) {
            const int64_t i = t*n_ids + kk;
            const int32_t e = ids[i];
            GGML_ASSERT(sl.expert_wave[e] != 0xff);
            if (sl.expert_wave[e] == (uint8_t) w) {
                const int32_t s = sl.expert_slot.at(e);
                GGML_ASSERT(sl.slot_state[s] == LLAMA_MOE_STREAM_SLOT_RESIDENT);
                sl.slot_last_use[s] = ++sl.use_counter;
                out[i] = s;
                sl.pool_used.push_back(s);
            }
        }

        // pass 2: the remaining (masked) pairs -> the next pool slot not yet used in this row
        size_t pi = 0;
        for (uint32_t kk = 0; kk < n_ids; kk++) {
            const int64_t i = t*n_ids + kk;
            if (sl.expert_wave[ids[i]] == (uint8_t) w) {
                continue;
            }
            while (std::find(sl.pool_used.begin(), sl.pool_used.end(), sl.plan_pool[pi]) != sl.pool_used.end()) {
                pi++;
                GGML_ASSERT(pi < sl.plan_pool.size());
            }
            GGML_ASSERT(sl.slot_state[sl.plan_pool[pi]] == LLAMA_MOE_STREAM_SLOT_RESIDENT);
            out[i] = sl.plan_pool[pi];
            sl.pool_used.push_back(sl.plan_pool[pi]);
            pi++;
        }
    }
}

// Custom-op callback for one pass of multi-pass prefill. When a ubatch touches more experts than the
// cache holds, build_moe_ffn runs the expert GEMMs in several waves; this runs once per wave (single-
// threaded on ith 0), in wave order. For wave w it makes that wave's expert slice resident (preloading
// the next wave), then writes the slot ids the GEMM indexes - see plan_waves_locked / stage_wave_locked
// / emit_wave_slots. The router's expert choice is untouched, so the output matches a non-streamed run.
void llama_moe_stream_wave_ids(ggml_tensor * dst, int ith, int nth, void * userdata) {
    GGML_UNUSED(nth);
    if (ith != 0) {
        return;
    }

    auto * ud  = (llama_moe_stream_wave *) userdata;
    auto * sl  = ud->sl;
    auto * mgr = sl->mgr;

    const int32_t w = ud->wave;

    const ggml_tensor * a = dst->src[0]; // contiguous selected ids
    GGML_ASSERT(a->type == GGML_TYPE_I32);
    GGML_ASSERT(ggml_is_contiguous(a));
    GGML_ASSERT(ggml_nelements(dst) == ggml_nelements(a));
    GGML_ASSERT(dst->data != a->data); // the emit must not clobber the ids other waves read

    const int64_t   n   = ggml_nelements(a);
    const int32_t * ids = (const int32_t *) a->data;
          int32_t * out = (int32_t *) dst->data;

    std::unique_lock<std::mutex> lk(mgr->mtx);

    if (mgr->load_failed) {
        GGML_ABORT("MoE expert streaming: expert load failed (I/O error)");
    }

    mgr->stats.n_wave_calls++;

    if (w == 0) {
        mgr->plan_waves_locked(*sl, ids, n);
    }
    GGML_ASSERT(sl->plan_next_wave == w); // waves must run in order (enforced by the graph ordering token)

    const uint32_t n_ids = (uint32_t) a->ne[0]; // experts per token (n_expert_used)

    mgr->stage_wave_locked(lk, *sl, w, n_ids); // make this wave resident, preload the next, build the pool
    sl->plan_next_wave = w + 1;

    mgr->emit_wave_slots(*sl, ids, out, w, n_ids, a->ne[1]);
}

// multi-pass prefill: 1.0 for pairs whose expert belongs to wave w, 0.0 otherwise; multiplied into
// this wave's expert GEMM output so the masked-out (parked) pairs contribute nothing to the sum
void llama_moe_stream_wave_mask(ggml_tensor * dst, int ith, int nth, void * userdata) {
    GGML_UNUSED(nth);
    if (ith != 0) {
        return;
    }

    auto * ud  = (llama_moe_stream_wave *) userdata;
    auto * sl  = ud->sl;
    auto * mgr = sl->mgr;

    const int32_t w = ud->wave;

    const ggml_tensor * a = dst->src[0]; // contiguous selected ids
    GGML_ASSERT(a->type == GGML_TYPE_I32);
    GGML_ASSERT(ggml_is_contiguous(a));
    GGML_ASSERT(dst->type == GGML_TYPE_F32);
    GGML_ASSERT(ggml_nelements(dst) == ggml_nelements(a));

    const int64_t   n   = ggml_nelements(a);
    const int32_t * ids = (const int32_t *) a->data;
          float   * out = (float *) dst->data;

    std::lock_guard<std::mutex> lock(mgr->mtx);

    GGML_ASSERT(sl->plan_next_wave > w); // this wave's ids op has already run

    for (int64_t i = 0; i < n; i++) {
        out[i] = sl->expert_wave[ids[i]] == (uint8_t) w ? 1.0f : 0.0f;
    }
}
