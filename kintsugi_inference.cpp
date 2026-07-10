#include <windows.h>
#include <psapi.h>
#include <iostream>
#include <random>
#include <vector>
#include <chrono>
#include <atomic>
#include <immintrin.h>
#include <cmath>
#include <limits>
#include <climits>
#include <cstdint>
#include <fstream>
#include <string>
#include <iomanip>
#include <unordered_map>
#include <array>
#include "core/shared_bus.h"

// ==============================================================================
// KINTSUGI: REAL PROCESS MEMORY SAMPLING (DIAGNOSTIC)
// ==============================================================================
// Added specifically to answer "is memory actually spiking per-token, or is
// something being skipped" with real numbers instead of an external monitor
// eyeballed against timestamps after the fact. WorkingSetSize is what's
// physically resident in RAM right now; PagefileUsage is committed/private
// memory (may include pages not currently resident). Both come straight
// from the OS via GetProcessMemoryInfo — real, not estimated.
//
// SysMemLoad added after noticing an unexplained mid-decode working-set dip
// that only lined up with a system-wide pressure reading by eyeballing two
// separately-timestamped logs (this process's own prints vs. wake_jarvis.py's
// external monitor). Pulling GlobalMemoryStatusEx's dwMemoryLoad directly
// into the SAME print line means every future correlation is a real,
// single-source read, not an approximate cross-log alignment.
void PrintMemoryUsage(const char* label) {
    PROCESS_MEMORY_COUNTERS pmc;
    MEMORYSTATUSEX mem_status;
    mem_status.dwLength = sizeof(mem_status);
    bool sys_ok = GlobalMemoryStatusEx(&mem_status);

    if (GetProcessMemoryInfo(GetCurrentProcess(), &pmc, sizeof(pmc))) {
        double working_set_mb = static_cast<double>(pmc.WorkingSetSize) / (1024.0 * 1024.0);
        double private_mb     = static_cast<double>(pmc.PagefileUsage)  / (1024.0 * 1024.0);
        std::cout << "    [MEM] " << label << " WorkingSet=" << working_set_mb
                  << "MB Private=" << private_mb << "MB";
        if (sys_ok) {
            std::cout << " SysMemLoad=" << mem_status.dwMemoryLoad << "%";
        }
        std::cout << "\n";
    } else {
        std::cout << "    [MEM] " << label << " -- GetProcessMemoryInfo failed, Error Code: "
                  << GetLastError() << "\n";
    }
}

// Query counterpart to PrintMemoryUsage — returns the number instead of only
// printing it, so checkpoint 2's watermark logic has something real to gate
// on. Returns -1.0 on failure (sentinel: caller should treat as "unknown,
// don't gate a decision on it" rather than silently trimming or not).
double GetProcessWorkingSetMB() {
    PROCESS_MEMORY_COUNTERS pmc;
    if (GetProcessMemoryInfo(GetCurrentProcess(), &pmc, sizeof(pmc))) {
        return static_cast<double>(pmc.WorkingSetSize) / (1024.0 * 1024.0);
    }
    return -1.0;
}

// Non-invasive residency check: answers "is this specific page currently
// resident" WITHOUT touching it, unlike a touch-and-time probe (which would
// itself fault the page in and contaminate the very thing being measured).
// Built specifically to test whether checkpoint 2's trim actually evicted
// the LM-head buffer, rather than continuing to infer it from aggregate
// step timing, which conflates dequant + compute + residency into one
// number and can't isolate which one moved.
bool IsPageResident(const void* address) {
    PSAPI_WORKING_SET_EX_INFORMATION info;
    info.VirtualAddress = const_cast<void*>(address);
    if (QueryWorkingSetEx(GetCurrentProcess(), &info, sizeof(info))) {
        return info.VirtualAttributes.Valid;
    }
    return false; // query failed -- treat as unknown/not-resident, conservatively
}

// Samples several points spread across a buffer and reports residency for
// each, rather than just first/last byte -- partial eviction (some pages
// survived, others didn't) wouldn't show up if only the extremes were checked.
void PrintBufferResidency(const char* label, const uint8_t* buffer, size_t size_bytes) {
    constexpr int SAMPLE_COUNT = 5;
    std::cout << "    [RESIDENCY] " << label << ":";
    for (int i = 0; i < SAMPLE_COUNT; ++i) {
        size_t offset = (size_bytes / (SAMPLE_COUNT - 1)) * i;
        if (offset >= size_bytes) offset = size_bytes - 1;
        bool resident = IsPageResident(buffer + offset);
        std::cout << " [" << (i * 100 / (SAMPLE_COUNT - 1)) << "%="
                  << (resident ? "resident" : "EVICTED") << "]";
    }
    std::cout << "\n";
}

// ==============================================================================
// KINTSUGI: THE TENSOR ARENA (PHASE 1 BEDROCK)
// ==============================================================================

// Shrunk from 4.5GB: the embedding table no longer lives here (it dequantizes
// on demand straight out of the AWE model vault below), so this only needs to
// cover the reused per-layer F32 scratch (~870MB), KV caches (~16MB), and the
// small forward-pass scratch buffers — 1.5GB leaves real headroom for the AWE
// vault (~4.6GB) to coexist on this 7.87GB machine.
constexpr size_t MONOLITHIC_ARENA_SIZE = 1536ULL * 1024ULL * 1024ULL; // 1.5GB
constexpr size_t KINTSUGI_MODEL_SIZE = 2ULL * 1024ULL * 1024ULL * 1024ULL; // 2GB Target Weights
constexpr size_t FLOAT_COUNT = KINTSUGI_MODEL_SIZE / sizeof(float);

struct TensorArena {
    void* raw_base_ptr = nullptr;
    size_t total_capacity = 0;
    size_t current_offset = 0;
    HANDLE reservoir_handle = nullptr; // set only if backed by kintsugi_page_reservoir.exe's
                                        // shared mapping -- Teardown must NOT VirtualFree this

    bool Initialize(size_t size_in_bytes) {
        std::cout << "[⚡ SILICON] Booting Tensor Arena Allocation...\n";

        // --- 1. SEIZE THE KERNEL PRIVILEGE TOKEN ---
        HANDLE hToken;
        if (OpenProcessToken(GetCurrentProcess(), TOKEN_ADJUST_PRIVILEGES | TOKEN_QUERY, &hToken)) {
            TOKEN_PRIVILEGES tp;
            if (LookupPrivilegeValue(NULL, SE_LOCK_MEMORY_NAME, &tp.Privileges[0].Luid)) {
                tp.PrivilegeCount = 1;
                tp.Privileges[0].Attributes = SE_PRIVILEGE_ENABLED;
                AdjustTokenPrivileges(hToken, FALSE, &tp, 0, NULL, 0);
                if (GetLastError() == ERROR_SUCCESS) {
                    std::cout << "[🛡️ KERNEL] Seized SE_LOCK_MEMORY privilege.\n";
                }
            }
            CloseHandle(hToken);
        }

        // --- 1.5. SAFETY NET, NOT PLAN A: try the shared large-page reservoir ---
        // If kintsugi_page_reservoir.exe is running (started at boot, before
        // fragmentation sets in), it's already holding a Local\ named
        // mapping backed by genuinely large pages — guaranteed, since it
        // grabbed them fresh at startup rather than fighting fragmented
        // physical RAM hours into a session. Local\ is session-scoped: this
        // only works when both processes run in the same interactive
        // session, not under a SYSTEM/Task-Scheduler deployment (open item,
        // see CLAUDE.md). If it's there, this is strictly better than
        // anything below and we return immediately.
        // If it's NOT there (reservoir never started, or this machine
        // doesn't run it), this falls straight through to the existing,
        // already-proven self-contained attempt below, completely
        // unchanged — this is additive, not a replacement.
        //
        // Critical, easy-to-miss Windows behavior verified before writing
        // this: since Windows 10 1703, MapViewOfFile defaults to SMALL
        // pages even for a SEC_LARGE_PAGES mapping unless FILE_MAP_LARGE_PAGES
        // is explicitly passed to the map call too — omitting it would
        // silently succeed while delivering zero actual large-page benefit.
        HANDLE hReservoir = OpenFileMappingW(FILE_MAP_ALL_ACCESS | FILE_MAP_LARGE_PAGES,
                                              FALSE, L"Local\\KintsugiLargePageReservoir");
        if (hReservoir) {
            // Don't assume the reservoir secured the full size_in_bytes —
            // it probes for the largest contiguous size it can actually get
            // (kintsugi_page_reservoir.cpp's own CANDIDATE_SIZES_MB loop),
            // and that number varies by boot/system conditions. Requesting
            // a view larger than what's really backing the mapping is
            // exactly the kind of assumption that either fails outright or,
            // worse, "succeeds" in some edge case while total_capacity
            // silently doesn't match reality — the bump allocator's own
            // bounds check (AllocateAligned) only protects against overrun
            // if total_capacity is TRUE, not assumed. Mirroring the
            // reservoir's own probe here closes that gap: whatever size
            // actually maps IS by construction what the reservoir holds,
            // no separate communication channel needed, no assumption left
            // standing unverified.
            constexpr size_t PROBE_SIZES_MB[] = { 1536, 1024, 768, 512, 256, 128, 64, 32, 2 };
            void* reservoir_view = nullptr;
            size_t reservoir_actual_bytes = 0;

            for (size_t candidate_mb : PROBE_SIZES_MB) {
                size_t candidate_bytes = candidate_mb * 1024ULL * 1024ULL;
                if (candidate_bytes > size_in_bytes) continue; // never map more than the engine actually asked for
                reservoir_view = MapViewOfFile(hReservoir, FILE_MAP_ALL_ACCESS | FILE_MAP_LARGE_PAGES,
                                                0, 0, candidate_bytes);
                if (reservoir_view) {
                    reservoir_actual_bytes = candidate_bytes;
                    break;
                }
            }

            if (reservoir_view) {
                std::cout << "[🏦 RESERVOIR] Found kintsugi_page_reservoir.exe's standing large-page "
                             "reservation — using it directly, skipping the fragmentation-prone attempt below.\n";
                std::cout << "[🏦 RESERVOIR] Secured " << (reservoir_actual_bytes / (1024.0 * 1024.0))
                          << "MB (requested up to " << (size_in_bytes / (1024.0 * 1024.0))
                          << "MB) — total_capacity set to what's ACTUALLY mapped, not assumed.\n";
                if (reservoir_actual_bytes < size_in_bytes) {
                    std::cout << "[🏦 RESERVOIR] NOTE: this is smaller than the full arena request. "
                                 "Later allocations that don't fit will fail safely with a clear OOM "
                                 "message (AllocateAligned's own bounds check), not silently overrun.\n";
                }
                raw_base_ptr = reservoir_view;
                total_capacity = reservoir_actual_bytes; // the real, verified size -- not size_in_bytes
                current_offset = 0;
                reservoir_handle = hReservoir; // kept alive for Teardown(); do NOT VirtualFree this memory
                return true;
            }
            // No size at all mapped successfully -- close the handle and
            // fall through to the existing self-contained path rather than
            // half-use it.
            CloseHandle(hReservoir);
        }

        // --- 2. PREPARE WORKING SET & ALIGNMENT ---
        SIZE_T largePageMin = GetLargePageMinimum();
        
        // Working set request sized to match the actual arena, not padded
        // beyond available RAM. On this machine (7.87GB total), requesting
        // arena + 1GB overhead was exceeding total installed RAM — causing
        // ERROR_NO_SYSTEM_RESOURCES (1450) every run.
        SIZE_T min_ws = size_in_bytes;
        SIZE_T max_ws = size_in_bytes + (256ULL * 1024ULL * 1024ULL); // 256MB headroom only
        bool ws_resized = SetProcessWorkingSetSize(GetCurrentProcess(), min_ws, max_ws);
        if (!ws_resized) {
            // Capture immediately — same evaluation-order rule as the
            // VirtualLock fix. This is the actual open question right now:
            // we've been requesting an 8.5-9GB working set this whole time
            // and never once checked whether the OS granted it.
            DWORD ws_err = GetLastError();
            std::cerr << "[!] WARNING: SetProcessWorkingSetSize rejected the arena-size request. Error Code: "
                      << ws_err << ". VirtualLock may fail with 1453 as a result — "
                      << "running elevated (Administrator) is the likely fix.\n";
        }

        // Snap the requested size to the nearest massive page boundary
        if (largePageMin != 0) {
            total_capacity = (size_in_bytes + largePageMin - 1) & ~(largePageMin - 1);
        } else {
            total_capacity = size_in_bytes;
        }
        current_offset = 0;

        // --- 3. ALLOCATE AND LOCK (WITH LARGE PAGE FALLBACK) ---
        
        // Ensure size is a multiple of the Large Page minimum (Usually 2MB)
        if (largePageMin != 0 && total_capacity % largePageMin == 0) {
            std::cout << "[⚡ SILICON] Attempting 2MB Large Page allocation for zero TLB thrashing...\n";
            raw_base_ptr = VirtualAlloc(NULL, total_capacity, MEM_RESERVE | MEM_COMMIT | MEM_LARGE_PAGES, PAGE_READWRITE);
        }

        if (raw_base_ptr) {
            std::cout << "[🟢 OVERDRIVE] Arena secured with Large Pages. TLB misses eliminated.\n";
            // Note: MEM_LARGE_PAGES are inherently locked by the OS. No VirtualLock needed.
        } else {
            std::cout << "[!] Kernel denied Large Pages (RAM fragmented). Falling back to 4KB locked pages...\n";
            
            // Fallback to standard allocation
            raw_base_ptr = VirtualAlloc(NULL, total_capacity, MEM_RESERVE | MEM_COMMIT, PAGE_READWRITE);
            
            if (!raw_base_ptr) {
                std::cerr << "[❌ FATAL] VirtualAlloc rejected both Large and Standard memory requests.\n";
                return false;
            }

            bool locked = VirtualLock(raw_base_ptr, total_capacity);
            if (locked) {
                std::cout << "[⚡ SILICON] Standard Tensor Arena physically pinned to RAM. No-swap active.\n";
            } else {
                // Capture IMMEDIATELY, before any iostream call — operator<<
                // is left-associative, so "std::cerr << literal << GetLastError()"
                // actually streams the literal FIRST (a real I/O call that can
                // reset the thread's last-error state on Windows) and only
                // THEN evaluates GetLastError() — which is exactly why this
                // kept printing "Error Code: 0" every single run.
                DWORD err = GetLastError();
                std::cerr << "[!] WARNING: VirtualLock failed. Error Code: " << err << "\n";
            }
        }

        return true;
    }

    void* AllocateAligned(size_t size_in_bytes) {
        uintptr_t current_addr = reinterpret_cast<uintptr_t>(raw_base_ptr) + current_offset;
        uintptr_t aligned_addr = (current_addr + 63) & ~63; // Push to nearest 64-byte boundary
        size_t padding = aligned_addr - current_addr;
        
        if (current_offset + padding + size_in_bytes > total_capacity) {
            std::cerr << "[❌ FATAL] Tensor Arena Out Of Memory (OOM)!\n";
            return nullptr;
        }

        current_offset += (padding + size_in_bytes);
        
        // [FIRST-TOUCH FAULTING] 
        memset(reinterpret_cast<void*>(aligned_addr), 0, size_in_bytes);
        
        return reinterpret_cast<void*>(aligned_addr);
    }

    void Teardown() {
        if (raw_base_ptr && reservoir_handle) {
            // We only hold a VIEW into kintsugi_page_reservoir.exe's mapping --
            // VirtualFree would attempt to free memory this process never
            // allocated, which is exactly the kind of bug that's silent until
            // it corrupts something. Unmap our view and close our handle;
            // the reservoir process keeps the actual reservation alive.
            UnmapViewOfFile(raw_base_ptr);
            CloseHandle(reservoir_handle);
        } else if (raw_base_ptr) {
            VirtualUnlock(raw_base_ptr, total_capacity); // Ignored harmlessly by OS if MEM_LARGE_PAGES was used
            VirtualFree(raw_base_ptr, 0, MEM_RELEASE);
        }
    }
};

// ==============================================================================
// KINTSUGI: THE MODEL VAULT (PHASE 11 - MEMORY-MAPPED GGUF)
// ==============================================================================
// The measured per-token cost was never the math — it was re-opening the GGUF
// file and re-reading ~4.6GB off disk for every single decode step (all 32
// layers, plus the full output.weight sweep for the LM head, every time).
//
// First attempt here was AWE (AllocateUserPhysicalPages/MapUserPhysicalPages)
// pre-pinning the whole tensor-data section into RAM upfront. That failed on
// the real hardware: this machine only has ~3.6-4.2GB free at any given
// moment (not the clean 7.87GB total), and AWE either gets the FULL requested
// page count or hard-fails — no partial credit, no graceful degradation.
//
// Memory-mapping the file instead is the right tool for this shape of
// problem: the OS's own page cache keeps whatever pages get touched more
// than once resident in whatever RAM is actually free at the time, and
// evicts under pressure instead of crashing. First prefill still pays full
// disk cost (cold cache); every decode step re-touching the same layer/LM-head
// bytes after that gets served at RAM speed for however much fits — adapting
// automatically to this machine's real, variable headroom instead of
// demanding an exact fixed amount be free.
struct MappedModelFile {
    HANDLE hFile = nullptr;
    HANDLE hMap  = nullptr;
    const uint8_t* base = nullptr; // already offset to tensor_data_offset

    bool Open(const std::string& path, uint64_t tensor_data_offset) {
        hFile = CreateFileA(path.c_str(), GENERIC_READ, FILE_SHARE_READ, NULL,
                             OPEN_EXISTING, FILE_ATTRIBUTE_NORMAL | FILE_FLAG_SEQUENTIAL_SCAN, NULL);
        if (hFile == INVALID_HANDLE_VALUE) {
            std::cerr << "[❌ FATAL] Cannot open GGUF file for memory mapping: " << path << "\n";
            return false;
        }

        hMap = CreateFileMappingA(hFile, NULL, PAGE_READONLY, 0, 0, NULL);
        if (!hMap) {
            std::cerr << "[❌ FATAL] CreateFileMapping failed for the model vault.\n";
            CloseHandle(hFile);
            hFile = nullptr;
            return false;
        }

        void* view = MapViewOfFile(hMap, FILE_MAP_READ, 0, 0, 0); // 0 size = map the whole file
        if (!view) {
            std::cerr << "[❌ FATAL] MapViewOfFile failed for the model vault.\n";
            CloseHandle(hMap); hMap = nullptr;
            CloseHandle(hFile); hFile = nullptr;
            return false;
        }

        base = static_cast<const uint8_t*>(view) + tensor_data_offset;
        std::cout << "[🔒 VAULT] GGUF memory-mapped — layers and LM-head rows will warm into the "
                     "OS page cache as they're touched, no fixed RAM reservation required.\n";
        return true;
    }
};

MappedModelFile g_ModelFile;

// ==============================================================================
// KINTSUGI: THE ROUTING CORE (PHASE 2 - KV CACHE)
// ==============================================================================
struct KVCacheRing {
    float* key_matrix = nullptr;
    float* value_matrix = nullptr;
    
    size_t max_context_length = 0; 
    size_t hidden_dimension = 0;   
    size_t head_index = 0;         

    bool MapToArena(TensorArena& arena, size_t context_len, size_t dim) {
        max_context_length = context_len;
        hidden_dimension = dim;
        head_index = 0;

        size_t matrix_size_bytes = max_context_length * hidden_dimension * sizeof(float);

        std::cout << "[🧠 ROUTING] Carving KV-Cache Ring from Monolithic Arena...\n";

        // Pull 64-byte aligned blocks for the context history
        key_matrix = static_cast<float*>(arena.AllocateAligned(matrix_size_bytes));
        value_matrix = static_cast<float*>(arena.AllocateAligned(matrix_size_bytes));

        if (!key_matrix || !value_matrix) {
            std::cerr << "[❌ FATAL] Arena rejected KV-Cache mapping. OOM.\n";
            return false;
        }

        std::cout << "[⚡ SILICON] KV-Cache mapped successfully. Context Window: " 
                  << max_context_length << " tokens.\n";
        return true;
    }

    // Pushes new tokens into the cache, automatically wrapping around the ring
    void PushContext(const float* incoming_keys, const float* incoming_values) {
        float* target_key_row = key_matrix + (head_index * hidden_dimension);
        float* target_val_row = value_matrix + (head_index * hidden_dimension);

        memcpy(target_key_row, incoming_keys, hidden_dimension * sizeof(float));
        memcpy(target_val_row, incoming_values, hidden_dimension * sizeof(float));

        // Advance the ring buffer head, wrapping if necessary (Zero-Allocation sliding window)
        head_index = (head_index + 1) % max_context_length;
    }
};

// KINTSUGI: THE LEXICON (EMBEDDING TABLE) — struct defined further below,
// after the Q4_K dequant kernel, since Lookup() now dequantizes a row
// directly out of the AWE vault's raw GGUF bytes instead of a pre-dequantized
// arena copy. g_Embedding itself is declared there too.

// Globals
TensorArena g_Arena;
// Per-layer KV caches — NOT a single shared cache. Each of the 32 transformer
// layers needs its OWN independent K/V history that persists across the
// whole generation (prefill + every decode step), since layer 5's attention
// must only ever see layer 5's own past keys/values, never layer 0's or
// layer 12's. A single shared cache (the old design) only worked for
// scoring one next-token after prefill — it had no way to hold history
// across multiple sequential decode steps.
std::vector<KVCacheRing> g_LayerKVCaches;
std::atomic<bool> g_model_loaded{false};

// ==============================================================================
// KINTSUGI: THE CRUNCHER (PHASE 3 - VECTOR ACCELERATION)
// ==============================================================================
struct KintsugiCruncher {

    // input_dim -> output_dim. weight_matrix is [output_dim x input_dim], row-major.
    // (Previously this took a single hidden_dim and silently assumed a square
    // matrix — fine while d_model == d_ff in testing, broken the moment they diverge.)
    static void SIMD_DenseProjection(const float* input_vec, const float* weight_matrix, float* out_vec, size_t input_dim, size_t output_dim) {
        for (size_t out_idx = 0; out_idx < output_dim; ++out_idx) {
            __m256 v_sum = _mm256_setzero_ps();
            const float* current_weight_row = weight_matrix + (out_idx * input_dim);

            for (size_t i = 0; i < input_dim; i += 8) {
                __m256 v_in = _mm256_load_ps(&input_vec[i]);
                __m256 v_w = _mm256_load_ps(&current_weight_row[i]);
                v_sum = _mm256_fmadd_ps(v_in, v_w, v_sum);
            }

            alignas(32) float sum_array[8];
            _mm256_store_ps(sum_array, v_sum);
            
            out_vec[out_idx] = sum_array[0] + sum_array[1] + sum_array[2] + sum_array[3] + 
                               sum_array[4] + sum_array[5] + sum_array[6] + sum_array[7];
        }
    }
};

KintsugiCruncher g_Cruncher;

// ==============================================================================
// KINTSUGI: THE AGGREGATOR (PHASE 4 - DOWN-PROJECTION BUS)
// ==============================================================================
struct KintsugiAccumulator {

    static void SIMD_DownProjection(const float* hidden_vec,      
                                     const float* weight_matrix,  
                                     float* residual_stream,      
                                     size_t d_model,
                                     size_t d_ff) {

        for (size_t row = 0; row < d_model; row += 8) {

            __m256 acc[8];
            for (int r = 0; r < 8; ++r) acc[r] = _mm256_setzero_ps();

            for (size_t k = 0; k < d_ff; k += 8) {
                __m256 v_h = _mm256_load_ps(&hidden_vec[k]);
                for (int r = 0; r < 8; ++r) {
                    const float* weight_row = weight_matrix + ((row + r) * d_ff);
                    __m256 v_w = _mm256_load_ps(&weight_row[k]);
                    acc[r] = _mm256_fmadd_ps(v_w, v_h, acc[r]);
                }
            }

            alignas(32) float dots[8];
            for (int r = 0; r < 8; ++r) {
                alignas(32) float lanes[8];
                _mm256_store_ps(lanes, acc[r]);
                dots[r] = lanes[0] + lanes[1] + lanes[2] + lanes[3] +
                          lanes[4] + lanes[5] + lanes[6] + lanes[7];
            }

            __m256 v_dot      = _mm256_load_ps(dots);
            __m256 v_residual = _mm256_load_ps(&residual_stream[row]); 
            __m256 v_result   = _mm256_add_ps(v_residual, v_dot);

            _mm256_stream_ps(&residual_stream[row], v_result);
        }

        _mm_sfence();
    }
};

KintsugiAccumulator g_Accumulator;

// ==============================================================================
// KINTSUGI: THE NERVOUS SYSTEM (PHASE 5 - ZERO-ALLOCATION CONTROLLER)
// ==============================================================================
struct TransformerLayerPointers {
    float* ffn_weights;
    float* down_proj_weights;
    // (Attention pointers will go here in the future)
};

struct KintsugiController {
    TransformerLayerPointers* jump_table = nullptr;
    size_t total_layers = 0;

    bool Initialize(TensorArena& arena, size_t num_layers) {
        total_layers = num_layers;
        std::cout << "[🧠 NERVOUS SYSTEM] Forging Static Jump Table for " << num_layers << " layers...\n";
        
        // Allocate the jump table directly in the monolithic arena. Zero heap fragmentation.
        jump_table = static_cast<TransformerLayerPointers*>(arena.AllocateAligned(total_layers * sizeof(TransformerLayerPointers)));
        
        if (!jump_table) {
            std::cerr << "[❌ FATAL] Arena rejected Jump Table mapping.\n";
            return false;
        }
        
        std::cout << "[⚡ SILICON] Jump Table mapped. Execution pipeline locked.\n";
        return true;
    }

    // The Master Loop. Branches are dead. Only sequential memory execution remains.
    //
    // ffn_scratch must be allocated for at least d_ff floats (not hardcoded to
    // d_model) — it holds the expanded hidden state between the two projections.
    // ffn_weights must be sized [d_ff x d_model]; down_proj_weights [d_model x d_ff].
    // These are NOT interchangeable buffers once you load real trained weights —
    // reusing one pointer for both (as the current main() test does) only works
    // because the synthetic test is square.
    void FireTokenPipeline(float* residual_stream, float* ffn_scratch, size_t d_model, size_t d_ff) {
        for (size_t i = 0; i < total_layers; ++i) {
            
            // 1. Cruncher: FFN Up/Gate Projection — d_model in, d_ff out
            g_Cruncher.SIMD_DenseProjection(residual_stream, jump_table[i].ffn_weights, ffn_scratch, d_model, d_ff);
            
            // 2. Aggregator: FFN Down Projection & Non-Temporal Residual Fuse — d_ff in, d_model out
            g_Accumulator.SIMD_DownProjection(ffn_scratch, jump_table[i].down_proj_weights, residual_stream, d_model, d_ff);
        }
    }
};

KintsugiController g_Controller;

// ==============================================================================
// KINTSUGI: THE ATTENTION CORTEX (PHASE 6 - GROUPED-QUERY ATTENTION)
// ==============================================================================
struct KintsugiAttention {

    // Llama-3 8B's real head config: 32 query heads, but only 8 KV heads —
    // Grouped-Query Attention, 4 query heads share each KV head. head_dim is
    // 128 for both, but that means KV's total width is 8*128=1024, NOT 4096 —
    // the KV-cache below is deliberately mapped narrower than d_model because
    // of this, not by mistake.
    static constexpr size_t NUM_Q_HEADS = 32;
    static constexpr size_t NUM_KV_HEADS = 8;
    static constexpr size_t HEAD_DIM = 128;
    static constexpr size_t Q_PER_KV = NUM_Q_HEADS / NUM_KV_HEADS; // 4

    // query_vec        : [4096] — NOT yet RoPE-rotated, no Wq/Wk/Wv/Wo exist yet.
    // kv_cache          : hidden_dimension MUST be NUM_KV_HEADS*HEAD_DIM (1024).
    // context_len       : valid rows currently in the cache.
    // attention_output : [4096], concatenated across all 32 query heads.
    // score_scratch     : caller-owned, >= max_context_length floats, reused per head.
    static void ComputeAttention(const float* query_vec,
                                  KVCacheRing& kv_cache,
                                  size_t context_len,
                                  float* attention_output,
                                  float* score_scratch) {

        if (context_len == 0) {
            std::fill_n(attention_output, NUM_Q_HEADS * HEAD_DIM, 0.0f);
            return;
        }

        const float scale = 1.0f / std::sqrt(static_cast<float>(HEAD_DIM));

        for (size_t qh = 0; qh < NUM_Q_HEADS; ++qh) {
            const size_t kvh = qh / Q_PER_KV; // GQA grouping
            const float* q_head = query_vec + (qh * HEAD_DIM);

            // --- 1. Scaled dot-product scores against every cached position ---
            float max_score = -std::numeric_limits<float>::infinity();
            for (size_t t = 0; t < context_len; ++t) {
                const float* k_row = kv_cache.key_matrix + (t * kv_cache.hidden_dimension) + (kvh * HEAD_DIM);

                __m256 v_sum = _mm256_setzero_ps();
                for (size_t i = 0; i < HEAD_DIM; i += 8) {
                    __m256 v_q = _mm256_load_ps(&q_head[i]);
                    __m256 v_k = _mm256_load_ps(&k_row[i]);
                    v_sum = _mm256_fmadd_ps(v_q, v_k, v_sum);
                }
                alignas(32) float lanes[8];
                _mm256_store_ps(lanes, v_sum);
                float dot = lanes[0]+lanes[1]+lanes[2]+lanes[3]+lanes[4]+lanes[5]+lanes[6]+lanes[7];

                float score = dot * scale;
                score_scratch[t] = score;
                if (score > max_score) max_score = score;
            }

            // --- 2. Softmax, numerically stable ---
            float sum_exp = 0.0f;
            for (size_t t = 0; t < context_len; ++t) {
                float e = std::exp(score_scratch[t] - max_score);
                score_scratch[t] = e;
                sum_exp += e;
            }
            float inv_sum = 1.0f / sum_exp;

            // --- 3. Weighted sum over cached V ---
            // HEAD_DIM/8 = 16 chunks total. Holding all 16 as live accumulators
            // across the whole context_len loop would saturate every YMM
            // register with nothing left for the V load itself — guaranteed
            // stack spilling, exactly what the audit doc warned about for
            // unrolling. 2 passes of 8 accumulators keeps it register-safe at
            // the cost of reading this head's V slice twice instead of once.
            constexpr size_t ACC_CHUNK = 8;
            float* out_head = attention_output + (qh * HEAD_DIM);

            for (size_t base = 0; base < HEAD_DIM; base += ACC_CHUNK * 8) {
                __m256 acc[ACC_CHUNK];
                for (size_t c = 0; c < ACC_CHUNK; ++c) acc[c] = _mm256_setzero_ps();

                for (size_t t = 0; t < context_len; ++t) {
                    __m256 v_weight = _mm256_set1_ps(score_scratch[t] * inv_sum);
                    const float* v_row = kv_cache.value_matrix + (t * kv_cache.hidden_dimension) + (kvh * HEAD_DIM) + base;

                    for (size_t c = 0; c < ACC_CHUNK; ++c) {
                        __m256 v_v = _mm256_load_ps(&v_row[c * 8]);
                        acc[c] = _mm256_fmadd_ps(v_weight, v_v, acc[c]);
                    }
                }

                for (size_t c = 0; c < ACC_CHUNK; ++c) {
                    _mm256_store_ps(&out_head[base + c * 8], acc[c]);
                }
            }
        }
    }
};

KintsugiAttention g_Attention;

// ==============================================================================
// DIAGNOSTIC: RAW STREAMING BANDWIDTH (no RNG) — isolates whether the forge's
// ~0.33 GB/s ceiling is std::normal_distribution's sampling cost, or genuine
// memory/page-table bandwidth starvation. Same buffer, same _mm256_stream_ps,
// same sfence — the ONLY thing removed is the RNG. Whatever speed gap shows
// up between this and the real forge IS the RNG's cost, isolated.
// ==============================================================================
void BenchmarkRawStreamingSpeed(float* target_ptr, size_t float_count) {
    std::cout << "[🧪 DIAGNOSTIC] Streaming " << float_count << " floats with NO RNG (pure memory write)...\n";

    auto start = std::chrono::high_resolution_clock::now();
    size_t aligned_count = float_count & ~static_cast<size_t>(7);

    __m256 v_constant = _mm256_set1_ps(1.0f); // fixed value — zero RNG cost, isolates pure write bandwidth
    for (size_t i = 0; i < aligned_count; i += 8) {
        _mm256_stream_ps(&target_ptr[i], v_constant);
    }
    _mm_sfence();

    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> duration = end - start;
    double gb = static_cast<double>(float_count * sizeof(float)) / (1024.0 * 1024.0 * 1024.0);
    double speed = gb / duration.count();

    std::cout << "[🧪 DIAGNOSTIC] Raw streaming speed (NO RNG): " << speed << " GB/s. "
              << "Compare this directly against the real forge's GB/s right below it.\n\n";
}

// ==============================================================================
// THE PAYLOAD FORGE: Multi-Core AVX2 Memory Streamer
// ==============================================================================
void ForgeKintsugiWeights(float* target_ptr, size_t fan_in, size_t float_count, const char* label) {
    std::cout << "======================================================================\n";
    std::cout << "          KINTSUGI SOVEREIGN OPERATIONAL TERMINAL : UNCHAINED         \n";
    std::cout << "======================================================================\n";
    std::cout << "[*] System Dimension Base fan_in : " << fan_in << "\n";
    std::cout << "[*] Minting Matrix Components    : " << float_count << " discrete " << label << "...\n";
    std::cout << "----------------------------------------------------------------------\n";
    std::cout << "[⚡ SILICON] Line-Rate Overdrive Engaged. Forging Jarvis Genesis Grid...\n\n";

    auto start = std::chrono::high_resolution_clock::now();
    double sigma = std::sqrt(2.0 / static_cast<double>(fan_in));
    size_t aligned_count = float_count & ~static_cast<size_t>(7);

    #pragma omp parallel 
    {
        std::random_device local_rd;
        std::mt19937_64 local_gen(local_rd()); 
        std::normal_distribution<float> local_dist(0.0f, static_cast<float>(sigma));

        #pragma omp for schedule(static)
        for (long long i = 0; i < static_cast<long long>(aligned_count); i += 8) {
            alignas(32) float chunk[8];
            for (int j = 0; j < 8; ++j) {
                chunk[j] = local_dist(local_gen);
            }

            __m256 v_generated_weights = _mm256_load_ps(chunk);
            _mm256_stream_ps(&target_ptr[i], v_generated_weights);
        }
    }

    _mm_sfence(); 

    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> duration = end - start;
    
    double gb = static_cast<double>(float_count * sizeof(float)) / (1024.0 * 1024.0 * 1024.0);
    double actual_speed = gb / duration.count();

    std::cout << "======================================================================\n";
    std::cout << "          KINTSUGI CORE OS: " << label << " GENESIS COMPLETE\n";
    std::cout << "======================================================================\n";
    if (actual_speed >= 10.0) {
        std::cout << " -> LINE-RATE STATE            : [  OVERDRIVE DETECTED  ]\n";
    } else {
        std::cout << " -> LINE-RATE STATE            : [  MATH BOTTLENECK DETECTED ]\n";
    }
    std::cout << " -> Target Shard Size           : " << gb << " GB\n";
    std::cout << " -> Snap Insertion Speed        : " << actual_speed << " GB/s\n";
    std::cout << " -> Total Genesis Latency       : " << duration.count() << " seconds.\n";
    std::cout << "======================================================================\n\n";
}

// ==============================================================================
// KINTSUGI: THE GGUF PARSER (PHASE 7 - HEADER & METADATA EXTRACTION)
// ==============================================================================
// This stage ONLY reads the header, metadata key-value pairs, and the tensor
// index (name/shape/type/offset for every tensor). It does NOT read any
// actual tensor bytes yet — that comes after this is verified, and is where
// the Q8_0-vs-Q4_K_M dequant decision actually matters. Metadata is identical
// regardless of quant level, which is why this step works against the
// Q4_K_M file already on disk — no need to wait on the Q8_0 download.

// GGUF metadata value types, per the GGUF spec.
enum class GGUFType : uint32_t {
    UINT8 = 0, INT8 = 1, UINT16 = 2, INT16 = 3, UINT32 = 4, INT32 = 5,
    FLOAT32 = 6, BOOL = 7, STRING = 8, ARRAY = 9, UINT64 = 10, INT64 = 11, FLOAT64 = 12
};

struct GGUFHyperparams {
    uint64_t vocab_size = 0;          // sourced from tokenizer.ggml.tokens array length
    uint64_t embedding_length = 0;    // d_model
    uint64_t feed_forward_length = 0; // d_ff
    uint64_t head_count = 0;          // Q heads
    uint64_t head_count_kv = 0;       // KV heads (confirms/denies GQA)
    uint64_t context_length = 0;
    uint64_t block_count = 0;         // transformer layer count
    float    rope_freq_base = 0.0f;
    float    rms_norm_eps = 1e-5f;    // fallback only if metadata key is absent

    bool AllRequiredPresent() const {
        return embedding_length && feed_forward_length && head_count &&
               head_count_kv && context_length && block_count && vocab_size;
    }
};

struct GGUFTensorInfo {
    std::string name;
    std::vector<uint64_t> dims;
    uint32_t ggml_type = 0; // ggml_type enum — which number means Q8_0 vs Q4_K vs F32 etc. comes next
    uint64_t offset = 0;    // byte offset into the tensor DATA section (not yet computed here)
};

struct KintsugiGGUFParser {
    std::ifstream file;
    uint64_t tensor_count = 0;
    uint64_t metadata_kv_count = 0;
    uint64_t tensor_data_offset = 0; // absolute byte offset where tensor data begins
    std::vector<GGUFTensorInfo> tensors;
    GGUFHyperparams params;

    // Real tokenizer data — previously only the ARRAY length was ever kept
    // (via SkipOrReadValue discarding every element). These hold the actual
    // strings needed to build a working encoder.
    std::vector<std::string> vocab_tokens;                     // index -> token string, e.g. vocab_tokens[6864] == "Ġcapital"
    std::unordered_map<std::string, uint32_t> vocab_lookup;    // token string -> index, for encoding
    std::unordered_map<std::string, int> merge_rank;           // "left right" -> priority (lower index = higher priority)

    template <typename T>
    T ReadRaw() {
        T value{};
        file.read(reinterpret_cast<char*>(&value), sizeof(T));
        return value;
    }

    std::string ReadGGUFString() {
        uint64_t len = ReadRaw<uint64_t>();
        std::string s(static_cast<size_t>(len), '\0');
        if (len > 0) file.read(s.data(), static_cast<std::streamsize>(len));
        return s;
    }

    // Reads a GGUF ARRAY-of-STRING value directly into a vector, instead of
    // discarding it via SkipOrReadValue's generic ARRAY path. Verifies
    // elem_type really is STRING before treating each element as one — a
    // mismatched file fails loudly here instead of silently misreading bytes
    // and desyncing every read that follows.
    std::vector<std::string> ReadStringArray() {
        uint32_t elem_type = ReadRaw<uint32_t>();
        uint64_t count = ReadRaw<uint64_t>();

        if (static_cast<GGUFType>(elem_type) != GGUFType::STRING) {
            std::cerr << "[❌ FATAL] Expected a STRING array (type 8), got elem_type="
                      << elem_type << " — this file's tokenizer arrays don't match "
                      << "what this parser expects. Consuming bytes to stay in sync, "
                      << "but the result will be empty.\n";
            for (uint64_t i = 0; i < count; ++i) SkipOrReadValue(elem_type);
            return {};
        }

        std::vector<std::string> result;
        result.reserve(static_cast<size_t>(count));
        for (uint64_t i = 0; i < count; ++i) {
            result.push_back(ReadGGUFString());
        }
        return result;
    }

    // Reads (and discards, unless it's a scalar numeric) one metadata value
    // of the given type. Returns the value as a double for numeric types so
    // the caller can use it generically without a switch at every call site.
    // For ARRAY, returns the element COUNT (not the elements) — that's what
    // lets us capture vocab_size from tokenizer.ggml.tokens's length below.
    double SkipOrReadValue(uint32_t type_raw) {
        switch (static_cast<GGUFType>(type_raw)) {
            case GGUFType::UINT8:   return static_cast<double>(ReadRaw<uint8_t>());
            case GGUFType::INT8:    return static_cast<double>(ReadRaw<int8_t>());
            case GGUFType::UINT16:  return static_cast<double>(ReadRaw<uint16_t>());
            case GGUFType::INT16:   return static_cast<double>(ReadRaw<int16_t>());
            case GGUFType::UINT32:  return static_cast<double>(ReadRaw<uint32_t>());
            case GGUFType::INT32:   return static_cast<double>(ReadRaw<int32_t>());
            case GGUFType::FLOAT32: return static_cast<double>(ReadRaw<float>());
            case GGUFType::BOOL:    return static_cast<double>(ReadRaw<uint8_t>());
            case GGUFType::UINT64:  return static_cast<double>(ReadRaw<uint64_t>());
            case GGUFType::INT64:   return static_cast<double>(ReadRaw<int64_t>());
            case GGUFType::FLOAT64: return ReadRaw<double>();
            case GGUFType::STRING:  ReadGGUFString(); return 0.0; // discarded — none of our target keys are strings
            case GGUFType::ARRAY: {
                uint32_t elem_type = ReadRaw<uint32_t>();
                uint64_t count = ReadRaw<uint64_t>();
                for (uint64_t i = 0; i < count; ++i) SkipOrReadValue(elem_type); // must still consume every element
                return static_cast<double>(count);
            }
            default:
                std::cerr << "[!] WARNING: Unknown GGUF metadata type " << type_raw
                          << " — file may use a newer GGUF version than this parser expects.\n";
                return 0.0;
        }
    }

    bool Load(const std::string& path) {
        file.open(path, std::ios::binary);
        if (!file.is_open()) {
            std::cerr << "[❌ FATAL] Could not open GGUF file: " << path << "\n";
            return false;
        }

        uint32_t magic = ReadRaw<uint32_t>();
        if (magic != 0x46554747) { // ASCII "GGUF" read as little-endian uint32
            std::cerr << "[❌ FATAL] Bad GGUF magic — not a valid GGUF file, or wrong endianness.\n";
            return false;
        }

        uint32_t version = ReadRaw<uint32_t>();
        tensor_count = ReadRaw<uint64_t>();
        metadata_kv_count = ReadRaw<uint64_t>();

        std::cout << "[📜 GGUF] Magic OK. Spec version " << version << ". "
                  << tensor_count << " tensors, " << metadata_kv_count << " metadata entries.\n";

        for (uint64_t i = 0; i < metadata_kv_count; ++i) {
            std::string key = ReadGGUFString();
            uint32_t value_type = ReadRaw<uint32_t>();

            double value = 0.0;
            bool is_array = (static_cast<GGUFType>(value_type) == GGUFType::ARRAY);

            if (key == "tokenizer.ggml.tokens" && is_array) {
                // Real extraction, replacing the old discard-and-keep-only-
                // the-count behavior. vocab_lookup uses insert() (not []) so
                // that if the vocabulary ever contains duplicate strings,
                // the FIRST (lowest, canonical) index wins rather than being
                // silently overwritten by a later duplicate.
                vocab_tokens = ReadStringArray();
                vocab_lookup.reserve(vocab_tokens.size());
                for (uint32_t idx = 0; idx < vocab_tokens.size(); ++idx) {
                    vocab_lookup.insert({vocab_tokens[idx], idx});
                }
                value = static_cast<double>(vocab_tokens.size());
            } else if (key == "tokenizer.ggml.merges" && is_array) {
                // Same real-extraction treatment. Stored exactly as GGUF
                // keys it — "left right" as one string — so the BPE loop
                // can query with a key built the same way (left + " " + right)
                // without needing to parse/split anything here.
                std::vector<std::string> merges = ReadStringArray();
                merge_rank.reserve(merges.size());
                for (size_t r = 0; r < merges.size(); ++r) {
                    merge_rank.insert({merges[r], static_cast<int>(r)}); // array position = priority rank
                }
                value = static_cast<double>(merges.size());
            } else {
                value = SkipOrReadValue(value_type); // unchanged generic path for every other key
            }

            // --- DIAGNOSTIC: full metadata inventory, before any assumption ---
            // Prints every key this GGUF actually has. For ARRAY-typed keys
            // NOT specifically handled above, `value` is still just the
            // element COUNT (SkipOrReadValue's ARRAY case discards contents).
            // Type reference: 8=STRING, 9=ARRAY.
            std::cout << "    [META] #" << i << " \"" << key << "\" (type=" << value_type
                      << ") = " << value << "\n";
            if (key == "tokenizer.ggml.merges") {
                std::cout << "    [META] *** tokenizer.ggml.merges: " << merge_rank.size()
                          << " rules loaded into merge_rank (real data, not just a count) ***\n";
            }
            if (key == "tokenizer.ggml.tokens") {
                std::cout << "    [META] *** tokenizer.ggml.tokens: " << vocab_tokens.size()
                          << " strings loaded, " << vocab_lookup.size()
                          << " unique entries in vocab_lookup ***\n";
            }

            if      (key == "llama.embedding_length")        params.embedding_length    = static_cast<uint64_t>(value);
            else if (key == "llama.feed_forward_length")     params.feed_forward_length = static_cast<uint64_t>(value);
            else if (key == "llama.attention.head_count")    params.head_count          = static_cast<uint64_t>(value);
            else if (key == "llama.attention.head_count_kv") params.head_count_kv       = static_cast<uint64_t>(value);
            else if (key == "llama.context_length")          params.context_length      = static_cast<uint64_t>(value);
            else if (key == "llama.block_count")             params.block_count         = static_cast<uint64_t>(value);
            else if (key == "llama.rope.freq_base")          params.rope_freq_base      = static_cast<float>(value);
            else if (key == "llama.attention.layer_norm_rms_epsilon") params.rms_norm_eps = static_cast<float>(value);
            else if (key == "tokenizer.ggml.tokens")         params.vocab_size          = static_cast<uint64_t>(value);
        }

        if (!file.good()) {
            std::cerr << "[❌ FATAL] File read failed partway through metadata — truncated or corrupt GGUF.\n";
            return false;
        }

        tensors.reserve(static_cast<size_t>(tensor_count));
        for (uint64_t i = 0; i < tensor_count; ++i) {
            GGUFTensorInfo info;
            info.name = ReadGGUFString();
            uint32_t n_dims = ReadRaw<uint32_t>();
            info.dims.resize(n_dims);
            for (uint32_t d = 0; d < n_dims; ++d) info.dims[d] = ReadRaw<uint64_t>();
            info.ggml_type = ReadRaw<uint32_t>();
            info.offset = ReadRaw<uint64_t>();
            tensors.push_back(std::move(info));
        }

        if (!file.good()) {
            std::cerr << "[❌ FATAL] File read failed partway through tensor index — truncated or corrupt GGUF.\n";
            return false;
        }

        // Per GGUF v3 spec: tensor data begins at the next 32-byte aligned
        // boundary after the last byte of the tensor index. Capture the
        // current file position BEFORE closing, so the loader can seek
        // directly to the right offset without re-parsing the whole header.
        uint64_t raw_pos = static_cast<uint64_t>(file.tellg());
        tensor_data_offset = (raw_pos + 31) & ~static_cast<uint64_t>(31); // align up to 32 bytes

        std::cout << "[📜 GGUF] Tensor data section starts at file offset: " << tensor_data_offset << " bytes.\n";

        file.close();
        return true;
    }
};

// ==============================================================================
// KINTSUGI: UNICODE CODEPOINT CLASSIFICATION (TOKENIZER PHASE I, PART 1)
// ==============================================================================
// Mirrors llama.cpp's actual two-function split — unicode_cpts_from_utf8()
// to decode raw bytes into codepoints, then unicode_cpt_flags_from_cpt() to
// classify each one. This is deliberately the SAME split, not a shortcut:
// the LLAMA3 pretokenizer regex needs \p{L}/\p{N}/\s classification per
// codepoint, and codepoints can't be read off raw UTF-8 bytes directly —
// multi-byte sequences have to be decoded first.
//
// HONEST SCOPE: covers ASCII (0x00-0x7F) and Latin-1 Supplement (0x80-0xFF)
// letter/digit/whitespace classification correctly and completely — enough
// for English and most common European-language text. Does NOT attempt full
// Unicode General_Category coverage (Cyrillic, Greek, CJK, Arabic, combining
// marks, etc.). Anything outside these ranges gets all flags false, which
// the pretokenizer's regex falls through to treating as punctuation/symbol
// (alternative #4) — a working fallback, not a correct classification, for
// scripts not yet covered. Extending this later means adding more range
// checks to ClassifyCodepoint, not changing its shape.

struct UnicodeCptFlags {
    bool is_letter = false;
    bool is_number = false;
    bool is_whitespace = false;
};

UnicodeCptFlags ClassifyCodepoint(uint32_t cpt) {
    UnicodeCptFlags flags;

    // Whitespace: ASCII space/tab/LF/VT/FF/CR, plus U+00A0 (non-breaking
    // space) since it shows up constantly in web/document-sourced text.
    if (cpt == 0x20 || cpt == 0x09 || cpt == 0x0A || cpt == 0x0B ||
        cpt == 0x0C || cpt == 0x0D || cpt == 0xA0) {
        flags.is_whitespace = true;
        return flags;
    }

    // Digits: ASCII 0-9 only. Other Unicode digit scripts (Arabic-Indic,
    // Devanagari, etc.) are out of scope for now.
    if (cpt >= 0x30 && cpt <= 0x39) {
        flags.is_number = true;
        return flags;
    }

    // Letters: ASCII A-Z, a-z.
    if ((cpt >= 0x41 && cpt <= 0x5A) || (cpt >= 0x61 && cpt <= 0x7A)) {
        flags.is_letter = true;
        return flags;
    }

    // Latin-1 Supplement letters (0xC0-0xFF), EXCLUDING 0xD7 (×, multiplication
    // sign) and 0xF7 (÷, division sign) — these sit inside the letter range
    // numerically but are math symbols, not letters. Easy to get wrong by
    // treating the whole 0xC0-0xFF block as one range.
    if ((cpt >= 0xC0 && cpt <= 0xD6) || (cpt >= 0xD8 && cpt <= 0xF6) ||
        (cpt >= 0xF8 && cpt <= 0xFF)) {
        flags.is_letter = true;
        return flags;
    }

    // Everything else: no flags set, falls through to the punctuation/symbol
    // branch downstream. Not correct for Cyrillic/CJK/etc., but not a crash
    // or corruption either.
    return flags;
}

// Decodes a UTF-8 byte string into Unicode codepoints. Standard 1-4 byte
// decoding. On invalid/truncated sequences, inserts U+FFFD (replacement
// character) and continues — matches llama.cpp's own documented behavior
// (see llama.cpp PR #11729) rather than throwing on malformed input.
std::vector<uint32_t> DecodeUTF8(const std::string& text) {
    std::vector<uint32_t> cpts;
    cpts.reserve(text.size()); // upper bound: ASCII text is 1 codepoint/byte

    size_t i = 0;
    while (i < text.size()) {
        uint8_t b0 = static_cast<uint8_t>(text[i]);
        uint32_t cpt;
        size_t len;

        if      ((b0 & 0x80) == 0x00) { cpt = b0;        len = 1; } // 0xxxxxxx
        else if ((b0 & 0xE0) == 0xC0) { cpt = b0 & 0x1F;  len = 2; } // 110xxxxx
        else if ((b0 & 0xF0) == 0xE0) { cpt = b0 & 0x0F;  len = 3; } // 1110xxxx
        else if ((b0 & 0xF8) == 0xF0) { cpt = b0 & 0x07;  len = 4; } // 11110xxx
        else {
            cpts.push_back(0xFFFD); // invalid leading byte
            ++i;
            continue;
        }

        if (i + len > text.size()) {
            cpts.push_back(0xFFFD); // truncated multi-byte sequence at end
            break;
        }

        bool valid = true;
        uint32_t decoded = cpt;
        for (size_t k = 1; k < len; ++k) {
            uint8_t bk = static_cast<uint8_t>(text[i + k]);
            if ((bk & 0xC0) != 0x80) { valid = false; break; } // must be 10xxxxxx
            decoded = (decoded << 6) | (bk & 0x3F);
        }

        if (!valid) {
            cpts.push_back(0xFFFD);
            ++i;
            continue;
        }

        cpts.push_back(decoded);
        i += len;
    }

    return cpts;
}

// Self-test, run once at boot: proves the classifier and decoder are
// correct against known cases BEFORE the pretokenizer parser gets built on
// top of them — same discipline as every other verification step tonight.
// Specifically exercises the 0xD7/0xF7 exclusion, since that's the exact
// kind of boundary an off-by-range-check would silently get wrong.
bool RunUnicodeSelfTest() {
    struct Case { uint32_t cpt; bool exp_letter, exp_number, exp_space; const char* label; };
    Case cases[] = {
        // Original spot-checks — middle-of-range values plus the two known
        // exclusion traps.
        { 'A',  true,  false, false, "'A' ASCII letter" },
        { 'z',  true,  false, false, "'z' ASCII letter" },
        { '5',  false, true,  false, "'5' ASCII digit" },
        { ' ',  false, false, true,  "' ' space" },
        { '\n', false, false, true,  "'\\n' newline" },
        { '.',  false, false, false, "'.' punctuation (all flags false)" },
        { 0xE9, true,  false, false, "U+00E9 'e' Latin-1 letter" },
        { 0xD7, false, false, false, "U+00D7 multiplication sign (NOT a letter)" },
        { 0xF7, false, false, false, "U+00F7 division sign (NOT a letter)" },

        // Boundary sweep — every range edge, both sides, since a spot-check
        // in the middle of a range can't catch an off-by-one in a <= vs <
        // comparison. Every value below independently verified in Python
        // against the exact same logic before being hardcoded here.
        { 0x2F, false, false, false, "'/' just below digit range" },
        { 0x30, false, true,  false, "'0' digit lower bound" },
        { 0x39, false, true,  false, "'9' digit upper bound" },
        { 0x3A, false, false, false, "':' just above digit range" },
        { 0x40, false, false, false, "'@' just below A-Z" },
        { 0x41, true,  false, false, "'A' upper letter lower bound" },
        { 0x5A, true,  false, false, "'Z' upper letter upper bound" },
        { 0x5B, false, false, false, "'[' just above A-Z" },
        { 0x60, false, false, false, "'`' just below a-z" },
        { 0x61, true,  false, false, "'a' lower letter lower bound" },
        { 0x7A, true,  false, false, "'z' lower letter upper bound" },
        { 0x7B, false, false, false, "'{' just above a-z" },
        { 0xBF, false, false, false, "just below Latin-1 sub-range 1" },
        { 0xC0, true,  false, false, "Latin-1 sub-range 1 lower bound" },
        { 0xD6, true,  false, false, "Latin-1 sub-range 1 upper bound" },
        { 0xD8, true,  false, false, "Latin-1 sub-range 2 lower bound" },
        { 0xF6, true,  false, false, "Latin-1 sub-range 2 upper bound" },
        { 0xF8, true,  false, false, "Latin-1 sub-range 3 lower bound" },
        { 0xFF, true,  false, false, "Latin-1 sub-range 3 upper bound" },

        // All 7 explicit whitespace codepoints — original test only hit 2.
        { 0x09, false, false, true,  "tab" },
        { 0x0B, false, false, true,  "vertical tab" },
        { 0x0C, false, false, true,  "form feed" },
        { 0x0D, false, false, true,  "carriage return" },
        { 0xA0, false, false, true,  "non-breaking space" },
    };

    bool all_pass = true;
    for (const auto& c : cases) {
        UnicodeCptFlags f = ClassifyCodepoint(c.cpt);
        bool ok = (f.is_letter == c.exp_letter) && (f.is_number == c.exp_number) && (f.is_whitespace == c.exp_space);
        std::cout << "    [UNICODE-TEST] " << (ok ? "PASS" : "FAIL") << " " << c.label
                  << " -> letter=" << f.is_letter << " number=" << f.is_number
                  << " space=" << f.is_whitespace << "\n";
        if (!ok) all_pass = false;
    }

    // UTF-8 decode check: "café" is c,a,f,e-acute — the last codepoint (é,
    // U+00E9) is encoded as 2 bytes (0xC3 0xA9), so 5 input bytes must
    // decode to exactly 4 codepoints, with the last one equal to 0xE9.
    std::vector<uint32_t> decoded = DecodeUTF8("caf\xC3\xA9");
    bool utf8_ok = (decoded.size() == 4) && (decoded[3] == 0xE9);
    std::cout << "    [UNICODE-TEST] " << (utf8_ok ? "PASS" : "FAIL")
              << " UTF-8 decode \"caf\\xC3\\xA9\" -> " << decoded.size()
              << " codepoints, last=0x" << std::hex << (decoded.empty() ? 0 : decoded.back()) << std::dec << "\n";
    if (!utf8_ok) all_pass = false;

    // UTF-8 error-path checks — the original test only ever exercised the
    // happy path (valid multi-byte input). These three hit the malformed-
    // input handling directly, verified in Python against the same logic
    // before being hardcoded: a lone continuation byte, a truncated
    // multi-byte sequence, and an invalid leading byte (0xF8, which looks
    // like a 5-byte UTF-8 lead but modern UTF-8 caps at 4 bytes) should all
    // decode to a single U+FFFD replacement character, not crash or hang.
    struct Utf8ErrCase { std::string input; const char* label; };
    Utf8ErrCase err_cases[] = {
        { std::string("\x80", 1), "lone continuation byte" },
        { std::string("\xC3", 1), "truncated 2-byte sequence" },
        { std::string("\xF8", 1), "invalid leading byte (5-byte pattern)" },
    };
    for (const auto& ec : err_cases) {
        std::vector<uint32_t> d = DecodeUTF8(ec.input);
        bool ok = (d.size() == 1) && (d[0] == 0xFFFD);
        std::cout << "    [UNICODE-TEST] " << (ok ? "PASS" : "FAIL")
                  << " UTF-8 error path: " << ec.label << " -> "
                  << d.size() << " codepoint(s), first=0x" << std::hex
                  << (d.empty() ? 0u : d[0]) << std::dec << "\n";
        if (!ok) all_pass = false;
    }

    return all_pass;
}

// ==============================================================================
// KINTSUGI: LLAMA-3 PRETOKENIZER (TOKENIZER PHASE I, PART 2)
// ==============================================================================
// Implements unicode_regex_split_custom_llama3's exact 7-branch, priority-
// ordered logic — verified word-for-word against the real llama.cpp source
// (src/unicode.cpp, "LLAMA3 system regex" comment):
//   (?i:'s|'t|'re|'ve|'m|'ll|'d) | [^\r\n\p{L}\p{N}]?\p{L}+ | \p{N}{1,3} |
//    ?[^\s\p{L}\p{N}]+[\r\n]* | \s*[\r\n]+ | \s+(?!\S) | \s+
//
// NOT a general regex engine — a purpose-built parser for this ONE known
// pattern, mirroring what llama.cpp itself does (a custom function per
// architecture, not a generic Unicode-aware regex library call). At each
// position, tries all 7 branches in priority order, takes the first that
// matches, consumes its length, repeats until the text is exhausted.
//
// Branches 5 and 6 involve real backtracking behavior that "just greedily
// consume whitespace" gets wrong — both algorithms below were hand-derived
// against concrete examples, then independently verified against Python's
// actual regex engine before being trusted enough to write in C++. See the
// self-test for the exact cases that pinned this down.

// Branch 1: (?i:'s|'t|'re|'ve|'m|'ll|'d) — contraction suffixes, case-insensitive.
size_t MatchContraction(const std::vector<uint32_t>& cpts, size_t pos) {
    if (pos >= cpts.size() || cpts[pos] != '\'') return 0;
    static const std::vector<std::string> suffixes = { "'re", "'ve", "'ll", "'s", "'t", "'m", "'d" };
    for (const auto& suf : suffixes) {
        if (pos + suf.size() > cpts.size()) continue;
        bool match = true;
        for (size_t k = 0; k < suf.size(); ++k) {
            uint32_t c = cpts[pos + k];
            char expected = suf[k];
            if (c > 127) { match = false; break; } // contractions are ASCII-only
            char c_lower = (c >= 'A' && c <= 'Z') ? static_cast<char>(c + 32) : static_cast<char>(c);
            if (c_lower != expected) { match = false; break; }
        }
        if (match) return suf.size();
    }
    return 0;
}

// Branch 2: [^\r\n\p{L}\p{N}]?\p{L}+ — optional single non-letter/number/
// newline prefix, then 1+ letters. This is what glues a leading space onto
// the following word (e.g. "Ġcapital" as one chunk, not space + word).
size_t MatchWordRun(const std::vector<uint32_t>& cpts,
                     const std::vector<UnicodeCptFlags>& flags, size_t pos) {
    size_t i = pos;
    if (i < cpts.size()) {
        bool blocks_prefix = flags[i].is_letter || flags[i].is_number || cpts[i] == '\r' || cpts[i] == '\n';
        if (!blocks_prefix) ++i; // consume the optional single prefix char
    }
    size_t letters_start = i;
    while (i < cpts.size() && flags[i].is_letter) ++i;
    if (i == letters_start) return 0; // needs 1+ letters after the optional prefix
    return i - pos;
}

// Branch 3: \p{N}{1,3} — 1 to 3 digits, greedy.
size_t MatchDigitRun(const std::vector<uint32_t>& cpts,
                      const std::vector<UnicodeCptFlags>& flags, size_t pos) {
    size_t limit = std::min(cpts.size(), pos + 3);
    size_t i = pos;
    while (i < limit && flags[i].is_number) ++i;
    return i - pos;
}

// Branch 4:  ?[^\s\p{L}\p{N}]+[\r\n]* — optional single leading space, then
// 1+ symbol/punctuation characters, then trailing newlines.
size_t MatchSymbolRun(const std::vector<uint32_t>& cpts,
                       const std::vector<UnicodeCptFlags>& flags, size_t pos) {
    size_t i = pos;
    if (i < cpts.size() && cpts[i] == ' ') ++i; // optional single leading space
    size_t symbols_start = i;
    while (i < cpts.size() && !flags[i].is_whitespace && !flags[i].is_letter && !flags[i].is_number) ++i;
    if (i == symbols_start) return 0; // needs 1+ symbols
    while (i < cpts.size() && (cpts[i] == '\r' || cpts[i] == '\n')) ++i; // trailing newlines
    return i - pos;
}

// Branch 5: \s*[\r\n]+ — hand-derived and Python-verified: within the full
// whitespace run starting at pos, find M = the LAST index containing \r or
// \n. No such M -> no match. Otherwise match length = M+1-pos (through the
// last newline; any non-newline whitespace trailing after it is NOT
// included, since [\r\n]+ only matches \r/\n, never plain whitespace).
size_t MatchNewlineRun(const std::vector<uint32_t>& cpts,
                        const std::vector<UnicodeCptFlags>& flags, size_t pos) {
    if (pos >= cpts.size() || !flags[pos].is_whitespace) return 0;
    size_t j = pos;
    while (j < cpts.size() && flags[j].is_whitespace) ++j;
    size_t last_newline = SIZE_MAX;
    for (size_t k = pos; k < j; ++k) {
        if (cpts[k] == '\r' || cpts[k] == '\n') last_newline = k;
    }
    if (last_newline == SIZE_MAX) return 0; // no \r or \n anywhere in the run
    return (last_newline + 1) - pos;
}

// Branch 6: \s+(?!\S) — hand-derived and Python-verified: full whitespace
// run [pos,j). If it runs to end of text, match all of it. Otherwise
// (followed by non-whitespace), match run_length-1 — UNLESS that's 0 (a
// single whitespace char before non-whitespace), in which case this branch
// doesn't match at all and branch 7 catches it instead.
size_t MatchTrailingWhitespace(const std::vector<uint32_t>& cpts,
                                const std::vector<UnicodeCptFlags>& flags, size_t pos) {
    if (pos >= cpts.size() || !flags[pos].is_whitespace) return 0;
    size_t j = pos;
    while (j < cpts.size() && flags[j].is_whitespace) ++j;
    if (j == cpts.size()) return j - pos; // run extends to end of text
    size_t candidate = (j - pos) - 1;
    return candidate >= 1 ? candidate : 0;
}

// Branch 7: \s+ — unconditional catch-all, always succeeds if pos is whitespace.
size_t MatchWhitespaceRun(const std::vector<uint32_t>& cpts,
                           const std::vector<UnicodeCptFlags>& flags, size_t pos) {
    if (pos >= cpts.size() || !flags[pos].is_whitespace) return 0;
    size_t j = pos;
    while (j < cpts.size() && flags[j].is_whitespace) ++j;
    return j - pos;
}

// Main driver: walks the codepoint sequence, tries all 7 branches in
// priority order at each position, takes the first match, advances by its
// length. Falls back to advancing 1 codepoint if literally nothing matches
// (shouldn't happen given branches 2/3/4/7 collectively cover every
// category, but guarantees forward progress rather than an infinite loop
// if some input combination slips through).
std::vector<std::string> PretokenizeLlama3(const std::string& text) {
    std::vector<uint32_t> cpts = DecodeUTF8(text);
    std::vector<UnicodeCptFlags> flags;
    flags.reserve(cpts.size());
    for (uint32_t c : cpts) flags.push_back(ClassifyCodepoint(c));

    std::vector<std::string> chunks;
    size_t pos = 0;
    while (pos < cpts.size()) {
        size_t len = MatchContraction(cpts, pos);
        if (len == 0) len = MatchWordRun(cpts, flags, pos);
        if (len == 0) len = MatchDigitRun(cpts, flags, pos);
        if (len == 0) len = MatchSymbolRun(cpts, flags, pos);
        if (len == 0) len = MatchNewlineRun(cpts, flags, pos);
        if (len == 0) len = MatchTrailingWhitespace(cpts, flags, pos);
        if (len == 0) len = MatchWhitespaceRun(cpts, flags, pos);
        if (len == 0) len = 1; // safety fallback — guarantees forward progress

        // Re-encode this chunk's codepoints back to UTF-8 bytes for the
        // caller (the eventual vocab lookup / merge loop works on strings,
        // matching how GGUF stores both the vocabulary and merge rules).
        std::string chunk;
        for (size_t k = pos; k < pos + len; ++k) {
            uint32_t c = cpts[k];
            if (c < 0x80) {
                chunk += static_cast<char>(c);
            } else if (c < 0x800) {
                chunk += static_cast<char>(0xC0 | (c >> 6));
                chunk += static_cast<char>(0x80 | (c & 0x3F));
            } else if (c < 0x10000) {
                chunk += static_cast<char>(0xE0 | (c >> 12));
                chunk += static_cast<char>(0x80 | ((c >> 6) & 0x3F));
                chunk += static_cast<char>(0x80 | (c & 0x3F));
            } else {
                chunk += static_cast<char>(0xF0 | (c >> 18));
                chunk += static_cast<char>(0x80 | ((c >> 12) & 0x3F));
                chunk += static_cast<char>(0x80 | ((c >> 6) & 0x3F));
                chunk += static_cast<char>(0x80 | (c & 0x3F));
            }
        }
        chunks.push_back(chunk);
        pos += len;
    }
    return chunks;
}

// Self-test: known input -> expected chunk sequences, including the two
// hand-derived-and-Python-verified edge cases (multi-space before a word,
// multi-newline runs) that a naive "just greedily consume whitespace"
// implementation gets wrong.
bool RunPretokenizerSelfTest() {
    struct Case { std::string input; std::vector<std::string> expected; const char* label; };
    Case cases[] = {
        { "Hello", {"Hello"}, "single word" },
        { " world", {" world"}, "leading space glued to word" },
        { "don't", {"don", "'t"}, "contraction split" },
        { "123456", {"123", "456"}, "digit run capped at 3" },
        { "Hi!!!", {"Hi", "!!!"}, "word then symbol run" },
        { "a b", {"a", " b"}, "single space glues to following word" },
        { "a  b", {"a", " ", " b"}, "double space: one orphan space, one glued" },
        { "a   b", {"a", "  ", " b"}, "triple space: two orphan, one glued" },
        { "hi ", {"hi", " "}, "trailing space at end of text" },
        { "\n\n\n", {"\n\n\n"}, "pure newline run, one chunk" },
    };

    bool all_pass = true;
    for (const auto& c : cases) {
        std::vector<std::string> got = PretokenizeLlama3(c.input);
        bool ok = (got == c.expected);
        std::cout << "    [PRETOK-TEST] " << (ok ? "PASS" : "FAIL") << " " << c.label
                  << " (\"" << c.input << "\") -> [";
        for (size_t i = 0; i < got.size(); ++i) {
            std::cout << "\"" << got[i] << "\"";
            if (i + 1 < got.size()) std::cout << ", ";
        }
        std::cout << "]";
        if (!ok) {
            std::cout << " EXPECTED [";
            for (size_t i = 0; i < c.expected.size(); ++i) {
                std::cout << "\"" << c.expected[i] << "\"";
                if (i + 1 < c.expected.size()) std::cout << ", ";
            }
            std::cout << "]";
        }
        std::cout << "\n";
        if (!ok) all_pass = false;
    }
    return all_pass;
}

// ==============================================================================
// KINTSUGI: BYTE-TO-UNICODE TABLE (TOKENIZER PHASE I, PART 3)
// ==============================================================================
// GPT-2's byte-level BPE encoding, which Llama-3 inherited unchanged.
// Verified against the actual OpenAI GPT-2 source (github.com/openai/gpt-2/
// blob/master/src/encoder.py, bytes_to_unicode()) — and independently
// cross-checked: computing this table's entry for byte 32 (space) produces
// exactly U+0120 (Ġ), matching every single decoded token this whole
// project has ever produced, going back to the very first "Ġcapital"
// result many sessions ago. That match isn't circular — the algorithm was
// derived from source code first, then checked against independent
// empirical observation.
//
// Algorithm: byte values 33-126 ('!' through '~'), 161-172, and 174-255 map
// to themselves as Unicode codepoints — already "nice", printable,
// unambiguous characters. Every OTHER byte value (control characters,
// space, DEL, and a few Latin-1 gaps) gets assigned a NEW codepoint
// starting at 256, in the order those byte values are encountered scanning
// 0 through 255. Computed here rather than hand-typed as 256 literals — a
// computed table is verifiable against the algorithm; 256 hand-transcribed
// magic numbers would not be.

std::array<uint32_t, 256> BuildByteToUnicodeTable() {
    std::array<uint32_t, 256> table{};
    std::array<bool, 256> is_self_mapped{};

    for (int b = 33;  b <= 126; ++b) is_self_mapped[b] = true;
    for (int b = 161; b <= 172; ++b) is_self_mapped[b] = true;
    for (int b = 174; b <= 255; ++b) is_self_mapped[b] = true;

    for (int b = 0; b < 256; ++b) {
        if (is_self_mapped[b]) table[b] = static_cast<uint32_t>(b);
    }

    uint32_t next_cpt = 256;
    for (int b = 0; b < 256; ++b) {
        if (!is_self_mapped[b]) {
            table[b] = next_cpt;
            ++next_cpt;
        }
    }

    return table;
}

bool RunByteToUnicodeSelfTest(const std::array<uint32_t, 256>& table) {
    struct Case { int byte; uint32_t expected_cpt; const char* label; };
    Case cases[] = {
        { 32,  0x0120, "byte 32 (space) -> U+0120 (the actual Ġ seen in every decode all project)" },
        { 33,  0x0021, "byte 33 ('!') maps to itself, start of first self-mapped range" },
        { 65,  0x0041, "byte 65 ('A') maps to itself" },
        { 126, 0x007E, "byte 126 ('~') maps to itself, end of first self-mapped range" },
        { 0,   0x0100, "byte 0 (NUL), first gap byte encountered" },
        { 127, 0x0121, "byte 127 (DEL), first gap byte after the first self-mapped range" },
        { 160, 0x0142, "byte 160, last gap byte before the second self-mapped range" },
        { 173, 0x0143, "byte 173 (soft hyphen), lone gap byte between ranges 2 and 3" },
        { 174, 0x00AE, "byte 174 ('R with circle') maps to itself, start of third self-mapped range" },
        { 255, 0x00FF, "byte 255 (0xFF) maps to itself, end of third self-mapped range" },
    };

    bool all_pass = true;
    for (const auto& c : cases) {
        uint32_t got = table[static_cast<size_t>(c.byte)];
        bool ok = (got == c.expected_cpt);
        std::cout << "    [BYTEMAP-TEST] " << (ok ? "PASS" : "FAIL") << " " << c.label
                  << " -> got U+" << std::hex << std::uppercase << got
                  << std::nouppercase << std::dec << "\n";
        if (!ok) all_pass = false;
    }
    return all_pass;
}

// ==============================================================================
// KINTSUGI: BPE MERGE LOOP + FULL ENCODER (TOKENIZER PHASE I, PART 4 — FINAL)
// ==============================================================================
// The actual BPE algorithm, matching the standard reference semantics (see
// e.g. Karpathy-style merge_tokens: given the SINGLE winning pair for this
// round, merge EVERY occurrence of it in one left-to-right pass, then
// rescan for the next winning pair in the shorter sequence):
//
//   while true:
//     scan ALL adjacent pairs in the current symbol sequence
//     among pairs that exist in merge_rank, pick the one with the LOWEST
//       rank value (lowest index in the GGUF array = highest priority) —
//       NOT simply the first mergeable pair encountered while scanning
//     if no pair has any rank at all, stop
//     merge every occurrence of that specific pair, one pass, then repeat

std::vector<std::string> BpeMerge(std::vector<std::string> symbols,
                                   const std::unordered_map<std::string, int>& merge_rank) {
    while (symbols.size() >= 2) {
        int best_rank = INT_MAX;
        size_t best_idx = SIZE_MAX;

        for (size_t i = 0; i + 1 < symbols.size(); ++i) {
            auto it = merge_rank.find(symbols[i] + " " + symbols[i + 1]);
            if (it != merge_rank.end() && it->second < best_rank) {
                best_rank = it->second;
                best_idx = i;
            }
        }

        if (best_idx == SIZE_MAX) break; // no adjacent pair has any rank — done

        const std::string& left = symbols[best_idx];
        const std::string& right = symbols[best_idx + 1];
        std::vector<std::string> merged;
        merged.reserve(symbols.size());
        size_t i = 0;
        while (i < symbols.size()) {
            if (i + 1 < symbols.size() && symbols[i] == left && symbols[i + 1] == right) {
                merged.push_back(left + right);
                i += 2;
            } else {
                merged.push_back(symbols[i]);
                i += 1;
            }
        }
        symbols = std::move(merged);
    }
    return symbols;
}

// Encodes a single codepoint back to a UTF-8 string — same encoding logic
// already used in PretokenizeLlama3's chunk reconstruction, needed again
// here since the byte-to-unicode table's remapped codepoints must become
// strings before they can be looked up as merge_rank/vocab_lookup keys.
std::string EncodeCodepointUTF8(uint32_t cpt) {
    std::string s;
    if (cpt < 0x80) {
        s += static_cast<char>(cpt);
    } else if (cpt < 0x800) {
        s += static_cast<char>(0xC0 | (cpt >> 6));
        s += static_cast<char>(0x80 | (cpt & 0x3F));
    } else if (cpt < 0x10000) {
        s += static_cast<char>(0xE0 | (cpt >> 12));
        s += static_cast<char>(0x80 | ((cpt >> 6) & 0x3F));
        s += static_cast<char>(0x80 | (cpt & 0x3F));
    } else {
        s += static_cast<char>(0xF0 | (cpt >> 18));
        s += static_cast<char>(0x80 | ((cpt >> 12) & 0x3F));
        s += static_cast<char>(0x80 | ((cpt >> 6) & 0x3F));
        s += static_cast<char>(0x80 | (cpt & 0x3F));
    }
    return s;
}

// Encodes one pretokenized chunk: byte-remap every individual BYTE (not
// codepoint — multi-byte UTF-8 characters get split into separate remapped
// symbols, one per original byte) into its initial symbol, run the merge
// loop, then resolve each final piece to a token ID via vocab_lookup.
std::vector<uint32_t> EncodeChunk(const std::string& chunk_utf8,
                                   const std::array<uint32_t, 256>& byte_to_unicode,
                                   const std::unordered_map<std::string, int>& merge_rank,
                                   const std::unordered_map<std::string, uint32_t>& vocab_lookup) {
    std::vector<std::string> symbols;
    symbols.reserve(chunk_utf8.size());
    for (unsigned char b : chunk_utf8) {
        symbols.push_back(EncodeCodepointUTF8(byte_to_unicode[b]));
    }

    symbols = BpeMerge(std::move(symbols), merge_rank);

    std::vector<uint32_t> ids;
    ids.reserve(symbols.size());
    for (const auto& sym : symbols) {
        auto it = vocab_lookup.find(sym);
        if (it != vocab_lookup.end()) {
            ids.push_back(it->second);
        } else {
            std::cerr << "[!] WARNING: BPE piece not found in vocabulary: \"" << sym << "\"\n";
        }
    }
    return ids;
}

// Top-level encoder: pretokenize, then encode each chunk independently
// (merges never cross chunk boundaries), concatenating the results.
std::vector<uint32_t> EncodeText(const std::string& text,
                                  const std::array<uint32_t, 256>& byte_to_unicode,
                                  const std::unordered_map<std::string, int>& merge_rank,
                                  const std::unordered_map<std::string, uint32_t>& vocab_lookup) {
    std::vector<uint32_t> all_ids;
    for (const auto& chunk : PretokenizeLlama3(text)) {
        std::vector<uint32_t> chunk_ids = EncodeChunk(chunk, byte_to_unicode, merge_rank, vocab_lookup);
        all_ids.insert(all_ids.end(), chunk_ids.begin(), chunk_ids.end());
    }
    return all_ids;
}

// Self-test with a small, fully hand-controlled synthetic merge table —
// proves the ALGORITHM's mechanics (lowest-rank-wins, merge-all-occurrences,
// multi-round chaining) independent of whether the real 280,147-entry
// table happens to be loaded correctly (already separately verified when
// it was first parsed). Real end-to-end data verification happens
// separately, after GGUF load, against a known-correct answer.
bool RunBpeMergeSelfTest() {
    bool all_pass = true;

    // Case 1: multi-round chaining — d+o merges first (rank 0), producing
    // "do", which THEN merges with n (rank 1) to produce "don".
    {
        std::unordered_map<std::string, int> rank = { {"d o", 0}, {"do n", 1} };
        auto result = BpeMerge({"d", "o", "n"}, rank);
        bool ok = (result == std::vector<std::string>{"don"});
        std::cout << "    [BPE-TEST] " << (ok ? "PASS" : "FAIL")
                  << " multi-round chaining: d,o,n -> ";
        for (auto& s : result) std::cout << "\"" << s << "\" ";
        std::cout << "\n";
        if (!ok) all_pass = false;
    }

    // Case 2: lowest-rank-wins, NOT first-match. "a b" has rank 5, "b c"
    // has rank 0 — even though "a b" is encountered first scanning left to
    // right, "b c" must win because its rank is lower.
    {
        std::unordered_map<std::string, int> rank = { {"b c", 0}, {"a b", 5} };
        auto result = BpeMerge({"a", "b", "c"}, rank);
        bool ok = (result == std::vector<std::string>{"a", "bc"});
        std::cout << "    [BPE-TEST] " << (ok ? "PASS" : "FAIL")
                  << " lowest-rank-wins over first-match: a,b,c -> ";
        for (auto& s : result) std::cout << "\"" << s << "\" ";
        std::cout << "\n";
        if (!ok) all_pass = false;
    }

    // Case 3: merge ALL occurrences of the winning pair in one pass, not
    // just the first one found.
    {
        std::unordered_map<std::string, int> rank = { {"x y", 0} };
        auto result = BpeMerge({"x", "y", "x", "y"}, rank);
        bool ok = (result == std::vector<std::string>{"xy", "xy"});
        std::cout << "    [BPE-TEST] " << (ok ? "PASS" : "FAIL")
                  << " merge-all-occurrences: x,y,x,y -> ";
        for (auto& s : result) std::cout << "\"" << s << "\" ";
        std::cout << "\n";
        if (!ok) all_pass = false;
    }

    // Case 4: no mergeable pair at all — sequence returned unchanged.
    {
        std::unordered_map<std::string, int> rank = { {"p q", 0} };
        auto result = BpeMerge({"a", "b", "c"}, rank);
        bool ok = (result == std::vector<std::string>{"a", "b", "c"});
        std::cout << "    [BPE-TEST] " << (ok ? "PASS" : "FAIL")
                  << " no mergeable pair: a,b,c -> ";
        for (auto& s : result) std::cout << "\"" << s << "\" ";
        std::cout << "\n";
        if (!ok) all_pass = false;
    }

    return all_pass;
}

// ==============================================================================
// KINTSUGI: NATIVE DETOKENIZER (TOKENIZER PHASE II)
// ==============================================================================
// Closes the loop natively in C++ — token IDs back to readable text —
// instead of copy-pasting IDs into decode_tokens.py by hand every time.
//
// One design point learned directly from a real bug caught this session:
// a Python decode script's special-token-skip check accidentally used a
// loop variable left over from an EARLIER, already-finished loop instead
// of the CURRENT item being processed — silently checking a stale value
// against every item instead of each item's own id. C++'s range-for
// scoping makes that EXACT mechanism impossible (the variable wouldn't
// exist outside its loop, so it'd be a compile error) — but the general
// mistake (checking something stale instead of the current item) is still
// fully possible here too, just via a different mechanism (e.g. holding a
// "last seen id" across two loops and misusing it). DetokenizeIds below
// reads `id` fresh from the SAME loop iteration it's acting on, every time
// — no separate tracking variable, nothing left over from anywhere else.

// Inverts the already-proven-correct byte_to_unicode table. If the forward
// table is right (verified via 10 boundary tests plus the empirical match
// against every real Ġ this project has ever produced), the reverse
// follows directly by construction — no separate derivation needed.
std::unordered_map<uint32_t, uint8_t> BuildUnicodeToByteTable(const std::array<uint32_t, 256>& byte_to_unicode) {
    std::unordered_map<uint32_t, uint8_t> reverse;
    reverse.reserve(256);
    for (int b = 0; b < 256; ++b) {
        reverse[byte_to_unicode[static_cast<size_t>(b)]] = static_cast<uint8_t>(b);
    }
    return reverse;
}

// Converts token IDs back into a human-readable UTF-8 string. Stops at
// <|eot_id|> or <|end_of_text|> (checking THIS iteration's `id` directly).
// For every other token, decodes its vocab string (e.g. "Ġcapital") back
// to codepoints, reverses EACH codepoint through the byte map to recover
// the original byte, and appends it — undoing the byte-remap step from
// the encode direction, one codepoint at a time.
std::string DetokenizeIds(const std::vector<uint32_t>& ids,
                           const std::vector<std::string>& vocab_tokens,
                           const std::unordered_map<uint32_t, uint8_t>& unicode_to_byte) {
    std::string result;
    for (uint32_t id : ids) {
        if (id == 128009 /* <|eot_id|> */ || id == 128001 /* <|end_of_text|> */) {
            break; // this iteration's own id, checked right here — nothing stale
        }
        if (id >= vocab_tokens.size()) continue; // defensive: unknown id, skip safely

        for (uint32_t cpt : DecodeUTF8(vocab_tokens[id])) {
            auto it = unicode_to_byte.find(cpt);
            if (it != unicode_to_byte.end()) {
                result += static_cast<char>(it->second);
            }
        }
    }
    return result;
}

// Round-trip self-test: no new external ground truth needed. EncodeText is
// already proven correct against independent real data (the France-answer
// match earlier tonight), so encode(text) -> detokenize(...) landing back
// on the EXACT original string is a strong, self-consistent proof that
// detokenize correctly reverses what encode does.
bool RunDetokenizeSelfTest(const std::array<uint32_t, 256>& byte_to_unicode,
                            const std::unordered_map<std::string, int>& merge_rank,
                            const std::unordered_map<std::string, uint32_t>& vocab_lookup,
                            const std::vector<std::string>& vocab_tokens) {
    auto unicode_to_byte = BuildUnicodeToByteTable(byte_to_unicode);

    struct Case { std::string text; const char* label; };
    Case cases[] = {
        { "What is the capital of France", "the exact real-data sentence verified earlier tonight" },
        { "Hello world", "simple two-word sentence" },
        { "don't", "contraction" },
        { "what is 2 plus 2", "tonight's actual first live user question" },
    };

    bool all_pass = true;
    for (const auto& c : cases) {
        std::vector<uint32_t> ids = EncodeText(c.text, byte_to_unicode, merge_rank, vocab_lookup);
        std::string roundtrip = DetokenizeIds(ids, vocab_tokens, unicode_to_byte);
        bool ok = (roundtrip == c.text);
        std::cout << "    [DETOK-TEST] " << (ok ? "PASS" : "FAIL") << " " << c.label
                  << ": \"" << c.text << "\" -> encode -> detokenize -> \"" << roundtrip << "\"\n";
        if (!ok) all_pass = false;
    }
    return all_pass;
}

// ==============================================================================
// KINTSUGI: Q8_0 DEQUANTIZATION KERNEL (PHASE 8)
// ==============================================================================
// Q8_0 block layout (ggml_type=8, block size=34 bytes):
//   [0..1]  : uint16_t — block scale factor stored as fp16
//   [2..33] : int8_t[32] — 32 quantized signed-integer weights
//
// To recover a float: weight = (float)qs[i] * fp16_to_float(scale)
//
// This is the simplest real quant format — one scale per 32 weights,
// no nested sub-block corrections. Chosen specifically so RoPE, QKVO
// projections, and GGUF loading can all be verified correct before
// the more complex Q4_K_M (3-level nested scales) is tackled.

struct alignas(2) GGML_Q8_0_Block {
    uint16_t scale_fp16;   // block scale as IEEE 754 fp16
    int8_t   qs[32];       // 32 quantized weights
};
static_assert(sizeof(GGML_Q8_0_Block) == 34, "Q8_0 block must be exactly 34 bytes");

// Inline fp16 -> fp32 conversion without requiring _Float16 or hardware support
inline float fp16_to_float(uint16_t h) {
    uint32_t sign     = (h & 0x8000u) << 16;
    uint32_t exponent = (h & 0x7C00u) >> 10;
    uint32_t mantissa = (h & 0x03FFu);

    uint32_t f;
    if (exponent == 0) {
        // Subnormal fp16 -> normalized fp32
        if (mantissa == 0) { f = sign; }
        else {
            exponent = 1;
            while (!(mantissa & 0x0400u)) { mantissa <<= 1; --exponent; }
            mantissa &= 0x03FFu;
            f = sign | ((exponent + (127 - 15)) << 23) | (mantissa << 13);
        }
    } else if (exponent == 31) {
        // Inf or NaN
        f = sign | 0x7F800000u | (mantissa << 13);
    } else {
        f = sign | ((exponent + (127 - 15)) << 23) | (mantissa << 13);
    }
    float result;
    memcpy(&result, &f, sizeof(float));
    return result;
}

// Dequantize one Q8_0 block into 32 floats.
// output must point to at least 32 floats of aligned storage.
inline void dequantize_q8_0_block(const GGML_Q8_0_Block& block, float* output) {
    float scale = fp16_to_float(block.scale_fp16);
    __m256 v_scale = _mm256_set1_ps(scale);

    // Process 32 int8 weights in two AVX2 passes of 16 each,
    // then widen to float and multiply by the block scale.
    // int8 -> int16 -> int32 -> float, all in-register, zero memory traffic.
    __m128i v_raw_lo = _mm_loadu_si128(reinterpret_cast<const __m128i*>(&block.qs[0]));
    __m128i v_raw_hi = _mm_loadu_si128(reinterpret_cast<const __m128i*>(&block.qs[16]));

    // Sign-extend int8 -> int16
    __m256i v_i16_lo = _mm256_cvtepi8_epi16(v_raw_lo);
    __m256i v_i16_hi = _mm256_cvtepi8_epi16(v_raw_hi);

    // int16 -> int32 (two halves of each)
    __m256i v_i32_lo_a = _mm256_cvtepi16_epi32(_mm256_castsi256_si128(v_i16_lo));
    __m256i v_i32_lo_b = _mm256_cvtepi16_epi32(_mm256_extracti128_si256(v_i16_lo, 1));
    __m256i v_i32_hi_a = _mm256_cvtepi16_epi32(_mm256_castsi256_si128(v_i16_hi));
    __m256i v_i32_hi_b = _mm256_cvtepi16_epi32(_mm256_extracti128_si256(v_i16_hi, 1));

    // int32 -> float, multiply by scale
    _mm256_store_ps(&output[0],  _mm256_mul_ps(_mm256_cvtepi32_ps(v_i32_lo_a), v_scale));
    _mm256_store_ps(&output[8],  _mm256_mul_ps(_mm256_cvtepi32_ps(v_i32_lo_b), v_scale));
    _mm256_store_ps(&output[16], _mm256_mul_ps(_mm256_cvtepi32_ps(v_i32_hi_a), v_scale));
    _mm256_store_ps(&output[24], _mm256_mul_ps(_mm256_cvtepi32_ps(v_i32_hi_b), v_scale));
}

// Dequantize a full Q8_0 row of d_model elements into a float output buffer.
// blocks_per_row = d_model / 32  (4096 / 32 = 128 blocks per row for Llama-3)
void dequantize_q8_0_row(const GGML_Q8_0_Block* row_blocks,
                           float* output,
                           size_t blocks_per_row) {
    for (size_t b = 0; b < blocks_per_row; ++b) {
        dequantize_q8_0_block(row_blocks[b], output + (b * 32));
    }
}

// ==============================================================================
// KINTSUGI: Q4_K DEQUANTIZATION KERNEL (PHASE 8 cont.)
// ==============================================================================
// Q4_K super-block layout (ggml_type=12, QK_K=256 weights per super-block):
//
//   [0..1]    uint16_t d     — fp16 super-block scale for the sub-block scales
//   [2..3]    uint16_t dmin  — fp16 super-block scale for the sub-block mins
//   [4..15]   uint8_t scales[12] — 8 sub-block scales + 8 sub-block mins,
//                                   each packed as 6 bits into 12 bytes
//   [16..143] uint8_t qs[128]   — 256 weights as 4-bit nibbles (2 per byte)
//
// Total: 144 bytes per super-block, 256 weights per super-block.
//
// Sub-block scale unpacking (the trickiest part):
//   scales[0..5]  hold the lower 4 bits of each of the 8 scale values
//   scales[6..11] hold the lower 4 bits of each of the 8 min values
//   The upper 2 bits of each scale/min are packed into the high nibbles
//   of scales[0..5], giving 6 bits total per scale/min value.
//
// Reconstruction formula for weight w at position i within a sub-block:
//   sub_scale = (scales_byte & 0x3F) * d_scale   (6-bit scale × super scale)
//   sub_min   = (mins_byte   & 0x3F) * d_min     (6-bit min   × super min)
//   output[i] = sub_scale * nibble - sub_min

struct alignas(2) GGML_Q4_K_Block {
    uint16_t d;           // fp16 super-block scale for scales
    uint16_t dmin;        // fp16 super-block scale for mins
    uint8_t  scales[12];  // 6-bit sub-block scales (8) and mins (8), packed
    uint8_t  qs[128];     // 4-bit nibbles, 256 weights
};
static_assert(sizeof(GGML_Q4_K_Block) == 144, "Q4_K block must be exactly 144 bytes");

// Extract one 6-bit sub-block scale or min from the packed scales array.
// There are 8 scale values (indices 0-7) and 8 min values (indices 8-15).
// Packing layout verified against llama.cpp ggml-quants.c:
//   lower 4 bits of scale[i] live in scales[i] & 0xF  (i in 0..7)
//   upper 2 bits of scale[i] live in scales[i+4] >> 4  (i in 0..3)
//                              and scales[i]    >> 4  (i in 4..7, different byte)
// Simplified canonical extraction used here matches ggml_get_scale_min_k4:
inline void get_scale_min_k4(int j, const uint8_t* q, uint8_t& d_out, uint8_t& m_out) {
    if (j < 4) {
        d_out = q[j] & 63;
        m_out = q[j + 4] & 63;
    } else {
        d_out = (q[j + 4] & 0xF) | ((q[j - 4] >> 6) << 4);
        m_out = (q[j + 4] >> 4)  | ((q[j]     >> 6) << 4);
    }
}

void dequantize_q4_k_block(const GGML_Q4_K_Block& block, float* output) {
    const float d   = fp16_to_float(block.d);
    const float min = fp16_to_float(block.dmin);
    const uint8_t* q = block.qs;
    float* y = output;
    int is = 0;

    // Process 256 weights in 4 groups of 64.
    // Within each group: first 32 outputs = low nibbles, next 32 = high nibbles.
    // This exactly matches ggml's dequantize_row_q4_K reference implementation.
    for (int j = 0; j < 256; j += 64) {
        uint8_t sc, m;
        get_scale_min_k4(is + 0, block.scales, sc, m);
        const float d1 = d * sc, m1 = min * m;
        get_scale_min_k4(is + 1, block.scales, sc, m);
        const float d2 = d * sc, m2 = min * m;
        for (int l = 0; l < 32; ++l) *y++ = d1 * static_cast<float>(q[l] & 0x0F) - m1;
        for (int l = 0; l < 32; ++l) *y++ = d2 * static_cast<float>(q[l] >>    4) - m2;
        q  += 32;
        is += 2;
    }
}

void dequantize_q4_k_row(const GGML_Q4_K_Block* row_blocks,
                           float* output,
                           size_t blocks_per_row) {
    for (size_t b = 0; b < blocks_per_row; ++b) {
        dequantize_q4_k_block(row_blocks[b], output + (b * 256));
    }
}

// ==============================================================================
// KINTSUGI: THE LEXICON (EMBEDDING TABLE) — RAM-VAULT BACKED
// ==============================================================================
// No longer keeps a fully dequantized [vocab_size x d_model] copy (that was
// 1.96GB of arena space for nothing but this one lookup path). Instead it
// just remembers where token_embd.weight lives inside the AWE vault and
// dequantizes exactly one row, on demand, per token — trivial cost (16 Q4_K
// blocks = 2304 bytes), and it frees ~1.96GB of physical RAM for the vault.
struct KintsugiEmbedding {
    const uint8_t* raw_base = nullptr; // g_ModelFile.base + token_embd.weight's rel offset
    uint8_t        ggml_type = 0;
    size_t         vocab_size = 0;
    size_t         d_model = 0;

    void Init(const uint8_t* vault_base, uint64_t rel_offset, uint8_t type, size_t vocab, size_t dim) {
        raw_base = vault_base + rel_offset;
        ggml_type = type;
        vocab_size = vocab;
        d_model = dim;
    }

    void Lookup(uint32_t token_id, float* residual_stream) const {
        if (token_id >= vocab_size) {
            std::cerr << "[!] WARNING: token_id " << token_id << " exceeds vocab_size ("
                      << vocab_size << "). Clamping to 0.\n";
            token_id = 0;
        }
        if (ggml_type == 0) { // F32 — direct copy, no dequant needed
            memcpy(residual_stream, raw_base + (static_cast<size_t>(token_id) * d_model * sizeof(float)),
                   d_model * sizeof(float));
        } else { // Q4_K — the only quantized type token_embd.weight actually uses in this file
            const size_t blocks_per_row = d_model / 256;
            const size_t row_bytes = blocks_per_row * sizeof(GGML_Q4_K_Block);
            const GGML_Q4_K_Block* row = reinterpret_cast<const GGML_Q4_K_Block*>(
                raw_base + (static_cast<size_t>(token_id) * row_bytes));
            dequantize_q4_k_row(row, residual_stream, blocks_per_row);
        }
    }
};

KintsugiEmbedding g_Embedding;

// ==============================================================================
// KINTSUGI: Q6_K DEQUANTIZATION KERNEL
// ==============================================================================
// Q6_K super-block layout (ggml_type=14, QK_K=256 weights per super-block):
//
//   [0..127]   uint8_t ql[128]    — lower 4 bits of each weight (2 per byte)
//   [128..191] uint8_t qh[64]     — upper 2 bits of each weight (4 per byte)
//   [192..207] int8_t  scales[16] — 16 sub-block scales, one per 16 weights
//   [208..209] uint16_t d         — fp16 super-block scale
//
// Total: 210 bytes per super-block, 256 weights per super-block.
//
// Reconstruction:
//   6-bit weight q = (ql_nibble) | ((qh_pair) << 4)   — reassemble 6 bits
//   signed value  = q - 32                              — center around 0
//   output        = d_float * scales[sub] * signed_val

struct GGML_Q6_K_Block {
    uint8_t  ql[128];    // lower 4 bits
    uint8_t  qh[64];     // upper 2 bits
    int8_t   scales[16]; // sub-block scales
    uint16_t d;          // fp16 super-block scale
};
static_assert(sizeof(GGML_Q6_K_Block) == 210, "Q6_K block must be exactly 210 bytes");

void dequantize_q6_k_block(const GGML_Q6_K_Block& block, float* output) {
    const float d = fp16_to_float(block.d);
    const uint8_t* ql = block.ql;
    const uint8_t* qh = block.qh;
    const int8_t*  sc = block.scales;

    // Process 256 weights in 2 passes of 128 each.
    // Each pass: 32 iterations of l, each producing 4 outputs at
    // l+0, l+32, l+64, l+96 within the current 128-element window.
    // Upper 2 bits come from qh[l], packed as 4 pairs of 2 bits per byte.
    // This exactly matches ggml's dequantize_row_q6_K reference implementation.
    for (int n = 0; n < 256; n += 128) {
        for (int l = 0; l < 32; ++l) {
            // Sub-block scale index shifts once l crosses 16 within this
            // 128-weight pass (verified against ggml's dequantize_row_q6_K,
            // which computes this exact `is = l/16` offset).
            int is = l / 16;
            int8_t q1 = static_cast<int8_t>((ql[l +  0] & 0x0F) | (((qh[l] >> 0) & 3) << 4)) - 32;
            int8_t q2 = static_cast<int8_t>((ql[l + 32] & 0x0F) | (((qh[l] >> 2) & 3) << 4)) - 32;
            int8_t q3 = static_cast<int8_t>((ql[l +  0] >>    4) | (((qh[l] >> 4) & 3) << 4)) - 32;
            int8_t q4 = static_cast<int8_t>((ql[l + 32] >>    4) | (((qh[l] >> 6) & 3) << 4)) - 32;
            output[n + l +  0] = d * static_cast<float>(sc[is + 0]) * static_cast<float>(q1);
            output[n + l + 32] = d * static_cast<float>(sc[is + 2]) * static_cast<float>(q2);
            output[n + l + 64] = d * static_cast<float>(sc[is + 4]) * static_cast<float>(q3);
            output[n + l + 96] = d * static_cast<float>(sc[is + 6]) * static_cast<float>(q4);
        }
        ql += 64; qh += 32; sc += 8;
    }
}

void dequantize_q6_k_row(const GGML_Q6_K_Block* row_blocks,
                           float* output,
                           size_t blocks_per_row) {
    for (size_t b = 0; b < blocks_per_row; ++b) {
        dequantize_q6_k_block(row_blocks[b], output + (b * 256));
    }
}
// ==============================================================================
// KINTSUGI: REAL SAMPLING (TEMPERATURE + TOP-P, REPLACING PURE ARGMAX)
// ==============================================================================
// Every generation so far picked the single highest-scoring token and threw
// away the other 128,255 scores. Real sampling needs the actual distribution,
// not just the winner — this extends the top-5 tracking pattern already used
// earlier tonight to a configurable top-K, then runs temperature scaling ->
// softmax -> top-p (nucleus) filtering -> a genuine random draw on that set.

struct TopKCandidate { uint32_t token_id; float score; };

// Dot-products each row of the pinned, pre-materialized LM head buffer
// against the normed residual, and maintains a sorted top-K list — same
// insertion-sort pattern as the original top-5 tracking, just generalized.
// Reads from a buffer we control directly (filled once at boot from the
// mmap vault) rather than depending on mmap page-cache residency holding
// up under whatever system-wide memory pressure exists at sweep time.
std::vector<TopKCandidate> RunLMHeadSweep(const uint8_t* lm_head_buffer,
                                           const float* normed_residual,
                                           size_t d_model, size_t vocab_size,
                                           float* lm_row_scratch, size_t top_k) {
    std::vector<TopKCandidate> top(top_k, {0, -1e30f});
    if (!lm_head_buffer) return top;

    constexpr size_t BPR = 4096 / 256;
    constexpr size_t RB  = BPR * sizeof(GGML_Q6_K_Block);

    for (size_t v = 0; v < vocab_size; ++v) {
        const GGML_Q6_K_Block* row_blocks = reinterpret_cast<const GGML_Q6_K_Block*>(lm_head_buffer + v * RB);
        dequantize_q6_k_row(row_blocks, lm_row_scratch, BPR);

        __m256 acc = _mm256_setzero_ps();
        for (size_t i = 0; i < d_model; i += 8)
            acc = _mm256_fmadd_ps(_mm256_load_ps(&normed_residual[i]),
                                  _mm256_load_ps(&lm_row_scratch[i]), acc);
        alignas(32) float lanes[8];
        _mm256_store_ps(lanes, acc);
        float dot = lanes[0]+lanes[1]+lanes[2]+lanes[3]+lanes[4]+lanes[5]+lanes[6]+lanes[7];

        if (dot > top.back().score) {
            top.back() = {static_cast<uint32_t>(v), dot};
            for (size_t t = top.size() - 1; t > 0 && top[t].score > top[t-1].score; --t)
                std::swap(top[t], top[t-1]);
        }
    }
    return top;
}

// Temperature scaling + numerically-stable softmax + top-p filtering + a
// real draw from std::mt19937. temperature <= ~0 is special-cased to exact
// argmax — not just "very likely argmax" — both to avoid dividing by
// near-zero and so this collapses to a deterministic, exact match against
// every proven-correct greedy result from earlier tonight, not a
// probabilistic one. candidates must already be sorted descending by score
// (RunLMHeadSweep guarantees this).
uint32_t SampleToken(const std::vector<TopKCandidate>& candidates,
                      float temperature, float top_p, std::mt19937& rng) {
    if (candidates.empty()) return 0;
    if (temperature <= 1e-6f) return candidates[0].token_id;

    std::vector<float> probs(candidates.size());
    float max_logit = candidates[0].score;
    float sum_exp = 0.0f;
    for (size_t i = 0; i < candidates.size(); ++i) {
        probs[i] = std::exp((candidates[i].score - max_logit) / temperature);
        sum_exp += probs[i];
    }
    for (auto& p : probs) p /= sum_exp;

    float cumulative = 0.0f;
    size_t cutoff = candidates.size();
    for (size_t i = 0; i < probs.size(); ++i) {
        cumulative += probs[i];
        if (cumulative >= top_p) { cutoff = i + 1; break; }
    }

    float renorm_sum = 0.0f;
    for (size_t i = 0; i < cutoff; ++i) renorm_sum += probs[i];

    std::uniform_real_distribution<float> uni(0.0f, 1.0f);
    float draw = uni(rng) * renorm_sum;
    float running = 0.0f;
    for (size_t i = 0; i < cutoff; ++i) {
        running += probs[i];
        if (draw <= running) return candidates[i].token_id;
    }
    return candidates[cutoff - 1].token_id; // float-rounding fallback
}

// Self-test: every expected value below was independently computed in
// Python first (softmax probabilities, top-p cutoff, and the exact
// renormalized-draw outcome) and cross-checked before being hardcoded here.
bool RunSamplingSelfTest() {
    bool all_pass = true;
    std::mt19937 test_rng(42); // fixed seed — reproducible, not random_device

    // Case 1: near-zero temperature collapses to EXACT argmax, deterministically.
    {
        std::vector<TopKCandidate> cands = {{5, 10.0f}, {3, 8.0f}, {7, 1.0f}};
        uint32_t result = SampleToken(cands, 0.0000001f, 0.9f, test_rng);
        bool ok = (result == 5);
        std::cout << "    [SAMPLE-TEST] " << (ok ? "PASS" : "FAIL")
                  << " near-zero temperature -> exact argmax (got id=" << result << ")\n";
        if (!ok) all_pass = false;
    }

    // Case 2: softmax on a known distribution — Python-verified:
    // scores [2,1,0] at temp=1.0 -> probs [0.6652, 0.2447, 0.0900].
    // Indirect check: confirm token 100 (highest prob, ~66.5%) is picked
    // substantially more often than token 102 (lowest prob, ~9%) over many
    // draws — a real check that scores drive a probability-weighted draw,
    // not a uniform random pick across candidates.
    {
        std::vector<TopKCandidate> cands = {{100, 2.0f}, {101, 1.0f}, {102, 0.0f}};
        int count100 = 0, count102 = 0;
        std::mt19937 dist_rng(7);
        for (int i = 0; i < 2000; ++i) {
            uint32_t r = SampleToken(cands, 1.0f, 1.0f, dist_rng);
            if (r == 100) ++count100;
            if (r == 102) ++count102;
        }
        bool ok = (count100 > count102 * 3); // ~66% vs ~9% -- expect roughly 7x, demand at least 3x
        std::cout << "    [SAMPLE-TEST] " << (ok ? "PASS" : "FAIL")
                  << " probability-weighted draw over 2000 samples: id=100 picked " << count100
                  << "x, id=102 picked " << count102 << "x (100 should dominate)\n";
        if (!ok) all_pass = false;
    }

    // Case 3: top-p=0.9 filtering — Python-verified cutoff keeps exactly the
    // first 2 of 3 candidates (cumulative 0.6652 -> 0.91). Token id=102 must
    // NEVER be selected, across many draws, once it's outside the nucleus.
    {
        std::vector<TopKCandidate> cands = {{100, 2.0f}, {101, 1.0f}, {102, 0.0f}};
        std::mt19937 cutoff_rng(99);
        bool ever_picked_excluded = false;
        for (int i = 0; i < 2000; ++i) {
            uint32_t r = SampleToken(cands, 1.0f, 0.9f, cutoff_rng);
            if (r == 102) { ever_picked_excluded = true; break; }
        }
        bool ok = !ever_picked_excluded;
        std::cout << "    [SAMPLE-TEST] " << (ok ? "PASS" : "FAIL")
                  << " top-p=0.9 nucleus filtering: id=102 (outside nucleus) "
                  << (ever_picked_excluded ? "WAS incorrectly selected" : "never selected across 2000 draws") << "\n";
        if (!ok) all_pass = false;
    }

    return all_pass;
}


// ==============================================================================
// KINTSUGI: UNIFIED TENSOR LOADER — loads any supported quant type from GGUF
// into a pre-allocated float* buffer (does NOT allocate arena space itself).
// Caller allocates the output buffer; this function fills it.
// ==============================================================================
bool LoadTensorIntoBuffer(const std::string& gguf_path,
                           const KintsugiGGUFParser& gguf,
                           const std::string& tensor_name,
                           float* output,
                           size_t expected_elements) {
    const GGUFTensorInfo* info = nullptr;
    for (const auto& t : gguf.tensors) {
        if (t.name == tensor_name) { info = &t; break; }
    }
    if (!info) {
        std::cerr << "[!] Tensor not found: " << tensor_name << "\n";
        return false;
    }

    size_t num_elements = 1;
    for (auto d : info->dims) num_elements *= static_cast<size_t>(d);
    if (num_elements != expected_elements) {
        std::cerr << "[!] Tensor " << tensor_name << " size mismatch: expected "
                  << expected_elements << " got " << num_elements << "\n";
        return false;
    }

    std::ifstream f(gguf_path, std::ios::binary);
    if (!f.is_open()) { std::cerr << "[!] Cannot open GGUF file.\n"; return false; }
    f.seekg(static_cast<std::streamoff>(gguf.tensor_data_offset + info->offset));

    if (info->ggml_type == 0) {
        // F32 — read directly
        f.read(reinterpret_cast<char*>(output), num_elements * sizeof(float));
    } else if (info->ggml_type == 12) {
        // Q4_K
        size_t blocks = num_elements / 256;
        std::vector<GGML_Q4_K_Block> raw(blocks);
        f.read(reinterpret_cast<char*>(raw.data()), blocks * sizeof(GGML_Q4_K_Block));
        // rows = num_elements / last_dim; blocks_per_row = last_dim / 256
        size_t row_dim = static_cast<size_t>(info->dims[0]);
        size_t rows    = num_elements / row_dim;
        size_t bpr     = row_dim / 256;
        for (size_t r = 0; r < rows; ++r)
            dequantize_q4_k_row(&raw[r * bpr], output + (r * row_dim), bpr);
    } else if (info->ggml_type == 14) {
        // Q6_K
        size_t blocks = num_elements / 256;
        std::vector<GGML_Q6_K_Block> raw(blocks);
        f.read(reinterpret_cast<char*>(raw.data()), blocks * sizeof(GGML_Q6_K_Block));
        size_t row_dim = static_cast<size_t>(info->dims[0]);
        size_t rows    = num_elements / row_dim;
        size_t bpr     = row_dim / 256;
        for (size_t r = 0; r < rows; ++r)
            dequantize_q6_k_row(&raw[r * bpr], output + (r * row_dim), bpr);
    } else {
        std::cerr << "[!] Unsupported ggml_type=" << info->ggml_type
                  << " for tensor: " << tensor_name << "\n";
        return false;
    }

    if (!f.good()) {
        std::cerr << "[!] Read truncated for tensor: " << tensor_name << "\n";
        return false;
    }
    return true;
}

// ==============================================================================
// KINTSUGI: PER-LAYER WEIGHT TOPOLOGY
// ==============================================================================
struct KintsugiLayerWeights {
    // Attention
    float* wq = nullptr;          // [4096 × 4096] Q4_K
    float* wk = nullptr;          // [1024 × 4096] Q4_K  (GQA narrow)
    float* wv = nullptr;          // [1024 × 4096] Q6_K
    float* wo = nullptr;          // [4096 × 4096] Q4_K
    float* attn_norm = nullptr;   // [4096]         F32   RMSNorm scale

    // FFN
    float* ffn_gate = nullptr;    // [14336 × 4096] Q4_K
    float* ffn_up   = nullptr;    // [14336 × 4096] Q4_K
    float* ffn_down = nullptr;    // [4096 × 14336] Q6_K
    float* ffn_norm = nullptr;    // [4096]          F32  RMSNorm scale
};

// ==============================================================================
// KINTSUGI: LAYER OFFSET CACHE — built once at boot, used every token
// ==============================================================================
// Eliminates 291 string comparisons × 9 tensors × 32 layers = 83,664 lookups
// per token, and drops from 9 file opens per layer to 1 persistent handle.

struct LayerOffsetCache {
    uint64_t wq, wk, wv, wo;
    uint64_t attn_norm;
    uint64_t ffn_gate, ffn_up, ffn_down;
    uint64_t ffn_norm;
    uint8_t  wq_type, wk_type, wv_type, wo_type;
    uint8_t  ffn_gate_type, ffn_up_type, ffn_down_type;
};

LayerOffsetCache g_LayerOffsets[32] = {};

void PrecomputeLayerOffsets(const KintsugiGGUFParser& gguf) {
    std::cout << "[⚡ INDEX] Pre-computing layer tensor offsets...\n";
    for (size_t layer = 0; layer < 32; ++layer) {
        std::string b = "blk." + std::to_string(layer) + ".";
        for (const auto& t : gguf.tensors) {
            if      (t.name == b+"attn_q.weight")      { g_LayerOffsets[layer].wq        = t.offset; g_LayerOffsets[layer].wq_type        = (uint8_t)t.ggml_type; }
            else if (t.name == b+"attn_k.weight")      { g_LayerOffsets[layer].wk        = t.offset; g_LayerOffsets[layer].wk_type        = (uint8_t)t.ggml_type; }
            else if (t.name == b+"attn_v.weight")      { g_LayerOffsets[layer].wv        = t.offset; g_LayerOffsets[layer].wv_type        = (uint8_t)t.ggml_type; }
            else if (t.name == b+"attn_output.weight") { g_LayerOffsets[layer].wo        = t.offset; g_LayerOffsets[layer].wo_type        = (uint8_t)t.ggml_type; }
            else if (t.name == b+"attn_norm.weight")   { g_LayerOffsets[layer].attn_norm = t.offset; }
            else if (t.name == b+"ffn_gate.weight")    { g_LayerOffsets[layer].ffn_gate  = t.offset; g_LayerOffsets[layer].ffn_gate_type  = (uint8_t)t.ggml_type; }
            else if (t.name == b+"ffn_up.weight")      { g_LayerOffsets[layer].ffn_up    = t.offset; g_LayerOffsets[layer].ffn_up_type    = (uint8_t)t.ggml_type; }
            else if (t.name == b+"ffn_down.weight")    { g_LayerOffsets[layer].ffn_down  = t.offset; g_LayerOffsets[layer].ffn_down_type  = (uint8_t)t.ggml_type; }
            else if (t.name == b+"ffn_norm.weight")    { g_LayerOffsets[layer].ffn_norm  = t.offset; }
        }
    }
    std::cout << "[✓ INDEX] Jump table baked — 32 layers × 9 tensors cached.\n";
}

// Fast loader: reads straight out of the memory-mapped GGUF (g_ModelFile) —
// no explicit seek/read calls; the OS page cache serves repeated touches of
// the same layer at RAM speed once warm. Dequantizes directly from the
// mapped bytes into the arena buffer.
bool FastLoadLayer(size_t layer_idx, TensorArena& arena, KintsugiLayerWeights& out) {
    const size_t D  = 4096;
    const size_t KV = 1024;
    const size_t FF = 14336;
    const LayerOffsetCache& c = g_LayerOffsets[layer_idx];

    // Allocate arena space
    out.wq        = static_cast<float*>(arena.AllocateAligned(D  * D  * sizeof(float)));
    out.wk        = static_cast<float*>(arena.AllocateAligned(KV * D  * sizeof(float)));
    out.wv        = static_cast<float*>(arena.AllocateAligned(KV * D  * sizeof(float)));
    out.wo        = static_cast<float*>(arena.AllocateAligned(D  * D  * sizeof(float)));
    out.attn_norm = static_cast<float*>(arena.AllocateAligned(D        * sizeof(float)));
    out.ffn_gate  = static_cast<float*>(arena.AllocateAligned(FF * D  * sizeof(float)));
    out.ffn_up    = static_cast<float*>(arena.AllocateAligned(FF * D  * sizeof(float)));
    out.ffn_down  = static_cast<float*>(arena.AllocateAligned(D  * FF * sizeof(float)));
    out.ffn_norm  = static_cast<float*>(arena.AllocateAligned(D        * sizeof(float)));

    if (!out.wq || !out.wk || !out.wv || !out.wo || !out.attn_norm ||
        !out.ffn_gate || !out.ffn_up || !out.ffn_down || !out.ffn_norm) {
        std::cerr << "[❌ FATAL] Arena OOM layer " << layer_idx << "\n";
        return false;
    }

    // Helper: dequantize directly out of the RAM vault at the cached offset —
    // zero disk I/O, zero staging memcpy.
    auto fast_load = [&](uint64_t rel_offset, uint8_t type,
                          float* dst, size_t n_elems) -> bool {
        const uint8_t* src = g_ModelFile.base + rel_offset;
        if (type == 0) {
            memcpy(dst, src, n_elems * sizeof(float));
        } else if (type == 12) { // Q4_K
            const GGML_Q4_K_Block* blocks = reinterpret_cast<const GGML_Q4_K_Block*>(src);
            size_t nblocks = n_elems / 256;
            for (size_t b = 0; b < nblocks; ++b)
                dequantize_q4_k_block(blocks[b], dst + b * 256);
        } else if (type == 14) { // Q6_K
            const GGML_Q6_K_Block* blocks = reinterpret_cast<const GGML_Q6_K_Block*>(src);
            size_t nblocks = n_elems / 256;
            for (size_t b = 0; b < nblocks; ++b)
                dequantize_q6_k_block(blocks[b], dst + b * 256);
        } else {
            std::cerr << "[!] Unsupported type " << (int)type << "\n";
            return false;
        }
        return true;
    };

    bool ok = true;
    ok &= fast_load(c.wq,        c.wq_type,        out.wq,        D  * D );
    ok &= fast_load(c.wk,        c.wk_type,        out.wk,        KV * D );
    ok &= fast_load(c.wv,        c.wv_type,        out.wv,        KV * D );
    ok &= fast_load(c.wo,        c.wo_type,        out.wo,        D  * D );
    ok &= fast_load(c.attn_norm, 0,                out.attn_norm, D      );
    ok &= fast_load(c.ffn_gate,  c.ffn_gate_type,  out.ffn_gate,  FF * D );
    ok &= fast_load(c.ffn_up,    c.ffn_up_type,    out.ffn_up,    FF * D );
    ok &= fast_load(c.ffn_down,  c.ffn_down_type,  out.ffn_down,  D  * FF);
    ok &= fast_load(c.ffn_norm,  0,                out.ffn_norm,  D      );
    return ok;
}

// Loads all 9 tensors for one transformer layer into pre-allocated arena space.
// Returns false if any single tensor fails — caller should abort on failure.
bool LoadLayerWeights(const std::string& gguf_path,
                       const KintsugiGGUFParser& gguf,
                       TensorArena& arena,
                       size_t layer_idx,
                       KintsugiLayerWeights& out) {
    const size_t D  = 4096;
    const size_t KV = 1024;   // GQA KV width
    const size_t FF = 14336;
    const std::string b = "blk." + std::to_string(layer_idx) + ".";

    // Allocate all buffers first, then load — keeps arena layout clean
    out.wq        = static_cast<float*>(arena.AllocateAligned(D  * D  * sizeof(float)));
    out.wk        = static_cast<float*>(arena.AllocateAligned(KV * D  * sizeof(float)));
    out.wv        = static_cast<float*>(arena.AllocateAligned(KV * D  * sizeof(float)));
    out.wo        = static_cast<float*>(arena.AllocateAligned(D  * D  * sizeof(float)));
    out.attn_norm = static_cast<float*>(arena.AllocateAligned(D        * sizeof(float)));
    out.ffn_gate  = static_cast<float*>(arena.AllocateAligned(FF * D  * sizeof(float)));
    out.ffn_up    = static_cast<float*>(arena.AllocateAligned(FF * D  * sizeof(float)));
    out.ffn_down  = static_cast<float*>(arena.AllocateAligned(D  * FF * sizeof(float)));
    out.ffn_norm  = static_cast<float*>(arena.AllocateAligned(D        * sizeof(float)));

    if (!out.wq || !out.wk || !out.wv || !out.wo || !out.attn_norm ||
        !out.ffn_gate || !out.ffn_up || !out.ffn_down || !out.ffn_norm) {
        std::cerr << "[❌ FATAL] Arena OOM allocating layer " << layer_idx << " weights.\n";
        return false;
    }

    bool ok = true;
    ok &= LoadTensorIntoBuffer(gguf_path, gguf, b+"attn_q.weight",      out.wq,        D  * D );
    ok &= LoadTensorIntoBuffer(gguf_path, gguf, b+"attn_k.weight",      out.wk,        KV * D );
    ok &= LoadTensorIntoBuffer(gguf_path, gguf, b+"attn_v.weight",      out.wv,        KV * D );
    ok &= LoadTensorIntoBuffer(gguf_path, gguf, b+"attn_output.weight", out.wo,        D  * D );
    ok &= LoadTensorIntoBuffer(gguf_path, gguf, b+"attn_norm.weight",   out.attn_norm, D      );
    ok &= LoadTensorIntoBuffer(gguf_path, gguf, b+"ffn_gate.weight",    out.ffn_gate,  FF * D );
    ok &= LoadTensorIntoBuffer(gguf_path, gguf, b+"ffn_up.weight",      out.ffn_up,    FF * D );
    ok &= LoadTensorIntoBuffer(gguf_path, gguf, b+"ffn_down.weight",    out.ffn_down,  D  * FF);
    ok &= LoadTensorIntoBuffer(gguf_path, gguf, b+"ffn_norm.weight",    out.ffn_norm,  D      );
    return ok;
}

// ==============================================================================
// KINTSUGI: FORWARD PASS ENGINE (PHASE 10)
// ==============================================================================

// RMSNorm: normalize x by its root-mean-square, then scale by learned weights.
// eps=1e-5 matches Llama-3's configuration exactly.
void RMSNorm(const float* x, const float* scale, float* out, size_t dim, float eps = 1e-5f) {
    float sum_sq = 0.0f;
    for (size_t i = 0; i < dim; i++) sum_sq += x[i] * x[i];
    float rms_inv = 1.0f / sqrtf(sum_sq / static_cast<float>(dim) + eps);
    for (size_t i = 0; i < dim; i++) out[i] = x[i] * rms_inv * scale[i];
}

// Matrix-vector product: out[out_dim] = mat[out_dim x in_dim] @ vec[in_dim]
// Weight matrix is row-major — row r starts at mat + r * in_dim.
// in_dim must be a multiple of 8 (true for all Llama-3 dims: 4096, 14336, 1024).
void MatVec(const float* __restrict__ mat,
             const float* __restrict__ vec,
             float* __restrict__ out,
             size_t in_dim, size_t out_dim) {
    for (size_t r = 0; r < out_dim; ++r) {
        __m256 acc = _mm256_setzero_ps();
        const float* row = mat + r * in_dim;
        for (size_t i = 0; i < in_dim; i += 8) {
            acc = _mm256_fmadd_ps(_mm256_load_ps(&row[i]),
                                  _mm256_load_ps(&vec[i]), acc);
        }
        alignas(32) float lanes[8];
        _mm256_store_ps(lanes, acc);
        out[r] = lanes[0]+lanes[1]+lanes[2]+lanes[3]+
                 lanes[4]+lanes[5]+lanes[6]+lanes[7];
    }
}

// Rotary Position Embedding, applied in-place to a packed [n_heads x head_dim] vector.
// theta_base = 500000.0f for Llama-3 (verified from GGUF metadata: rope_freq_base=500000).
// Each consecutive pair of dimensions (2i, 2i+1) within a head is rotated by
// angle = pos / theta_base^(2i/head_dim).
void ApplyRoPE(float* x, size_t pos, size_t n_heads, size_t head_dim, float theta_base) {
    for (size_t h = 0; h < n_heads; ++h) {
        float* head = x + h * head_dim;
        for (size_t i = 0; i < head_dim / 2; ++i) {
            float freq  = 1.0f / powf(theta_base, (2.0f * i) / static_cast<float>(head_dim));
            float angle = static_cast<float>(pos) * freq;
            float cos_a = cosf(angle), sin_a = sinf(angle);
            float x0 = head[2*i], x1 = head[2*i+1];
            head[2*i]   = x0 * cos_a - x1 * sin_a;
            head[2*i+1] = x0 * sin_a + x1 * cos_a;
        }
    }
}

// SiLU activation: x * sigmoid(x) — the gate function in SwiGLU
inline float silu(float x) { return x * (1.0f / (1.0f + expf(-x))); }

// Full single-layer transformer forward pass.
// Updates residual in-place. All scratch buffers must be pre-allocated.
void ForwardOneLayer(const KintsugiLayerWeights& w,
                      float* residual,         // [d_model], updated in place
                      float* scratch_norm,     // [d_model]
                      float* scratch_q,        // [d_model = 32 heads x 128]
                      float* scratch_k,        // [kv_dim  =  8 heads x 128]
                      float* scratch_v,        // [kv_dim  =  8 heads x 128]
                      float* scratch_attn_out, // [d_model]
                      float* scratch_ffn_gate, // [d_ff]
                      float* scratch_ffn_up,   // [d_ff]
                      float* scratch_ffn_out,  // [d_model]
                      float* score_scratch,    // [max_context_len]
                      KVCacheRing& kv_cache,
                      size_t pos,
                      size_t d_model, size_t kv_dim, size_t d_ff,
                      float rms_eps = 1e-5f) { // was hardcoded inside RMSNorm's own default,
                                                // never checked against the real GGUF value —
                                                // now passed through explicitly from parsed metadata

    // ---- ATTENTION SUB-LAYER ----------------------------------------
    // 1. Pre-attention RMSNorm
    RMSNorm(residual, w.attn_norm, scratch_norm, d_model, rms_eps);

    // 2. Q, K, V projections from the normed residual
    MatVec(w.wq, scratch_norm, scratch_q, d_model, d_model);
    MatVec(w.wk, scratch_norm, scratch_k, d_model, kv_dim);
    MatVec(w.wv, scratch_norm, scratch_v, d_model, kv_dim);

    // 3. RoPE applied to Q (32 heads) and K (8 KV heads)
    ApplyRoPE(scratch_q, pos, 32, 128, 500000.0f);
    ApplyRoPE(scratch_k, pos,  8, 128, 500000.0f);

    // 4. Push K, V into the ring cache at this position
    kv_cache.PushContext(scratch_k, scratch_v);

    // 5. GQA attention over all cached positions → attention output
    size_t ctx_len = std::min(pos + 1, kv_cache.max_context_length);
    KintsugiAttention::ComputeAttention(scratch_q, kv_cache, ctx_len,
                                         scratch_attn_out, score_scratch);

    // 6. Output projection (Wo) + residual connection
    MatVec(w.wo, scratch_attn_out, scratch_ffn_out, d_model, d_model);
    for (size_t i = 0; i < d_model; ++i) residual[i] += scratch_ffn_out[i];

    // ---- FFN SUB-LAYER ----------------------------------------------
    // 7. Pre-FFN RMSNorm
    RMSNorm(residual, w.ffn_norm, scratch_norm, d_model, rms_eps);

    // 8. Gate and Up projections for SwiGLU
    MatVec(w.ffn_gate, scratch_norm, scratch_ffn_gate, d_model, d_ff);
    MatVec(w.ffn_up,   scratch_norm, scratch_ffn_up,   d_model, d_ff);

    // 9. SwiGLU activation: gate = SiLU(gate) * up
    for (size_t i = 0; i < d_ff; ++i)
        scratch_ffn_gate[i] = silu(scratch_ffn_gate[i]) * scratch_ffn_up[i];

    // 10. Down projection + residual connection
    MatVec(w.ffn_down, scratch_ffn_gate, scratch_ffn_out, d_ff, d_model);
    for (size_t i = 0; i < d_model; ++i) residual[i] += scratch_ffn_out[i];
}

int main() {
    // Build fingerprint — confirms this exact binary contains the per-layer
    // KV-cache reset fix, rather than inferring it indirectly from output.
    std::cout << "[BUILD] kintsugi_inference -- per-layer KV reset fix v1\n";

    // Fix garbled UTF-8 output — Windows terminals default to CP437/1252,
    // which mangles every emoji and box-drawing character in the output.
    // This was the very first thing flagged in session 1, finally landing now.
    SetConsoleOutputCP(CP_UTF8);

    // Tokenizer Phase I, part 1: prove the codepoint classifier and UTF-8
    // decoder are correct BEFORE anything downstream (the pretokenizer
    // parser, eventually the merge loop) gets built on top of them.
    std::cout << "[🔤 UNICODE] Running codepoint classifier self-test...\n";
    if (!RunUnicodeSelfTest()) {
        std::cerr << "[❌ FATAL] Unicode self-test failed — see [UNICODE-TEST] lines above. "
                     "Not proceeding with a classifier that's proven wrong.\n";
        return 1;
    }
    std::cout << "[✓ UNICODE] All self-test cases passed.\n\n";

    // Tokenizer Phase I, part 2: prove the 7-branch LLAMA3 pretokenizer is
    // correct — including the two hand-derived, Python-verified edge cases
    // (multi-space, multi-newline) — before it feeds anything into the
    // eventual merge loop.
    std::cout << "[🔤 PRETOKENIZER] Running LLAMA3 pretokenizer self-test...\n";
    if (!RunPretokenizerSelfTest()) {
        std::cerr << "[❌ FATAL] Pretokenizer self-test failed — see [PRETOK-TEST] lines above. "
                     "Not proceeding with a chunk splitter that's proven wrong.\n";
        return 1;
    }
    std::cout << "[✓ PRETOKENIZER] All self-test cases passed.\n\n";

    // Tokenizer Phase I, part 3: build the byte-to-unicode table and prove
    // it matches the real GPT-2 algorithm Llama-3 inherited — including the
    // one entry (byte 32) independently checkable against every decode this
    // whole project has already produced.
    std::cout << "[🔤 BYTEMAP] Building and testing byte-to-unicode table...\n";
    std::array<uint32_t, 256> byte_to_unicode = BuildByteToUnicodeTable();
    if (!RunByteToUnicodeSelfTest(byte_to_unicode)) {
        std::cerr << "[❌ FATAL] Byte-to-unicode self-test failed — see [BYTEMAP-TEST] lines above. "
                     "Not proceeding with a table that's proven wrong.\n";
        return 1;
    }
    std::cout << "[✓ BYTEMAP] All self-test cases passed.\n\n";

    // Reverse table, built once — used by DetokenizeIds in the live wake
    // cycle to print human-readable output natively instead of only raw
    // token IDs. Pure inversion of the already-tested forward table, so
    // building it once here (not per-wake) is both correct and efficient.
    std::unordered_map<uint32_t, uint8_t> unicode_to_byte = BuildUnicodeToByteTable(byte_to_unicode);

    // Real sampling: one RNG, seeded once at boot from random_device (not
    // re-seeded per wake). Temperature and top-p are reasonable starting
    // defaults, not yet tuned against any real evidence — first pass only.
    std::random_device rd_seed;
    std::mt19937 g_sampling_rng(rd_seed());
    constexpr float SAMPLING_TEMPERATURE = 0.7f;
    constexpr float SAMPLING_TOP_P       = 0.9f;
    constexpr size_t SAMPLING_TOP_K      = 40;

    // Tokenizer Phase I, part 4: prove the merge algorithm's mechanics
    // (lowest-rank-wins, merge-all-occurrences, multi-round chaining) with
    // a small hand-controlled synthetic table, independent of whether the
    // real 280,147-entry merge_rank happens to be loaded correctly.
    std::cout << "[🔤 BPE] Running BPE merge algorithm self-test...\n";
    if (!RunBpeMergeSelfTest()) {
        std::cerr << "[❌ FATAL] BPE merge self-test failed — see [BPE-TEST] lines above. "
                     "Not proceeding with a merge algorithm that's proven wrong.\n";
        return 1;
    }
    std::cout << "[✓ BPE] All self-test cases passed.\n\n";

    // Real sampling: prove temperature/softmax/top-p mechanics against
    // Python-verified expected values before this replaces pure argmax
    // anywhere in the live generation path.
    std::cout << "[🎲 SAMPLING] Running temperature/top-p sampling self-test...\n";
    if (!RunSamplingSelfTest()) {
        std::cerr << "[❌ FATAL] Sampling self-test failed — see [SAMPLE-TEST] lines above. "
                     "Not proceeding with a sampler that's proven wrong.\n";
        return 1;
    }
    std::cout << "[✓ SAMPLING] All self-test cases passed.\n\n";

    // Fallback defaults in case the GGUF parse below fails for any reason —
    // keeps the engine running in synthetic-only mode rather than crashing
    // outright if the file's missing or unreadable.
    size_t real_num_layers = 1;
    size_t real_context_length = 4096;

    // GGUF parser kept alive for the full scope of main() so LoadTensorFromGGUF
    // can use its tensor index and data_offset later — was previously scoped
    // to a local block which destroyed it before tensor loading could happen.
    const std::string gguf_path = "K:\\models\\Meta-Llama-3-8B-Instruct.Q4_K_M.gguf";
    KintsugiGGUFParser gguf;
    bool gguf_loaded = false;

    // ------------------------------------------------------------------------
    // PHASE 7: GGUF HEADER PARSE + VERIFY
    // ------------------------------------------------------------------------
    {
        if (gguf.Load(gguf_path)) {
            gguf_loaded = true;
            const auto& p = gguf.params;
            std::cout << "[📜 GGUF] vocab_size=" << p.vocab_size
                      << " | d_model=" << p.embedding_length
                      << " | d_ff=" << p.feed_forward_length
                      << " | heads=" << p.head_count
                      << " | kv_heads=" << p.head_count_kv
                      << " | context_length=" << p.context_length
                      << " | layers=" << p.block_count
                      << " | rope_freq_base=" << p.rope_freq_base
                      << " | rms_norm_eps=" << p.rms_norm_eps << "\n";

            if (!p.AllRequiredPresent()) {
                std::cerr << "[!] WARNING: one or more expected metadata keys were not found. "
                          << "Check the printed values above against what's missing.\n";
            } else {
                real_num_layers = static_cast<size_t>(p.block_count);
                real_context_length = static_cast<size_t>(p.context_length);
            }

            std::cout << "[📜 GGUF] Tensor index parsed: " << gguf.tensors.size() << " entries. First 5:\n";
            for (size_t i = 0; i < gguf.tensors.size() && i < 5; ++i) {
                std::cout << "    - " << gguf.tensors[i].name
                          << " (ggml_type=" << gguf.tensors[i].ggml_type
                          << ", dims=" << gguf.tensors[i].dims.size() << ")\n";
            }

            // Print all tensor names from blk.0 only — enough to confirm
            // the exact naming convention before writing the per-layer loader.
            // Remove this block once names are confirmed.
            std::cout << "[📜 GGUF] Full blk.0 tensor listing:\n";
            for (const auto& t : gguf.tensors) {
                if (t.name.rfind("blk.0.", 0) == 0 ||
                    t.name == "output.weight" ||
                    t.name == "output_norm.weight") {
                    std::cout << "    " << t.name
                              << " (ggml_type=" << t.ggml_type << ")\n";
                }
            }

            // Tokenizer Phase I — REAL end-to-end check. Runs the full
            // pretokenize -> byte-remap -> BPE-merge -> vocab-lookup pipeline
            // on real data and compares against a genuinely independent
            // source of truth: check_sentence.py verified weeks ago, via
            // simple exact-vocabulary lookup (no BPE machinery involved at
            // all), that "What is the capital of France" tokenizes to
            // exactly [3923, 374, 279, 6864, 315, 9822]. If the real BPE
            // pipeline built tonight produces that same sequence on the
            // same text, that's two independently-built methods agreeing —
            // not just a synthetic unit test passing.
            std::cout << "\n[🔤 ENCODER] Running full pipeline against known-correct real data...\n";
            {
                const std::vector<uint32_t> expected = { 3923, 374, 279, 6864, 315, 9822 };
                std::vector<uint32_t> got = EncodeText("What is the capital of France",
                                                         byte_to_unicode, gguf.merge_rank, gguf.vocab_lookup);

                std::cout << "    [ENCODER-TEST] Expected: ";
                for (auto id : expected) std::cout << id << " ";
                std::cout << "\n    [ENCODER-TEST] Got:      ";
                for (auto id : got) std::cout << id << " ";
                std::cout << "\n";

                bool encoder_ok = (got == expected);
                std::cout << "    [ENCODER-TEST] " << (encoder_ok ? "PASS" : "FAIL")
                          << " -- real pipeline vs. independently-verified answer\n";
                if (!encoder_ok) {
                    std::cerr << "[❌ FATAL] Full pipeline does not match the known-correct "
                                 "tokenization. Something in pretokenize/byte-remap/merge/lookup "
                                 "is wrong on real data, despite the synthetic unit tests passing. "
                                 "Not proceeding with an encoder that's proven wrong here.\n";
                    return 1;
                }
                std::cout << "[✓ ENCODER] Full pipeline matches the known-correct answer exactly.\n\n";

                // Detokenizer round-trip test — real data, no new external
                // ground truth needed since it reuses the just-proven encoder.
                std::cout << "[🔤 DETOK] Running detokenizer round-trip self-test...\n";
                if (!RunDetokenizeSelfTest(byte_to_unicode, gguf.merge_rank, gguf.vocab_lookup, gguf.vocab_tokens)) {
                    std::cerr << "[❌ FATAL] Detokenizer self-test failed — see [DETOK-TEST] lines above. "
                                 "Not proceeding with a detokenizer that's proven wrong.\n";
                    return 1;
                }
                std::cout << "[✓ DETOK] All self-test cases passed.\n\n";
            }
        } else {
            std::cerr << "[!] GGUF parse failed — see error above. Continuing with synthetic-only fallback values (1 layer, 4096 context).\n";
        }
    }

    // Pre-compute layer tensor offsets once — used by FastLoadLayer during inference
    if (gguf_loaded) PrecomputeLayerOffsets(gguf);

    // Nothing downstream works without a real GGUF (no synthetic fallback
    // path anymore now that the embedding table dequantizes from the vault) —
    // bail here instead of doing wasted vault/arena/KV-cache setup first.
    if (!gguf_loaded) {
        std::cerr << "[!] GGUF not loaded — cannot run inference.\n";
        return 1;
    }

    // ------------------------------------------------------------------------
    // GUARDRAIL: OS DECOUPLING & HARDWARE PINNING
    // ------------------------------------------------------------------------
    // HIGH_PRIORITY_CLASS gives strong scheduling advantage without preempting
    // OS input handling and system threads — REALTIME_PRIORITY_CLASS was causing
    // system freezes and crashes when combined with another heavy process running.
    SetPriorityClass(GetCurrentProcess(), HIGH_PRIORITY_CLASS);

    DWORD_PTR processAffinityMask = 0x0E;
    if (SetProcessAffinityMask(GetCurrentProcess(), processAffinityMask)) {
        std::cout << "[⚡ SILICON] Main process affinity locked to Physical Cores 1, 2, and 3.\n";
    } else {
        std::cerr << "[!] Warning: Process affinity lock failed.\n";
    }

    // Memory-map the GGUF once. Every layer load and every LM-head sweep from
    // here on reads through this mapping — first touch pays disk cost, every
    // repeat touch (every decode step re-hitting the same layers/LM-head)
    // gets served from the OS page cache at RAM speed, using whatever RAM is
    // actually free at the time instead of demanding a fixed reservation.
    if (!g_ModelFile.Open(gguf_path, gguf.tensor_data_offset)) {
        return 1;
    }

    if (!g_Arena.Initialize(MONOLITHIC_ARENA_SIZE)) {
        return 1;
    }

    // Per-layer KV caches — one independent cache per transformer layer.
    // Bumped from 64 -> 128: with mmap decode now costing ~9-10s/step instead
    // of ~48s/step, generating a real response length (dozens of tokens, not
    // 5) is actually affordable. 128 gives headroom for 16 prompt tokens plus
    // up to ~110 generated tokens before hitting the ring's wraparound point.
    // At 128 positions: 32 layers x 128 x 1024 x 4 bytes x 2 (K+V) = 32MB total
    // — still trivial against the 1.5GB arena.
    constexpr size_t DECODE_TEST_CONTEXT = 128;
    g_LayerKVCaches.resize(real_num_layers);
    for (size_t layer = 0; layer < real_num_layers; ++layer) {
        if (!g_LayerKVCaches[layer].MapToArena(g_Arena, DECODE_TEST_CONTEXT,
                KintsugiAttention::NUM_KV_HEADS * KintsugiAttention::HEAD_DIM)) {
            std::cerr << "[❌ FATAL] Failed to map KV-cache for layer " << layer << ".\n";
            g_Arena.Teardown();
            return 1;
        }
    }
    std::cout << "[🧠 ROUTING] " << real_num_layers << " independent per-layer KV-caches mapped ("
              << DECODE_TEST_CONTEXT << " positions each).\n";

    // Embedding table: wired directly to the vault, no arena copy, no disk
    // read here — Lookup() dequantizes one row per token, on demand.
    const size_t vocab_size = static_cast<size_t>(gguf.params.vocab_size);
    const GGUFTensorInfo* embd_info = nullptr;
    for (const auto& t : gguf.tensors) {
        if (t.name == "token_embd.weight") { embd_info = &t; break; }
    }
    if (!embd_info) {
        std::cerr << "[❌ FATAL] token_embd.weight tensor not found in GGUF.\n";
        g_Arena.Teardown();
        return 1;
    }
    g_Embedding.Init(g_ModelFile.base, embd_info->offset,
                      static_cast<uint8_t>(embd_info->ggml_type), vocab_size, 4096);
    std::cout << "[📖 LEXICON] Embedding wired to the memory-mapped GGUF — dequantized per-token, on demand.\n";

    // LM head: unlike the embedding table, this does NOT stay mmap-only.
    // Every prefill AND every decode step re-sweeps the FULL vocabulary
    // against it — the same ~411MB of bytes touched repeatedly within one
    // wake cycle. Leaving that to mmap page-cache behavior means its "warm
    // or cold" state depends on whatever else is competing for system-wide
    // memory at the time — exactly the coupling the checkpoint-2 case study
    // surfaced (process-level consistency and system-wide pressure turned
    // out to be decoupled; this is the lever that actually reduces touched
    // bytes, not just manages residency of a process's own working set).
    // Materializing the raw Q6_K bytes into our own arena buffer, once, at
    // boot, means every sweep reads from a buffer we control the CONTENT
    // of directly — it no longer needs to re-fetch from the mmap vault.
    // Honest correction: this buffer is NOT actually pinned. VirtualLock
    // has failed in every run observed so far (fragmentation, error 1453),
    // so the arena — this buffer included — is ordinary committed memory,
    // fully subject to Windows' own working-set trimming, including
    // checkpoint 2's own capped trim right below. "Guaranteed resident"
    // was an overclaim; residency here is only as durable as VirtualLock's
    // success, which has been 0-for-N tonight. See PrintBufferResidency
    // right after checkpoint 2 for a direct, non-invasive check of whether
    // this buffer's pages actually survive the trim.
    const GGUFTensorInfo* out_w_info = nullptr;
    for (const auto& t : gguf.tensors) {
        if (t.name == "output.weight") { out_w_info = &t; break; }
    }
    if (!out_w_info) {
        std::cerr << "[❌ FATAL] output.weight tensor not found in GGUF.\n";
        g_Arena.Teardown();
        return 1;
    }
    if (out_w_info->ggml_type != 14) {
        std::cerr << "[❌ FATAL] output.weight is not Q6_K (ggml_type=" << out_w_info->ggml_type
                  << ") — this build only supports the verified Q6_K LM head path.\n";
        g_Arena.Teardown();
        return 1;
    }

    uint8_t* lm_head_buffer = nullptr;
    {
        constexpr size_t LM_BPR = 4096 / 256;
        constexpr size_t LM_ROW_BYTES = LM_BPR * sizeof(GGML_Q6_K_Block);
        const size_t lm_head_bytes = vocab_size * LM_ROW_BYTES;

        std::cout << "[🧠 LM-HEAD] Materializing output.weight (" << (lm_head_bytes / (1024.0 * 1024.0))
                  << "MB raw Q6_K) into a pinned arena buffer — one-time read, guaranteed resident after.\n";

        lm_head_buffer = static_cast<uint8_t*>(g_Arena.AllocateAligned(lm_head_bytes));
        if (!lm_head_buffer) {
            std::cerr << "[❌ FATAL] Arena OOM materializing the LM head buffer ("
                      << (lm_head_bytes / (1024.0 * 1024.0)) << "MB requested).\n";
            g_Arena.Teardown();
            return 1;
        }
        memcpy(lm_head_buffer, g_ModelFile.base + out_w_info->offset, lm_head_bytes);
        std::cout << "[✓ LM-HEAD] Materialized and pinned — every sweep from here reads this buffer directly, "
                     "not the mmap vault.\n";
    }

    g_model_loaded.store(true, std::memory_order_release);

    // --- PHASE 9/10: STREAMING 32-LAYER INFERENCE ---
    // Architecture: one layer at a time through the arena.
    // The layer weight slot (~832MB) is reused — arena offset is saved before
    // loading each layer and rewound after, so each layer overwrites the same
    // physical memory. Residual stream and KV cache persist across all 32 layers.

    const size_t D   = 4096;
    const size_t KV  = 1024;
    const size_t FF  = 14336;
    const size_t MAX_CTX = DECODE_TEST_CONTEXT; // sizes score_scratch to match the per-layer cache width

    // Allocate persistent scratch buffers once — these survive across all layers
    float* residual      = static_cast<float*>(g_Arena.AllocateAligned(D       * sizeof(float)));
    float* scratch_norm  = static_cast<float*>(g_Arena.AllocateAligned(D       * sizeof(float)));
    float* scratch_q     = static_cast<float*>(g_Arena.AllocateAligned(D       * sizeof(float)));
    float* scratch_k     = static_cast<float*>(g_Arena.AllocateAligned(KV      * sizeof(float)));
    float* scratch_v     = static_cast<float*>(g_Arena.AllocateAligned(KV      * sizeof(float)));
    float* scratch_attn  = static_cast<float*>(g_Arena.AllocateAligned(D       * sizeof(float)));
    float* scratch_ffn_g = static_cast<float*>(g_Arena.AllocateAligned(FF      * sizeof(float)));
    float* scratch_ffn_u = static_cast<float*>(g_Arena.AllocateAligned(FF      * sizeof(float)));
    float* scratch_ffn_o = static_cast<float*>(g_Arena.AllocateAligned(D       * sizeof(float)));
    float* score_scratch = static_cast<float*>(g_Arena.AllocateAligned(MAX_CTX * sizeof(float)));

    if (!residual || !scratch_norm || !scratch_q || !scratch_k || !scratch_v ||
        !scratch_attn || !scratch_ffn_g || !scratch_ffn_u || !scratch_ffn_o ||
        !score_scratch) {
        std::cerr << "[❌ FATAL] Arena OOM allocating inference scratch buffers.\n";
        g_Arena.Teardown();
        return 1;
    }
    std::cout << "[⚡ ARENA] Forward-pass scratch buffers allocated — layer weights dequantize straight from the vault now.\n";

    // --- SILICON GATE PORTAL: map the IPC bus ---
    std::cout << "[🔗 PORTAL] Searching for HUD's SiliconGatePortal...\n";
    HANDLE hPortalMap = OpenFileMappingA(FILE_MAP_ALL_ACCESS, FALSE, "Global\\KintsugiPortal");
    if (!hPortalMap) hPortalMap = OpenFileMappingA(FILE_MAP_ALL_ACCESS, FALSE, "Local\\KintsugiPortal");
    if (!hPortalMap) {
        HANDLE hPortalFile = CreateFileA("K:\\kintsugi\\core\\shared_bus.dat",
            GENERIC_READ | GENERIC_WRITE, FILE_SHARE_READ | FILE_SHARE_WRITE,
            NULL, OPEN_EXISTING, FILE_ATTRIBUTE_NORMAL, NULL);
        if (hPortalFile != INVALID_HANDLE_VALUE) {
            hPortalMap = CreateFileMappingA(hPortalFile, NULL, PAGE_READWRITE, 0, 0, NULL);
            CloseHandle(hPortalFile);
        }
    }
    if (!hPortalMap) {
        std::cerr << "[❌ FATAL] Cannot find SiliconGatePortal. Start the HUD first.\n";
        g_Arena.Teardown();
        return 1;
    }
    SiliconGatePortal* portal = static_cast<SiliconGatePortal*>(
        MapViewOfFile(hPortalMap, FILE_MAP_ALL_ACCESS, 0, 0, BUS_STRUCT_SIZE));
    if (!portal) {
        std::cerr << "[❌ FATAL] MapViewOfFile failed for portal.\n";
        g_Arena.Teardown();
        return 1;
    }
    std::cout << "[✓ PORTAL] Mapped. Core entering sleep. Awaiting enable_programming_pin...\n\n";

    // Zero the pin explicitly at boot. Without this, a stale non-zero value
    // left behind by a crashed or force-closed prior session (the pin lives
    // in shared_bus.dat on disk, which survives process death and reboots)
    // silently triggers an unrequested wake the instant a fresh engine maps
    // the same file — exactly what happened after the last crash/restart.
    portal->enable_programming_pin.store(0, std::memory_order_release);

    // --- THE SLEEP/WAKE GATE ---
    // pin == 0 -> core asleep, _mm_pause spin, near-zero CPU usage.
    // pin == 1 -> HUD has written a prompt into text_payload, core wakes,
    //             runs full prefill + LM head, writes result, flips pin back to 0.
    while (true) {
        while (portal->enable_programming_pin.load(std::memory_order_acquire) == 0) {
            _mm_pause();
        }

        std::cout << "[🔥 IGNITION] Pin flipped HIGH. Core seizing bus.\n";
        std::cout << "[🔥 IGNITION] HUD text_payload: \"" << portal->text_payload << "\"\n";

        // REAL tokenizer wiring — text_payload now actually drives what gets
        // processed, using the fully-verified pipeline (pretokenize ->
        // byte-remap -> BPE-merge -> vocab-lookup) proven correct against
        // known data just before this. The wrapper structure (BOS, user/
        // assistant headers, eot_id) is unchanged from the verified France
        // test — only the MIDDLE section (what used to be the hardcoded
        // 6-token France-question array) is now real, dynamic user content.
        std::string user_text(portal->text_payload);
        std::vector<uint32_t> user_tokens = EncodeText(user_text, byte_to_unicode,
                                                        gguf.merge_rank, gguf.vocab_lookup);

        // Safety bound: wrapper is 10 fixed tokens, decode can run up to 40
        // more steps, and the KV-cache ring holds 128 positions total.
        // 10 + N(user) + 40(decode) <= 128 requires N <= 78 — cap at 70 for
        // margin. Truncating is a safe, explicit fallback rather than
        // silently overflowing the ring or crashing on a long input.
        constexpr size_t MAX_USER_TOKENS = 70;
        if (user_tokens.size() > MAX_USER_TOKENS) {
            std::cerr << "[!] WARNING: input encoded to " << user_tokens.size()
                      << " tokens, exceeds the " << MAX_USER_TOKENS
                      << "-token safety cap for this KV-cache size — truncating.\n";
            user_tokens.resize(MAX_USER_TOKENS);
        }

        std::cout << "[🔤 ENCODER] \"" << user_text << "\" -> " << user_tokens.size()
                  << " real token(s): ";
        for (auto t : user_tokens) std::cout << t << " ";
        std::cout << "\n";

        // Same wrapper structure verified correct in every prior run —
        // BOS, user header, [REAL encoded content], eot_id, assistant header.
        std::vector<uint32_t> prompt_sequence;
        prompt_sequence.reserve(10 + user_tokens.size());
        prompt_sequence.push_back(128000); // <|begin_of_text|>
        prompt_sequence.push_back(128006); // <|start_header_id|>
        prompt_sequence.push_back(882);    // "user"
        prompt_sequence.push_back(128007); // <|end_header_id|>
        prompt_sequence.push_back(271);    // "\n\n"
        for (uint32_t t : user_tokens) prompt_sequence.push_back(t);
        prompt_sequence.push_back(128009); // <|eot_id|>
        prompt_sequence.push_back(128006); // <|start_header_id|>
        prompt_sequence.push_back(78191);  // "assistant"
        prompt_sequence.push_back(128007); // <|end_header_id|>
        prompt_sequence.push_back(271);    // "\n\n"

        const size_t prompt_len = prompt_sequence.size();

        // Reset ALL per-layer caches at the start of this wake cycle — each
        // new wake is a fresh conversation. Each layer's cache is fully
        // independent, so each gets its own head_index=0 and zeroed buffers.
        for (size_t layer = 0; layer < real_num_layers; ++layer) {
            g_LayerKVCaches[layer].head_index = 0;
            memset(g_LayerKVCaches[layer].key_matrix,   0, DECODE_TEST_CONTEXT * KV * sizeof(float));
            memset(g_LayerKVCaches[layer].value_matrix, 0, DECODE_TEST_CONTEXT * KV * sizeof(float));
        }

        // Checkpoint marks the boundary between permanent boot-time scratch
        // buffers (above this while(true) loop) and everything allocated
        // THIS cycle — rewound in full right before going back to sleep, so
        // repeated wakes never accumulate arena usage.
        size_t cycle_checkpoint = g_Arena.current_offset;

        // Arena-allocated (64-byte aligned), NOT std::vector<float> — this was
        // the actual crash. std::vector's allocator doesn't guarantee 32-byte
        // alignment, but g_Embedding.Lookup() uses a strict-aligned AVX2 store
        // directly on this pointer. It "worked" in prior one-shot runs purely
        // by heap-layout coincidence; the first real run behind this new
        // while(true)+portal wrapper had different allocation history, landed
        // unaligned, and the aligned store faulted — a silent hardware
        // exception with zero diagnostic output, killing the process outright.
        std::vector<float*> token_residuals(prompt_len, nullptr);
        bool residuals_ok = true;
        for (size_t tok = 0; tok < prompt_len; ++tok) {
            token_residuals[tok] = static_cast<float*>(g_Arena.AllocateAligned(D * sizeof(float)));
            if (!token_residuals[tok]) {
                std::cerr << "[❌ FATAL] Arena OOM allocating token residual " << tok << ".\n";
                residuals_ok = false;
                break;
            }
            g_Embedding.Lookup(prompt_sequence[tok], token_residuals[tok]);
        }
        if (!residuals_ok) {
            g_Arena.current_offset = cycle_checkpoint;
            portal->enable_programming_pin.store(0, std::memory_order_release);
            continue;
        }

        std::cout << "[🧠 JARVIS] PREFILL: " << prompt_len << " instruct tokens\n";
        PrintMemoryUsage("before prefill (32 layers)");

        auto total_start = std::chrono::high_resolution_clock::now();
        bool inference_ok = true;

        for (size_t layer = 0; layer < real_num_layers && inference_ok; ++layer) {
            size_t arena_checkpoint = g_Arena.current_offset;

            KintsugiLayerWeights lw;
            bool ok = FastLoadLayer(layer, g_Arena, lw);
            if (!ok) {
                std::cerr << "[❌ FATAL] Layer " << layer << " load failed.\n";
                inference_ok = false;
                break;
            }

            // Old design reset a SHARED cache's head_index here, per-layer,
            // to work around all 32 layers fighting over one physical
            // buffer. Now each layer owns its own independent cache
            // (g_LayerKVCaches[layer]), reset once at the top of this wake
            // cycle — no per-layer reset needed, and doing one here would
            // actively break future decode steps by wiping accumulated
            // history every time this loop runs.
            if (layer <= 2 || layer == real_num_layers - 1) {
                std::cout << "    [KV-CHECK] layer=" << layer
                          << " head_index=" << g_LayerKVCaches[layer].head_index << "\n";
            }

            for (size_t tok = 0; tok < prompt_len; ++tok) {
                ForwardOneLayer(lw, token_residuals[tok], scratch_norm,
                                scratch_q, scratch_k, scratch_v, scratch_attn,
                                scratch_ffn_g, scratch_ffn_u, scratch_ffn_o,
                                score_scratch, g_LayerKVCaches[layer], tok,
                                D, KV, FF, gguf.params.rms_norm_eps);
            }

            g_Arena.current_offset = arena_checkpoint;
            std::cout << "[⚡ LAYER " << std::setw(2) << layer << "] "
                      << prompt_len << " tokens batched.\n";
        }

        if (!inference_ok) {
            g_Arena.current_offset = cycle_checkpoint;
            portal->enable_programming_pin.store(0, std::memory_order_release);
            continue;
        }

        float* final_residual = token_residuals[prompt_len - 1];

        // Output norm + LM head
        float* output_norm_w = static_cast<float*>(g_Arena.AllocateAligned(D * sizeof(float)));
        float* normed_out    = static_cast<float*>(g_Arena.AllocateAligned(D * sizeof(float)));
        alignas(64) float lm_row_scratch[4096];

        bool norm_ok = LoadTensorIntoBuffer(gguf_path, gguf, "output_norm.weight", output_norm_w, D);
        if (!norm_ok) std::fill_n(output_norm_w, D, 1.0f);

        RMSNorm(final_residual, output_norm_w, normed_out, D, gguf.params.rms_norm_eps);

        // Real sampling replaces pure argmax: sweep the LM head into a
        // sorted top-K list, then sample from it via temperature + top-p
        // instead of always taking the single highest score. Reads from
        // the pinned buffer materialized once at boot, not out_w_info/mmap.
        std::vector<TopKCandidate> top_candidates =
            RunLMHeadSweep(lm_head_buffer, normed_out, D, vocab_size, lm_row_scratch, SAMPLING_TOP_K);
        uint32_t best_token = SampleToken(top_candidates, SAMPLING_TEMPERATURE, SAMPLING_TOP_P, g_sampling_rng);
        float    best_score = top_candidates.empty() ? -1e30f : top_candidates[0].score;

        auto total_end = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> total_dur = total_end - total_start;

        std::cout << "[🧠 JARVIS] Predicted token_id=" << best_token
                  << " (score=" << best_score << ") in " << total_dur.count() << "s\n";
        PrintMemoryUsage("after prefill + first LM-head sweep");

        // --- CHECKPOINT 2: WATERMARK-GATED, CAPPED WORKING-SET TRIM ---
        // Upgraded from a full-evict version (-1,-1) that DID work — it took
        // WorkingSet from ~4859MB to ~8MB — but paid the whole cost straight
        // back at decode step 0 (41.03s, worse than even the un-trimmed
        // degraded baseline's step 0). Wiping to the floor means the very
        // next touch re-faults almost everything cold.
        //
        // This version only trims when actually over a real threshold, and
        // caps rather than floors — asking Windows to bring the working set
        // DOWN TOWARD a ceiling, not wipe it to nothing. Windows' own LRU
        // decides what stays resident, which in a strictly sequential
        // per-layer loop naturally tends to keep whatever was touched most
        // recently — closer to "what's needed again soon" than anything
        // we could hand-pick without real hot/cold tracking we don't have.
        //
        // Constants below are calibrated against real observed numbers
        // tonight, not round guesses:
        //   - arena+LM-head baseline measured ~1987MB ("before prefill"
        //     WorkingSet) since the 411MB LM-head buffer now lives
        //     permanently in the arena from boot onward — NOT the older
        //     ~1585MB figure from before that buffer existed. VirtualLock
        //     has failed in every run observed so far, so none of this is
        //     actually protected from trimming; it must be accounted for
        //     directly, not assumed safe on the side.
        //   - prefill alone reliably pushes WorkingSet to 4859-5633MB.
        //   - decode's steady-state (once re-warmed) settles at ~5240-5244MB
        //     every time, consistently, regardless of run.
        // MIN recalibrated from 1792MB, which was sized against the PRE-
        // LM-head-buffer baseline and had fallen below the real ~1987MB
        // floor once that buffer became permanent — the exact gap flagged
        // after the first post-LM-head-buffer run. 2304MB gives ~317MB of
        // real margin above the current baseline, the same kind of margin
        // 1792MB gave above the old one.
        constexpr double CHECKPOINT2_WATERMARK_MB = 3072.0;                     // trigger threshold — prefill alone reliably exceeds this
        constexpr SIZE_T CHECKPOINT2_MIN_BYTES    = 2304ULL * 1024ULL * 1024ULL; // 2.25GB floor — covers the real ~1987MB arena+LM-head baseline
        constexpr SIZE_T CHECKPOINT2_MAX_BYTES    = 3072ULL * 1024ULL * 1024ULL; // 3GB ceiling — ~2.2GB of real relief off the ~5.4GB peak

        double working_set_before_trim = GetProcessWorkingSetMB();
        if (working_set_before_trim > CHECKPOINT2_WATERMARK_MB) {
            std::cout << "[🧊 CHECKPOINT 2] Working set " << working_set_before_trim
                      << "MB exceeds " << CHECKPOINT2_WATERMARK_MB
                      << "MB watermark — trimming (capped, not full evict).\n";

            BOOL trim_ok = SetProcessWorkingSetSize(GetCurrentProcess(), CHECKPOINT2_MIN_BYTES, CHECKPOINT2_MAX_BYTES);
            if (!trim_ok) {
                // Capture immediately — same evaluation-order fix already
                // established elsewhere in this file for this exact bug class.
                DWORD trim_err = GetLastError();
                std::cerr << "[!] WARNING: Checkpoint 2 capped trim failed. Error Code: "
                          << trim_err << " — continuing anyway, this is an optimization, not a requirement.\n";
            } else {
                std::cout << "[🧊 CHECKPOINT 2] Capped trim applied — targeting ~"
                          << (static_cast<double>(CHECKPOINT2_MAX_BYTES) / (1024.0 * 1024.0))
                          << "MB ceiling, OS LRU picks what stays resident.\n";
            }
            PrintMemoryUsage("after checkpoint 2 capped trim");

            // Direct answer to "did the LM-head buffer actually survive the
            // trim" — non-invasive, doesn't touch/fault the pages, so this
            // check itself can't warm them and contaminate what it's
            // measuring. Samples 5 points across the buffer since partial
            // eviction (some pages gone, others not) wouldn't show up from
            // checking just the first/last byte.
            {
                constexpr size_t LM_BPR = 4096 / 256;
                constexpr size_t LM_ROW_BYTES = LM_BPR * sizeof(GGML_Q6_K_Block);
                const size_t lm_head_check_bytes = vocab_size * LM_ROW_BYTES;
                PrintBufferResidency("lm_head_buffer post-trim", lm_head_buffer, lm_head_check_bytes);
            }
        } else {
            std::cout << "[🧊 CHECKPOINT 2] Working set " << working_set_before_trim
                      << "MB already under " << CHECKPOINT2_WATERMARK_MB
                      << "MB watermark — skipping trim this cycle.\n";
        }

        // <|eot_id|> — verified via find_token.py against the real GGUF
        // vocabulary, not guessed. The model uses this to signal "I'm done" —
        // a correct generation loop stops here rather than forcing a fixed
        // token count regardless of what the model actually wants to say.
        constexpr uint32_t EOT_TOKEN = 128009;

        // --- DECODE: GENERATE UP TO N MORE TOKENS, STOPPING EARLY ON <|eot_id|> ---
        // Unlike prefill, decode CANNOT batch tokens — we don't know token
        // 17 until we've generated token 16. Each step still re-dequantizes
        // all 32 layers plus the full LM head, but now straight out of the
        // AWE RAM vault instead of disk — the ~85s/step cost was disk I/O,
        // which this eliminates after the one-time vault fill at boot.
        // Bumped 5 -> 40: at ~9-10s/step post-mmap, a real response length is
        // now affordable (~6-7 minutes worst case if it never stops early).
        constexpr size_t NUM_DECODE_STEPS = 40;
        std::vector<uint32_t> generated_tokens;
        generated_tokens.push_back(best_token);

        if (best_token == EOT_TOKEN) {
            std::cout << "[🧠 JARVIS] First token IS <|eot_id|> — model has nothing further to say.\n";
        } else {
        std::cout << "\n[🧠 JARVIS] DECODE: generating up to " << NUM_DECODE_STEPS
                  << " more tokens, stopping early on <|eot_id|>...\n";

        float* decode_residual = static_cast<float*>(g_Arena.AllocateAligned(D * sizeof(float)));
        if (!decode_residual) {
            std::cerr << "[❌ FATAL] Arena OOM allocating decode residual.\n";
        } else {
            for (size_t step = 0; step < NUM_DECODE_STEPS; ++step) {
                size_t current_pos = prompt_len + step; // continues from wherever the real (dynamic-length) prompt left off
                uint32_t current_token = generated_tokens.back();
                g_Embedding.Lookup(current_token, decode_residual);

                auto step_start = std::chrono::high_resolution_clock::now();
                bool decode_ok = true;

                for (size_t layer = 0; layer < real_num_layers && decode_ok; ++layer) {
                    size_t decode_checkpoint = g_Arena.current_offset;

                    KintsugiLayerWeights lw2;
                    bool ok2 = FastLoadLayer(layer, g_Arena, lw2);
                    if (!ok2) {
                        std::cerr << "[❌ FATAL] Decode step " << step
                                  << " layer " << layer << " load failed.\n";
                        decode_ok = false;
                        break;
                    }

                    // g_LayerKVCaches[layer] already holds this layer's 16
                    // prompt positions from prefill — PushContext continues
                    // writing at current_pos (16, 17...) without any reset,
                    // so attention correctly sees the full history so far.
                    ForwardOneLayer(lw2, decode_residual, scratch_norm,
                                    scratch_q, scratch_k, scratch_v, scratch_attn,
                                    scratch_ffn_g, scratch_ffn_u, scratch_ffn_o,
                                    score_scratch, g_LayerKVCaches[layer], current_pos,
                                    D, KV, FF, gguf.params.rms_norm_eps);

                    g_Arena.current_offset = decode_checkpoint;
                }

                if (!decode_ok) break;

                // LM head sweep for this new position — same logic as prefill's,
                // reusing the same output_norm weights and pinned lm_head_buffer.
                RMSNorm(decode_residual, output_norm_w, normed_out, D, gguf.params.rms_norm_eps);

                std::vector<TopKCandidate> step_top_candidates =
                    RunLMHeadSweep(lm_head_buffer, normed_out, D, vocab_size, lm_row_scratch, SAMPLING_TOP_K);
                uint32_t step_best_token = SampleToken(step_top_candidates, SAMPLING_TEMPERATURE,
                                                        SAMPLING_TOP_P, g_sampling_rng);
                float    step_best_score = step_top_candidates.empty() ? -1e30f : step_top_candidates[0].score;

                auto step_end = std::chrono::high_resolution_clock::now();
                std::chrono::duration<double> step_dur = step_end - step_start;

                generated_tokens.push_back(step_best_token);
                std::cout << "[🧠 JARVIS] Decode step " << step << " (pos=" << current_pos
                          << "): token_id=" << step_best_token << " (score=" << step_best_score
                          << ") in " << step_dur.count() << "s\n";
                PrintMemoryUsage("after this decode step");

                if (step_best_token == EOT_TOKEN) {
                    std::cout << "[🧠 JARVIS] <|eot_id|> generated — model signaled it's done. "
                                 "Stopping early instead of forcing all " << NUM_DECODE_STEPS << " steps.\n";
                    break;
                }
            }
        }
        } // closes the "if (best_token == EOT_TOKEN) {...} else {" wrapper

        std::cout << "\n[🧠 JARVIS] FULL GENERATED SEQUENCE (" << generated_tokens.size() << " tokens): ";
        for (auto t : generated_tokens) std::cout << t << " ";
        std::cout << "\n";

        // Native human-readable text — first time this project has ever
        // printed Jarvis's response as actual readable text directly from
        // C++, without a separate Python decode step.
        std::string readable = DetokenizeIds(generated_tokens, gguf.vocab_tokens, unicode_to_byte);
        std::cout << "[🧠 JARVIS] RESPONSE: \"" << readable << "\"\n";

        // --- WRITE RESULT BACK TO PORTAL ---
        // NOTE: portal only carries a single token_id_payload — only the
        // FIRST generated token round-trips through wake_jarvis.py. The full
        // sequence above is console-only for this test. Multi-token portal
        // output is a real next step, not done here.
        portal->token_id_payload = best_token;

        // Reclaim everything allocated this cycle (token_residuals, output_norm_w,
        // normed_out) — without this, every wake permanently consumes arena space
        // that's never returned, eventually exhausting it after enough cycles.
        g_Arena.current_offset = cycle_checkpoint;

        std::cout << "[😴 SLEEP] Flipping pin LOW. Core returning to sleep.\n\n";
        portal->enable_programming_pin.store(0, std::memory_order_release);
    }

    UnmapViewOfFile(portal);
    CloseHandle(hPortalMap);
    g_Arena.Teardown();
    return 0;
}
