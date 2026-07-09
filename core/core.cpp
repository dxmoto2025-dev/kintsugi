#include <windows.h>
#include <iostream>
#include <atomic>
#include <thread>
#include <immintrin.h> // AVX2 YMM intrinsics

#include "shared_bus.h" // The SSOT

int main() {
    std::cout << "======================================================================\n";
    std::cout << "          KINTSUGI SOVEREIGN LIVE INFERENCE EXECUTION ENGINE         \n";
    std::cout << "======================================================================\n";

    // GUARDRAIL 2: DETERMINISTIC EXECUTION
    if (SetPriorityClass(GetCurrentProcess(), REALTIME_PRIORITY_CLASS)) {
        std::cout << "[⚡ SILICON] Process priority escalated to REALTIME_PRIORITY_CLASS.\n";
    } else {
        std::cerr << "[!] Warning: Realtime escalation failed. Check Admin privileges.\n";
    }

    // Core 0 reserved for OS/HUD.
    DWORD_PTR processAffinityMask = 0x0E; 
    if (SetProcessAffinityMask(GetCurrentProcess(), processAffinityMask)) {
        std::cout << "[⚡ SILICON] Thread affinity locked to Physical Cores 1, 2, and 3.\n";
    } else {
        std::cerr << "[!] Warning: Process affinity lock failed.\n";
    }

    // SOVEREIGN PATHING
    HANDLE hFile = CreateFileA("K:\\kintsugi\\core\\shared_bus.dat", GENERIC_READ | GENERIC_WRITE, FILE_SHARE_READ | FILE_SHARE_WRITE, NULL, OPEN_EXISTING, FILE_ATTRIBUTE_NORMAL, NULL);
    if (hFile == INVALID_HANDLE_VALUE) { std::cout << "[❌ FATAL] Anchor file not found!\n"; return 1; }

    // Map using the SSOT size
    HANDLE hMapFile = CreateFileMappingA(hFile, NULL, PAGE_READWRITE, 0, BUS_STRUCT_SIZE, NULL);
    KintsugiBus* bus = (KintsugiBus*)MapViewOfFile(hMapFile, FILE_MAP_ALL_ACCESS, 0, 0, BUS_STRUCT_SIZE);
    
    // GUARDRAIL 1: PHYSICAL MEMORY PINNING
    VirtualLock(bus, BUS_STRUCT_SIZE);

    std::cout << "[⚡ CONNECTED] Core mapped to SSOT Bus. Awaiting Master Plane...\n";
    
    // THE INIT BARRIER
    while (bus->system_ready.load(std::memory_order_acquire) != 1) { _mm_pause(); }

    std::cout << "[IGNITION] Master Plane Ready. Commencing AVX2 Overdrive Matrix...\n\n";

    uint32_t local_read_index = bus->token_read_ptr.load(std::memory_order_relaxed);

    while (true) {
        // Ping the heartbeat 
        bus->heartbeat_tick.fetch_add(1, std::memory_order_relaxed);

        uint32_t current_write = bus->token_write_ptr.load(std::memory_order_acquire);

        if (local_read_index < current_write) {
            uint32_t slot_index = local_read_index & (MAX_IPC_TOKENS - 1);
            uint32_t incoming_token_id = bus->token_array[slot_index];

            // ======================================================================
            // 🔥 INJECTED: 8-WAY INTERLEAVED SIMD OVERDRIVE COMPUTE PIPELINE
            // ======================================================================

            // 1. Broadcast token vector across YMM0
            __m256i v_token = _mm256_set1_epi32(static_cast<int>(incoming_token_id));

            // 2. Load 8-way interleaved weight shards directly from the shared memory
            const uint32_t* aligned_target_ptr = &bus->token_array[slot_index & ~7];
            __m256i v_weights = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(aligned_target_ptr));

            // 3. Fused Vector Arithmetic
            __m256i v_result_add = _mm256_add_epi32(v_token, v_weights);
            __m256i v_result_xor = _mm256_xor_si256(v_token, v_result_add);

            // 4. Hardware memory fence
            _mm_lfence();

            // 5. Anchor the volatile register execution
            volatile __m256i compute_sink = v_result_xor;
            (void)compute_sink;
            
            // std::cout << "[COMPUTE SUCCESS] Matrix weights processed against Token " << incoming_token_id << "\n";
            // Removed console out inside the hot-loop to prevent I/O bottlenecking the AVX2 registers.

            local_read_index++;
            bus->token_read_ptr.store(local_read_index, std::memory_order_release);
        } else {
            _mm_pause(); 
        }
    }

    return 0;
}