// ==============================================================================
// KINTSUGI: LARGE PAGE RESERVOIR
// ==============================================================================
// Safety net, not Plan A. kintsugi_inference.cpp's own large-page attempt
// (fresh VirtualAlloc + MEM_LARGE_PAGES, tried at every wake) has proven
// solid all night on its own — checkpoint 2's watermark trim and the LM-head
// buffer materialization were both built and verified specifically for the
// case where that attempt fails, and they hold up. This program exists for
// one narrow, real observation: Large Pages succeed reliably right after a
// fresh boot, and fail once the system's been running a while — because
// large pages need genuinely contiguous physical RAM, and no amount of
// application-level memory flushing puts fragmented physical frames back
// together. Only grabbing them early, before fragmentation sets in, and
// holding them, actually solves that.
//
// This program does exactly that: acquire a large-page-backed shared memory
// reservation once, at (or near) boot, and hold it open indefinitely. The
// main engine tries to find and use this reservation FIRST on every wake —
// see the "SAFETY NET, NOT PLAN A" block in TensorArena::Initialize(). If
// this program isn't running, the engine falls straight through to its own
// proven, self-contained attempt, completely unchanged. This is additive,
// not a dependency.
//
// CONSOLE OUTPUT NOTE (learned the hard way): this file uses std::cout with
// plain UTF-8 string literals plus SetConsoleOutputCP(CP_UTF8), exactly
// matching the pattern already proven working all night in
// kintsugi_inference.cpp. An earlier version of this file used std::wcout
// with wide L"..." literals instead — SetConsoleOutputCP only affects the
// narrow code page, does nothing for wide streams, and the emoji used here
// require UTF-16 surrogate pairs that Windows console's wcout handles badly.
// That version printed a single garbled line and exited almost immediately.
// Wide strings are used ONLY where the Windows API itself requires them —
// the W-suffixed functions' name parameters — never for console output.
//
// SETUP (Task Scheduler, not a full Windows Service — simpler, same effect):
//   1. Compile this file (see build command below).
//   2. Task Scheduler -> Create Task (not Basic Task, for full options)
//      - General: run as SYSTEM ("Run whether user is logged on or not",
//        check "Run with highest privileges"). Running as SYSTEM matters:
//        SeCreateGlobalPrivilege (needed for the Global\ namespace) is
//        enabled by default for the system account, so no extra privilege
//        code is needed here for that specific permission.
//      - Triggers: "At startup", with a short delay (~30s) so the OS has
//        settled before the large-page request fires.
//      - Actions: Start a program -> path to kintsugi_page_reservoir.exe
//   3. Reboot once to confirm it grabs the reservation cleanly and stays
//      resident — check Task Manager for kintsugi_page_reservoir.exe still
//      running, and watch for the engine's "[RESERVOIR] Found..." line on
//      its next wake instead of the usual Large Page fallback messages.
//
// BUILD: g++ -O3 -std=c++17 -o kintsugi_page_reservoir.exe kintsugi_page_reservoir.cpp
// (No special libs needed beyond the default Windows import libraries.)
// ==============================================================================

#include <windows.h>
#include <iostream>
#include <cstring>
#include <chrono>

// MUST match kintsugi_inference.cpp's MONOLITHIC_ARENA_SIZE exactly — the
// engine requests a view of exactly this size when opening the reservoir.
// If these two ever drift out of sync, the engine's MapViewOfFile call will
// fail (safely — it just falls through to its own fallback path, per the
// design), not corrupt anything, but the whole point of this program is
// defeated silently until someone notices the fallback message reappearing.
constexpr size_t RESERVOIR_SIZE_BYTES = 1536ULL * 1024ULL * 1024ULL; // 1.5GB

// Wide string ONLY because CreateFileMappingW/OpenFileMappingW's name
// parameter requires LPCWSTR — this is a Windows API requirement, not a
// console-output choice, so it's exempt from the narrow-only rule above.
const wchar_t* RESERVOIR_NAME = L"Local\\KintsugiLargePageReservoir";

HANDLE g_hMapping = nullptr;
void* g_pView = nullptr;

// Graceful shutdown: explicit unmap + close on Ctrl+C, console close, or
// service stop, rather than relying solely on process-exit cleanup. Also
// makes the teardown visible in the log instead of a silent disappearance.
BOOL WINAPI ConsoleCtrlHandler(DWORD ctrl_type) {
    std::cout << "[ RESERVOIR] Shutdown signal received (type=" << ctrl_type
              << ") -- releasing the large-page reservation cleanly.\n" << std::flush;
    if (g_pView) { UnmapViewOfFile(g_pView); g_pView = nullptr; }
    if (g_hMapping) { CloseHandle(g_hMapping); g_hMapping = nullptr; }
    return TRUE; // handled — allow the default termination to proceed after cleanup
}

int main() {
    // Same setup already proven working all night: SetConsoleOutputCP paired
    // with std::cout + narrow UTF-8 literals, NOT std::wcout.
    SetConsoleOutputCP(CP_UTF8);

    std::cout << "[BUILD] kintsugi_page_reservoir -- safety net, not Plan A\n" << std::flush;
    std::cout << "[ RESERVOIR] Target: " << (RESERVOIR_SIZE_BYTES / (1024.0 * 1024.0))
              << "MB, name=Local\\KintsugiLargePageReservoir\n" << std::flush;

    SetConsoleCtrlHandler(ConsoleCtrlHandler, TRUE);

    // --- 1. SEIZE REQUIRED PRIVILEGES ---
    // Modified to seize both memory locks and global namespace creation,
    // allowing direct execution from an elevated terminal without requiring SYSTEM.
    HANDLE hToken;
    bool got_lock_memory = false;
    bool got_create_global = false;

    if (OpenProcessToken(GetCurrentProcess(), TOKEN_ADJUST_PRIVILEGES | TOKEN_QUERY, &hToken)) {
        TOKEN_PRIVILEGES tp;
        tp.PrivilegeCount = 1;
        tp.Privileges[0].Attributes = SE_PRIVILEGE_ENABLED;

        // Seize Lock Memory (Required for Large Pages)
        if (LookupPrivilegeValue(NULL, SE_LOCK_MEMORY_NAME, &tp.Privileges[0].Luid)) {
            AdjustTokenPrivileges(hToken, FALSE, &tp, 0, NULL, 0);
            if (GetLastError() == ERROR_SUCCESS) {
                got_lock_memory = true;
                std::cout << "[KERNEL] Seized SE_LOCK_MEMORY privilege.\n" << std::flush;
            }
        }

        // Seize Create Global (Required for "Global\" namespace outside SYSTEM context)
        if (LookupPrivilegeValue(NULL, SE_CREATE_GLOBAL_NAME, &tp.Privileges[0].Luid)) {
            AdjustTokenPrivileges(hToken, FALSE, &tp, 0, NULL, 0);
            if (GetLastError() == ERROR_SUCCESS) {
                got_create_global = true;
                std::cout << "[KERNEL] Seized SE_CREATE_GLOBAL privilege.\n" << std::flush;
            }
        }
        CloseHandle(hToken);
    }

    if (!got_lock_memory) {
        std::cerr << "[FATAL] Could not seize SE_LOCK_MEMORY privilege. This program needs to "
                     "run elevated (Administrator) or as SYSTEM. Exiting.\n" << std::flush;
        return 1;
    }

    if (!got_create_global) {
        std::cout << "[WARN] Could not seize SE_CREATE_GLOBAL privilege. If probing fails with "
                     "Error Code 5, it is due to Global\\ namespace restrictions.\n" << std::flush;
    }

    // --- 2. GET LARGE PAGE MINIMUM ---
    SIZE_T large_page_min = GetLargePageMinimum();
    std::cout << "[ RESERVOIR] Large page minimum on this system: " << large_page_min << " bytes.\n" << std::flush;
    if (large_page_min == 0) {
        std::cerr << "[FATAL] GetLargePageMinimum returned 0 -- this system may not support "
                     "large pages at all.\n" << std::flush;
        return 1;
    }

    // --- 3. PROBE FOR THE LARGEST SIZE THIS MACHINE CAN ACTUALLY PROVIDE ---
    // A fixed 1536MB request turned "does this work" into a single yes/no
    // that doesn't distinguish "no large pages at all" from "1536MB
    // specifically is too big an ask, something smaller would work." This
    // tries decreasing sizes until one succeeds, turning that into a real
    // measurement instead of a guess -- confirmed necessary tonight when
    // 1536MB failed with 6273MB "available" on a genuinely fresh boot,
    // before anything else had launched.
    constexpr size_t CANDIDATE_SIZES_MB[] = { 1536, 1024, 768, 512, 256, 128, 64, 32, 2 };
    size_t chosen_size_bytes = 0;

    for (size_t candidate_mb : CANDIDATE_SIZES_MB) {
        size_t candidate_bytes = candidate_mb * 1024ULL * 1024ULL;
        if ((candidate_bytes % large_page_min) != 0) continue; // skip anything misaligned

        DWORD size_high = static_cast<DWORD>(candidate_bytes >> 32);
        DWORD size_low  = static_cast<DWORD>(candidate_bytes & 0xFFFFFFFFULL);

        HANDLE h = CreateFileMappingW(INVALID_HANDLE_VALUE, NULL,
                                       PAGE_READWRITE | SEC_COMMIT | SEC_LARGE_PAGES,
                                       size_high, size_low, RESERVOIR_NAME);
        if (!h) {
            DWORD err = GetLastError();
            std::cout << "[PROBE] " << candidate_mb << "MB -- CreateFileMappingW failed, Error Code: "
                      << err << ". Trying smaller.\n" << std::flush;
            continue;
        }

        bool already_existed = (GetLastError() == ERROR_ALREADY_EXISTS);
        void* v = MapViewOfFile(h, FILE_MAP_ALL_ACCESS | FILE_MAP_LARGE_PAGES, 0, 0, candidate_bytes);
        if (!v) {
            DWORD err = GetLastError();
            std::cout << "[PROBE] " << candidate_mb << "MB -- created but MapViewOfFile failed, "
                      << "Error Code: " << err << ". Trying smaller.\n" << std::flush;
            CloseHandle(h);
            continue;
        }

        // Success -- this is the largest size we're going to get. Stop probing.
        g_hMapping = h;
        g_pView = v;
        chosen_size_bytes = candidate_bytes;
        std::cout << "[PROBE] " << candidate_mb << "MB -- SUCCESS."
                  << (already_existed ? " (reservation with this name already existed -- opened it instead of creating new)" : "")
                  << "\n" << std::flush;
        break;
    }

    if (!g_pView) {
        std::cerr << "[FATAL] Every candidate size failed, down to the 2MB minimum. This machine "
                     "cannot provide ANY contiguous large-page memory right now, even at boot. "
                     "The reservoir approach may not be viable on this hardware -- worth knowing "
                     "precisely rather than continuing to guess.\n" << std::flush;
        return 1;
    }

    // --- 4. TOUCH IT ---
    // Large pages are documented to be fully committed at creation, not
    // lazily faulted — this touch is cheap insurance, not strictly required,
    // and gives a concrete timing number for the log rather than just
    // trusting the documentation blindly.
    auto touch_start = std::chrono::high_resolution_clock::now();
    memset(g_pView, 0, chosen_size_bytes);
    auto touch_end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> touch_dur = touch_end - touch_start;

    std::cout << "[OK] " << (chosen_size_bytes / (1024.0 * 1024.0))
              << "MB large-page reservation live and touched in " << touch_dur.count()
              << "s. Holding indefinitely -- do not close this window/service.\n" << std::flush;
    if (chosen_size_bytes < RESERVOIR_SIZE_BYTES) {
        std::cout << "[ RESERVOIR] NOTE: this is SMALLER than the engine's " << (RESERVOIR_SIZE_BYTES / (1024.0*1024.0))
                  << "MB arena. The engine currently requests a view matching its own arena size "
                  << "exactly, so it will NOT be able to use this smaller reservation as-is -- it will "
                  << "fall through to its own fallback path, same as if this program weren't running "
                  << "at all. This number is real, useful diagnostic data regardless: it's the actual "
                  << "ceiling this machine can provide right now, and the engine-side matching logic "
                  << "is the next thing to build once this number is known.\n" << std::flush;
    }
    std::cout << "[ RESERVOIR] Waiting. kintsugi_inference.exe will find this automatically "
                 "on its next wake.\n" << std::flush;

    // --- 5. HOLD IT, FOREVER, UNTIL EXPLICITLY STOPPED ---
    // The reservation is only alive as long as this handle+view stay open —
    // this is not a background task that can safely exit after setup.
    while (true) {
        Sleep(60000); // idle; ConsoleCtrlHandler handles clean shutdown
    }

    return 0; // unreachable, but keeps the compiler happy about a return path
}