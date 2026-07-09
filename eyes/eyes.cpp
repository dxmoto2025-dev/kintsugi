#include <windows.h>
#include <iostream>
#include <atomic>
#include <string>

#include "../core/shared_bus.h" // The SSOT

KintsugiBus* g_bus = nullptr;
HHOOK hMouseHook = NULL;
HHOOK hKeyboardHook = NULL;
HWINEVENTHOOK hFocusHook = NULL;

const std::wstring TARGET_APP = L"Code.exe"; // The allowed process

// ------------------------------------------------------------------------
// THE FOCUS GATE (EVENT_SYSTEM_FOREGROUND)
// ------------------------------------------------------------------------
void CALLBACK WinEventProc(HWINEVENTHOOK hWinEventHook, DWORD event, HWND hwnd, LONG idObject, LONG idChild, DWORD dwEventThread, DWORD dwmsEventTime) {
    if (hwnd && g_bus) {
        DWORD pid;
        GetWindowThreadProcessId(hwnd, &pid);
        HANDLE hProcess = OpenProcess(PROCESS_QUERY_INFORMATION | PROCESS_VM_READ, FALSE, pid);
        if (hProcess) {
            wchar_t processName[MAX_PATH];
            DWORD size = MAX_PATH;
            if (QueryFullProcessImageNameW(hProcess, 0, processName, &size)) {
                std::wstring name(processName);
                bool isTarget = (name.find(TARGET_APP) != std::wstring::npos);
                g_bus->capture_armed.store(isTarget ? 1 : 0, std::memory_order_release);
            }
            CloseHandle(hProcess);
        }
    }
}

// ------------------------------------------------------------------------
// THE SEQLOCK TELEMETRY FEED (For the Ghost HUD)
// ------------------------------------------------------------------------
void WriteTelemetryToBus(UINT message, int x, int y, DWORD vkCode) {
    if (!g_bus || g_bus->capture_armed.load(std::memory_order_acquire) == 0) return;

    // 1. Lock (Make Odd)
    uint32_t seq = g_bus->seq_counter.load(std::memory_order_relaxed);
    g_bus->seq_counter.store(seq + 1, std::memory_order_release);

    // 2. Write Payload
    g_bus->latest_input.input_type = message;
    g_bus->latest_input.target_x = x;
    g_bus->latest_input.target_y = y;
    g_bus->latest_input.vk_code = vkCode;

    // 3. Unlock (Make Even)
    g_bus->seq_counter.store(seq + 2, std::memory_order_release);
}

LRESULT CALLBACK LowLevelMouseProc(int nCode, WPARAM wParam, LPARAM lParam) {
    if (nCode == HC_ACTION) {
        MSLLHOOKSTRUCT* mouse = reinterpret_cast<MSLLHOOKSTRUCT*>(lParam);
        WriteTelemetryToBus((UINT)wParam, mouse->pt.x, mouse->pt.y, 0);
    }
    return CallNextHookEx(hMouseHook, nCode, wParam, lParam);
}

LRESULT CALLBACK LowLevelKeyboardProc(int nCode, WPARAM wParam, LPARAM lParam) {
    if (nCode == HC_ACTION) {
        KBDLLHOOKSTRUCT* kb = reinterpret_cast<KBDLLHOOKSTRUCT*>(lParam);
        WriteTelemetryToBus((UINT)wParam, 0, 0, kb->vkCode);
    }
    return CallNextHookEx(hKeyboardHook, nCode, wParam, lParam);
}

int main() {
    std::cout << "======================================================================\n";
    std::cout << "          KINTSUGI SOVEREIGN TELEMETRY OBSERVER (EYES)               \n";
    std::cout << "======================================================================\n";

    HANDLE hFile = CreateFileA("K:\\kintsugi\\core\\shared_bus.dat", GENERIC_READ | GENERIC_WRITE, FILE_SHARE_READ | FILE_SHARE_WRITE, NULL, OPEN_EXISTING, FILE_ATTRIBUTE_NORMAL, NULL);
    if (hFile != INVALID_HANDLE_VALUE) {
        HANDLE hMapFile = CreateFileMappingA(hFile, NULL, PAGE_READWRITE, 0, BUS_STRUCT_SIZE, NULL);
        g_bus = (KintsugiBus*)MapViewOfFile(hMapFile, FILE_MAP_ALL_ACCESS, 0, 0, BUS_STRUCT_SIZE);
    }

    if (!g_bus) { std::cout << "[❌ FATAL] Could not map to SSOT!\n"; return 1; }

    std::cout << "[⚡ CONNECTED] Eyes mapped to SSOT Bus. Awaiting Master Plane...\n";
    while (g_bus->system_ready.load(std::memory_order_acquire) != 1) { Sleep(1); }
    std::cout << "[IGNITION] Master Plane Ready. Arming Observer...\n";

    hFocusHook = SetWinEventHook(EVENT_SYSTEM_FOREGROUND, EVENT_SYSTEM_FOREGROUND, NULL, WinEventProc, 0, 0, WINEVENT_OUTOFCONTEXT);
    hMouseHook = SetWindowsHookEx(WH_MOUSE_LL, LowLevelMouseProc, GetModuleHandle(NULL), 0);
    hKeyboardHook = SetWindowsHookEx(WH_KEYBOARD_LL, LowLevelKeyboardProc, GetModuleHandle(NULL), 0);

    MSG msg;
    while (GetMessage(&msg, NULL, 0, 0)) { TranslateMessage(&msg); DispatchMessage(&msg); }

    return 0;
}