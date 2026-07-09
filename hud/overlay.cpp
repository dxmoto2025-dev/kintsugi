#include <windows.h>
#include <d2d1.h>
#include <dwrite.h>
#include <string>
#include <iostream>
#include <immintrin.h>

#include "../core/shared_bus.h" // The SSOT

#pragma comment(lib, "d2d1.lib")
#pragma comment(lib, "dwrite.lib")

// Direct2D & DirectWrite Interfaces
ID2D1Factory* pD2DFactory = nullptr;
ID2D1HwndRenderTarget* pRenderTarget = nullptr;
ID2D1SolidColorBrush* pArmedBrush = nullptr;
ID2D1SolidColorBrush* pBlindBrush = nullptr;
IDWriteFactory* pDWriteFactory = nullptr;
IDWriteTextFormat* pTextFormat = nullptr;

// Bus Anchor
KintsugiBus* g_bus = nullptr;
HANDLE hMapFile = nullptr;

void InitBusMapping() {
    HANDLE hFile = CreateFileA("K:\\kintsugi\\core\\shared_bus.dat", GENERIC_READ, FILE_SHARE_READ | FILE_SHARE_WRITE, NULL, OPEN_EXISTING, FILE_ATTRIBUTE_NORMAL, NULL);
    if (hFile != INVALID_HANDLE_VALUE) {
        hMapFile = CreateFileMappingA(hFile, NULL, PAGE_READONLY, 0, BUS_STRUCT_SIZE, NULL);
        if (hMapFile) {
            g_bus = (KintsugiBus*)MapViewOfFile(hMapFile, FILE_MAP_READ, 0, 0, BUS_STRUCT_SIZE);
        }
        CloseHandle(hFile);
    }
}

// ------------------------------------------------------------------------
// [THE ANCHOR LINK]: Lock-free Seqlock Reader Loop
// ------------------------------------------------------------------------
KintsugiBus::LatestInput ReadSafeTelemetry() {
    KintsugiBus::LatestInput data = {0};
    if (!g_bus) return data;

    uint32_t seq1, seq2;
    do {
        seq1 = g_bus->seq_counter.load(std::memory_order_acquire);
        
        if (seq1 & 1) {
            _mm_pause(); // Yield pipeline while writer is mutating payload
            continue; 
        }

        data = g_bus->latest_input; // Atomic safe copy
        
        seq2 = g_bus->seq_counter.load(std::memory_order_acquire);
    } while (seq1 != seq2); // Loop if torn during read

    return data;
}

// ------------------------------------------------------------------------
// THE GHOST RENDERER: Hardware-accelerated Immediate Mode
// ------------------------------------------------------------------------
void RenderOverlay() {
    if (!pRenderTarget) return;

    pRenderTarget->BeginDraw();
    
    // Clear to absolute black. Because of LWA_COLORKEY, black becomes 100% transparent glass.
    pRenderTarget->Clear(D2D1::ColorF(0.0f, 0.0f, 0.0f, 0.0f));

    D2D1_RECT_F layoutRect = D2D1::RectF(50.0f, 50.0f, 1800.0f, 200.0f);
    std::wstring displayText;
    ID2D1SolidColorBrush* activeBrush = pBlindBrush;

    if (!g_bus) {
        displayText = L"[KINTSUGI] OFFLINE: SSOT Bus Disconnected.";
    } else if (g_bus->system_ready.load(std::memory_order_acquire) != 1) {
        displayText = L"[KINTSUGI] STANDBY: Awaiting Master Plane Ignition...";
    } else {
        // Pull State Islands
        uint32_t is_armed = g_bus->capture_armed.load(std::memory_order_acquire);
        uint32_t vel = g_bus->velocity_tps.load(std::memory_order_acquire);
        uint64_t beat = g_bus->heartbeat_tick.load(std::memory_order_acquire);
        
        // Pull Lock-Free Seqlock Stream
        KintsugiBus::LatestInput input = ReadSafeTelemetry();

        // Format Input Stream Data
        wchar_t inputStr[128];
        if (input.input_type == WM_MOUSEMOVE) {
            swprintf(inputStr, 128, L"MOUSE [%d, %d]", input.target_x, input.target_y);
        } else if (input.input_type == WM_KEYDOWN || input.input_type == WM_SYSKEYDOWN) {
            swprintf(inputStr, 128, L"KEY [0x%X]", input.vk_code);
        } else {
            swprintf(inputStr, 128, L"IDLE");
        }

        // Compose Final String based on Target Lock (Armed vs Blind)
        wchar_t buffer[512];
        if (is_armed) {
            swprintf(buffer, 512, L"[CORE] Beat: %llu | Vel: %u t/s   [EYES] ARMED | Last: %s", beat, vel, inputStr);
            activeBrush = pArmedBrush; // Terminal Green
        } else {
            swprintf(buffer, 512, L"[CORE] Beat: %llu | Vel: %u t/s   [EYES] BLIND | Context Lost", beat, vel);
            activeBrush = pBlindBrush; // Dark Gray
        }
        displayText = buffer;
    }

    // Draw the telemetry text to the Swapchain
    pRenderTarget->DrawTextW(
        displayText.c_str(),
        displayText.length(),
        pTextFormat,
        layoutRect,
        activeBrush
    );

    pRenderTarget->EndDraw();
}

// ------------------------------------------------------------------------
// WINDOW MESSAGE PUMP
// ------------------------------------------------------------------------
LRESULT CALLBACK WindowProc(HWND hwnd, UINT uMsg, WPARAM wParam, LPARAM lParam) {
    if (uMsg == WM_PAINT) {
        RenderOverlay();
        ValidateRect(hwnd, NULL);
        return 0;
    }
    if (uMsg == WM_DESTROY) {
        PostQuitMessage(0);
        return 0;
    }
    return DefWindowProc(hwnd, uMsg, wParam, lParam);
}

int WINAPI wWinMain(HINSTANCE hInstance, HINSTANCE hPrevInstance, PWSTR pCmdLine, int nCmdShow) {
    // 1. Initialize Direct2D and DirectWrite Factories
    D2D1CreateFactory(D2D1_FACTORY_TYPE_SINGLE_THREADED, &pD2DFactory);
    DWriteCreateFactory(DWRITE_FACTORY_TYPE_SHARED, __uuidof(IDWriteFactory), reinterpret_cast<IUnknown**>(&pDWriteFactory));

    // Anchor to the Shared Memory Bus
    InitBusMapping();

    // 2. Create the Ghost Window Class
    const wchar_t CLASS_NAME[] = L"KintsugiGhostHUD";
    WNDCLASS wc = { };
    wc.lpfnWndProc = WindowProc;
    wc.hInstance = hInstance;
    wc.lpszClassName = CLASS_NAME;
    RegisterClass(&wc);

    // 3. Spawn the Layered, Transparent, Click-Through Window
    int screenX = GetSystemMetrics(SM_CXSCREEN);
    int screenY = GetSystemMetrics(SM_CYSCREEN);
    
    HWND hwnd = CreateWindowEx(
        WS_EX_LAYERED | WS_EX_TRANSPARENT | WS_EX_TOPMOST | WS_EX_TOOLWINDOW,
        CLASS_NAME, L"Kintsugi HUD",
        WS_POPUP,
        0, 0, screenX, screenY,
        NULL, NULL, hInstance, NULL
    );

    // 4. Set absolute black as the transparency key
    SetLayeredWindowAttributes(hwnd, RGB(0, 0, 0), 0, LWA_COLORKEY);

    // 5. Initialize Hardware Render Target
    RECT rc;
    GetClientRect(hwnd, &rc);
    pD2DFactory->CreateHwndRenderTarget(
        D2D1::RenderTargetProperties(),
        D2D1::HwndRenderTargetProperties(hwnd, D2D1::SizeU(rc.right - rc.left, rc.bottom - rc.top)),
        &pRenderTarget
    );

    // 6. Setup typography and state colors
    pRenderTarget->CreateSolidColorBrush(D2D1::ColorF(0.0f, 1.0f, 0.3f, 1.0f), &pArmedBrush);
    pRenderTarget->CreateSolidColorBrush(D2D1::ColorF(0.4f, 0.4f, 0.4f, 1.0f), &pBlindBrush);
    
    pDWriteFactory->CreateTextFormat(
        L"Consolas", NULL, DWRITE_FONT_WEIGHT_BOLD, DWRITE_FONT_STYLE_NORMAL, 
        DWRITE_FONT_STRETCH_NORMAL, 24.0f, L"en-us", &pTextFormat
    );

    ShowWindow(hwnd, nCmdShow);

    // 7. Real-Time Message Loop
    MSG msg = { };
    while (true) {
        if (PeekMessage(&msg, NULL, 0, 0, PM_REMOVE)) {
            if (msg.message == WM_QUIT) break;
            TranslateMessage(&msg);
            DispatchMessage(&msg);
        } else {
            // Constantly repaint the ghost overlay at hardware speed
            RenderOverlay();
            Sleep(16); // Cap at ~60fps to prevent core 0 starvation
        }
    }

    // Cleanup
    if (g_bus) UnmapViewOfFile(g_bus);
    if (hMapFile) CloseHandle(hMapFile);
    if (pTextFormat) pTextFormat->Release();
    if (pArmedBrush) pArmedBrush->Release();
    if (pBlindBrush) pBlindBrush->Release();
    if (pRenderTarget) pRenderTarget->Release();
    if (pDWriteFactory) pDWriteFactory->Release();
    if (pD2DFactory) pD2DFactory->Release();
    
    return 0;
}