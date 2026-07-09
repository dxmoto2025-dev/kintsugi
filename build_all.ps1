# K:\kintsugi\build_all.ps1

Write-Host "======================================================================" -ForegroundColor Cyan
Write-Host "          KINTSUGI SOVEREIGN RUNTIME - MASTER COMPILER                " -ForegroundColor Cyan
Write-Host "======================================================================" -ForegroundColor Cyan

$buildDir = "K:\kintsugi\build"

# Ensure build directory exists
if (-not (Test-Path $buildDir)) {
    New-Item -ItemType Directory -Path $buildDir | Out-Null
}

# 1. Compile Core (The Brain)
# - /O2: Maximize speed
# - /arch:AVX2: Unlock Advanced Vector Extensions for matrix math
# - /EHsc: Standard C++ exception handling
Write-Host "`n[COMPILING] kintsugi_70b.exe (AVX2 Optimized)..." -ForegroundColor Yellow
cl.exe /nologo /O2 /arch:AVX2 /EHsc K:\kintsugi\core\core.cpp /Fe:"$buildDir\kintsugi_70b.exe"

# 2. Compile Eyes (The Observer)
# - Links user32.lib for the SetWindowsHookEx API
Write-Host "`n[COMPILING] eyes.exe (Win32 Hooks)..." -ForegroundColor Yellow
cl.exe /nologo /O2 /EHsc K:\kintsugi\eyes\eyes.cpp user32.lib /Fe:"$buildDir\eyes.exe"

# 3. Compile HUD (The Ghost)
# - Links user32.lib, d2d1.lib, and dwrite.lib for the Direct2D rendering loop
# - /D UNICODE /D _UNICODE: Forces modern wide-character Win32 APIs to fix C2440
Write-Host "`n[COMPILING] overlay.exe (Direct2D Swapchain)..." -ForegroundColor Yellow
cl.exe /nologo /O2 /EHsc /D UNICODE /D _UNICODE K:\kintsugi\hud\overlay.cpp user32.lib d2d1.lib dwrite.lib /Fe:"$buildDir\overlay.exe"

# Cleanup temporary object files from the root directory to keep the silo pristine
Write-Host "`n[CLEANUP] Removing temporary object files..." -ForegroundColor DarkGray
Remove-Item *.obj -ErrorAction SilentlyContinue

Write-Host "`n======================================================================" -ForegroundColor Green
Write-Host " [SUCCESS] All Sovereign binaries minted to K:\kintsugi\build\" -ForegroundColor Green
Write-Host "======================================================================" -ForegroundColor Green