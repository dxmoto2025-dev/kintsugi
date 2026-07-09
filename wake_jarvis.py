"""
wake_jarvis.py — Flips the SiliconGatePortal enable_programming_pin to wake
the sleeping kintsugi_inference.exe engine, waits for it to finish, and
reads back the predicted token.

STATUS UPDATE (was stale — corrected after tonight's runs):
text_payload below IS actually consumed now. The real BPE tokenizer went
in earlier tonight — every wake's [ENCODER] log line shows this exact
string getting tokenized and run through the model, not a hardcoded
fixed prompt. "Send Jarvis any prompt you want" already works; this note
just hadn't been updated to say so.

Run this from K:\\kintsugi\\ while kintsugi_inference.exe is already
running and asleep at "Awaiting enable_programming_pin...".

Usage:
    python wake_jarvis.py
"""
import mmap
import time
import sys
import csv
import ctypes
import subprocess
from ctypes import wintypes
from shared_bus import SiliconGatePortal, BUS_STRUCT_SIZE, KINT_MAGIC, BUS_VERSION

BUS_PATH = r"K:\kintsugi\core\shared_bus.dat"
TEST_PROMPT = "a farmer has 17 sheep, all but 9 die, how many are left?"
POLL_INTERVAL_SEC = 0.5
# The engine no longer stops after one prefill token — it now runs up to
# NUM_DECODE_STEPS=40 additional full 32-layer decode passes internally
# per wake (kintsugi_inference.cpp), reloading+dequantizing weights from
# disk every step. Observed ~41-43s/step on this machine, not the ~9-10s
# the C++ side's own comment optimistically assumed, so budget for the
# worst case (prefill + 40 steps @ ~45s) plus real margin, not the old
# single-token-wake figure of 90-100s.
TIMEOUT_SEC = 2400  # 40 minutes
ENGINE_EXE_NAME = "kintsugi_inference.exe"


# --- Memory monitor: samples the already-running engine process + system-wide
# memory load on every poll tick below, since the engine is a persistent
# sleep/wake loop rather than something this script launches. ---

class MEMORYSTATUSEX(ctypes.Structure):
    _fields_ = [
        ("dwLength", wintypes.DWORD),
        ("dwMemoryLoad", wintypes.DWORD),
        ("ullTotalPhys", ctypes.c_uint64),
        ("ullAvailPhys", ctypes.c_uint64),
        ("ullTotalPageFile", ctypes.c_uint64),
        ("ullAvailPageFile", ctypes.c_uint64),
        ("ullTotalVirtual", ctypes.c_uint64),
        ("ullAvailVirtual", ctypes.c_uint64),
        ("ullAvailExtendedVirtual", ctypes.c_uint64),
    ]


class PROCESS_MEMORY_COUNTERS_EX(ctypes.Structure):
    _fields_ = [
        ("cb", wintypes.DWORD),
        ("PageFaultCount", wintypes.DWORD),
        ("PeakWorkingSetSize", ctypes.c_size_t),
        ("WorkingSetSize", ctypes.c_size_t),
        ("QuotaPeakPagedPoolUsage", ctypes.c_size_t),
        ("QuotaPagedPoolUsage", ctypes.c_size_t),
        ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t),
        ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
        ("PagefileUsage", ctypes.c_size_t),
        ("PeakPagefileUsage", ctypes.c_size_t),
        ("PrivateUsage", ctypes.c_size_t),
    ]


PROCESS_QUERY_INFORMATION = 0x0400
PROCESS_VM_READ = 0x0010


def get_system_memory_load_pct():
    """Same number Task Manager shows as overall memory usage."""
    stat = MEMORYSTATUSEX()
    stat.dwLength = ctypes.sizeof(MEMORYSTATUSEX)
    ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(stat))
    return stat.dwMemoryLoad


def find_pid_by_name(exe_name):
    try:
        out = subprocess.check_output(
            ["tasklist", "/FI", f"IMAGENAME eq {exe_name}", "/FO", "CSV", "/NH"],
            text=True,
        )
    except subprocess.CalledProcessError:
        return None
    line = out.strip().splitlines()[0] if out.strip() else ""
    if not line or "No tasks" in line:
        return None
    fields = next(csv.reader([line]))
    return int(fields[1])


def open_process_for_query(pid):
    handle = ctypes.windll.kernel32.OpenProcess(
        PROCESS_QUERY_INFORMATION | PROCESS_VM_READ, False, pid
    )
    return handle if handle else None


def get_process_memory_mb(handle):
    counters = PROCESS_MEMORY_COUNTERS_EX()
    counters.cb = ctypes.sizeof(PROCESS_MEMORY_COUNTERS_EX)
    ok = ctypes.windll.psapi.GetProcessMemoryInfo(
        handle, ctypes.byref(counters), counters.cb
    )
    if not ok:
        return None
    mb = 1024 * 1024
    return (
        counters.WorkingSetSize / mb,
        counters.PrivateUsage / mb,
        counters.PeakWorkingSetSize / mb,
    )


def main():
    try:
        f = open(BUS_PATH, "r+b")
    except FileNotFoundError:
        print(f"[❌ FATAL] {BUS_PATH} does not exist on this machine.")
        print("The engine's own log showed '[✓ PORTAL] Mapped.', which means this")
        print("file DOES exist wherever kintsugi_inference.exe is actually running.")
        print("Run this script on that same machine.")
        sys.exit(1)

    file_size = f.seek(0, 2)
    f.seek(0)
    print(f"[i] {BUS_PATH} is {file_size} bytes on disk (portal needs >= {BUS_STRUCT_SIZE}).")

    if file_size < BUS_STRUCT_SIZE:
        print(f"[❌ FATAL] File is smaller than the portal struct ({BUS_STRUCT_SIZE} bytes).")
        print("The running engine couldn't have mapped this successfully either —")
        print("not touching the file size automatically. Investigate before proceeding.")
        f.close()
        sys.exit(1)

    mm = mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_WRITE)
    portal = SiliconGatePortal.from_buffer(mm)

    print("\n[i] Current portal state BEFORE wake:")
    print(f"    magic   = 0x{portal.magic:08X}  (expected 0x{KINT_MAGIC:08X})")
    print(f"    version = {portal.version}  (expected {BUS_VERSION})")
    print(f"    enable_programming_pin = {portal.enable_programming_pin}")
    print(f"    token_id_payload (stale) = {portal.token_id_payload}")

    if portal.enable_programming_pin != 0:
        print("\n[!] WARNING: pin is already non-zero. Either the engine is already")
        print("    mid-inference from something else, or this is stale data.")
        print("    Proceeding to just wait for it to return to 0.")
    else:
        # Harmless bookkeeping — the C++ side never checks magic/version, this is
        # just for clean diagnostics on future reads.
        portal.magic = KINT_MAGIC
        portal.version = BUS_VERSION
        portal.text_payload = TEST_PROMPT.encode("utf-8")

        print(f"\n[⚡ HUD] Writing text_payload = \"{TEST_PROMPT}\" "
              f"(not yet consumed by the engine — see note above)")
        print("[⚡ HUD] Flipping enable_programming_pin -> 1. Waking Jarvis...\n")
        portal.enable_programming_pin = 1

    mem_log_path = f"mem_trace_wake_{time.strftime('%Y%m%d_%H%M%S')}.csv"
    mem_log = open(mem_log_path, "w", newline="")
    mem_writer = csv.writer(mem_log)
    mem_writer.writerow(["elapsed_s", "proc_working_set_mb", "proc_private_mb",
                          "proc_peak_working_set_mb", "sys_mem_load_pct"])

    engine_pid = find_pid_by_name(ENGINE_EXE_NAME)
    engine_handle = open_process_for_query(engine_pid) if engine_pid else None
    if engine_handle:
        print(f"[monitor] Tracking {ENGINE_EXE_NAME} PID={engine_pid} -> {mem_log_path}")
    else:
        print(f"[monitor] {ENGINE_EXE_NAME} not found via tasklist — "
              f"logging system-wide memory load only -> {mem_log_path}")

    peak_ws_mb, peak_ws_at, peak_sys_pct = 0.0, 0.0, 0

    def close_monitor():
        if engine_handle:
            ctypes.windll.kernel32.CloseHandle(engine_handle)
        mem_log.close()

    start = time.time()
    tick = 0
    while portal.enable_programming_pin != 0:
        elapsed = time.time() - start
        if elapsed > TIMEOUT_SEC:
            print(f"\n[!] Timed out after {TIMEOUT_SEC}s waiting for pin to drop.")
            print("    Check the C++ terminal directly — prior full prefill runs took")
            print("    90-100+ seconds, so this might genuinely still be working.")
            close_monitor()
            del portal  # drop the ctypes buffer export before closing the mmap
            mm.close()
            f.close()
            sys.exit(1)
        time.sleep(POLL_INTERVAL_SEC)
        tick += 1

        sys_pct = get_system_memory_load_pct()
        ws_mb, priv_mb, pk_mb = "", "", ""
        if engine_handle:
            sample = get_process_memory_mb(engine_handle)
            if sample:
                ws_mb, priv_mb, pk_mb = sample
                if ws_mb > peak_ws_mb:
                    peak_ws_mb, peak_ws_at = ws_mb, elapsed
        peak_sys_pct = max(peak_sys_pct, sys_pct)

        mem_writer.writerow([round(elapsed, 2), ws_mb, priv_mb, pk_mb, sys_pct])
        mem_log.flush()

        print(".", end="", flush=True)
        if tick % 60 == 0:
            print(f"  [{int(elapsed)}s elapsed, sys_mem={sys_pct}%]")

    elapsed = time.time() - start
    print(f"\n\n[✓] Pin returned to 0 after {elapsed:.1f}s. Jarvis finished and slept again.")
    result_token = portal.token_id_payload  # capture the value before dropping portal
    print(f"[🧠 RESULT] token_id_payload = {result_token}")
    print(f"[i] Add {result_token} to decode_tokens.py's TOKEN_IDS and run it.")

    close_monitor()
    print(f"[monitor] Memory trace written to {mem_log_path}")
    if engine_handle:
        print(f"[monitor] Peak engine working set: {peak_ws_mb:.1f} MB at t={peak_ws_at:.1f}s")
    print(f"[monitor] Peak system memory load during this wake: {peak_sys_pct}%")

    del portal  # drop the ctypes buffer export before closing the mmap
    mm.close()
    f.close()


if __name__ == "__main__":
    main()
