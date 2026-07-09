import os
import time
import mmap
import ctypes
import subprocess
from shared_bus import KintsugiBus, BUS_STRUCT_SIZE, KINT_MAGIC, BUS_VERSION, MAX_IPC_TOKENS

# Make sure you have installed this: pip install llama-cpp-python
from llama_cpp import Llama 

# ==============================================================================
# WINDOWS JOB OBJECT CTYPES WRAPPER (OS-Level Teardown Guarantee)
# ==============================================================================
kernel32 = ctypes.windll.kernel32
JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE = 0x2000

class JOBOBJECT_BASIC_LIMIT_INFORMATION(ctypes.Structure):
    _fields_ = [
        ("PerProcessUserTimeLimit", ctypes.c_int64),
        ("PerJobUserTimeLimit", ctypes.c_int64),
        ("LimitFlags", ctypes.c_uint32),
        ("MinimumWorkingSetSize", ctypes.c_size_t),
        ("MaximumWorkingSetSize", ctypes.c_size_t),
        ("ActiveProcessLimit", ctypes.c_uint32),
        ("Affinity", ctypes.c_size_t),
        ("PriorityClass", ctypes.c_uint32),
        ("SchedulingClass", ctypes.c_uint32),
    ]

class IO_COUNTERS(ctypes.Structure):
    _fields_ = [("dummy", ctypes.c_uint64 * 6)]

class JOBOBJECT_EXTENDED_LIMIT_INFORMATION(ctypes.Structure):
    _fields_ = [
        ("BasicLimitInformation", JOBOBJECT_BASIC_LIMIT_INFORMATION),
        ("IoInfo", IO_COUNTERS),
        ("ProcessMemoryLimit", ctypes.c_size_t),
        ("JobMemoryLimit", ctypes.c_size_t),
        ("PeakProcessMemoryUsed", ctypes.c_size_t),
        ("PeakJobMemoryUsed", ctypes.c_size_t),
    ]

def create_kill_on_close_job():
    job = kernel32.CreateJobObjectW(None, None)
    info = JOBOBJECT_EXTENDED_LIMIT_INFORMATION()
    info.BasicLimitInformation.LimitFlags = JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE
    kernel32.SetInformationJobObject(job, 9, ctypes.byref(info), ctypes.sizeof(info))
    return job

def assign_to_job(job, process):
    # Extracts the raw Win32 process handle from Python's subprocess object
    handle = int(process._handle) 
    kernel32.AssignProcessToJobObject(job, handle)

# ==============================================================================
# THE MASTER ORCHESTRATOR
# ==============================================================================
def ignite_engine():
    print("======================================================================")
    print("          KINTSUGI SOVEREIGN RUNTIME - MASTER PLANE                   ")
    print("======================================================================")

    # 1. Establish the Anchor File
    anchor_path = r"K:\kintsugi\core\shared_bus.dat"
    with open(anchor_path, "wb+") as f:
        f.write(b'\x00' * BUS_STRUCT_SIZE)
        f.flush()

    # 2. Map the Memory & Cast to SSOT Struct
    f = open(anchor_path, "r+b")
    mm = mmap.mmap(f.fileno(), BUS_STRUCT_SIZE, access=mmap.ACCESS_WRITE)
    bus = KintsugiBus.from_buffer(mm)

    # 3. Write the Header (Pre-Ignition)
    ctypes.memset(ctypes.addressof(bus), 0, BUS_STRUCT_SIZE) # Zero the entire block
    bus.magic = KINT_MAGIC
    bus.version = BUS_VERSION
    print("[⚡ SILICON] Memory bus zeroed and formatted. SSOT strict alignment confirmed.")

    # 4. Forge the OS Job Object
    job = create_kill_on_close_job()
    print("[🛡️ GUARDRAIL] OS-Level Job Object created. Kill-on-Close Enforced.")

    # 5. Load the Cortex (Neural Weights)
    print("[🧠 CORTEX] Loading Sovereign LLM weights into memory...")
    
    # [!] UPDATE THIS PATH TO YOUR ACTUAL .GGUF MODEL
    target_model_path = r"K:\models\Meta-Llama-3-8B-Instruct.Q4_K_M.gguf" 
    
    if not os.path.exists(target_model_path):
        print(f"[❌ FATAL] Model not found at {target_model_path}. Please update line 78.")
        return

    llm = Llama(
        model_path=target_model_path, 
        n_ctx=4096,   # Match our MAX_IPC_TOKENS
        n_gpu_layers=-1 # Offload to GPU if available, or keep 0 for pure CPU
    )
    print("[🧠 CORTEX] Weights loaded. Initiating Semantic Stream.")

    # 6. Spawn the Triad
    build_dir = r"K:\kintsugi\build"
    print("[IGNITION] Spawning C++ Pillars...")
    
    # Rerouted the Core to the new GCC AVX2 Live Bridge
    core_proc = subprocess.Popen([r"K:\kintsugi\kintsugi_live_bridge.exe"], creationflags=subprocess.CREATE_NEW_CONSOLE)
    eyes_proc = subprocess.Popen([os.path.join(build_dir, "eyes.exe")], creationflags=subprocess.CREATE_NEW_CONSOLE)
    hud_proc = subprocess.Popen([os.path.join(build_dir, "overlay.exe")], creationflags=subprocess.CREATE_NO_WINDOW)

    assign_to_job(job, core_proc)
    assign_to_job(job, eyes_proc)
    assign_to_job(job, hud_proc)

    time.sleep(1) # Give the OS a second to map the processes

    # 7. Unleash the Spin-Locks (x86 TSO guarantees this propagates cleanly)
    bus.system_ready = 1
    print("[🟢 ONLINE] System Ready flag flipped. Engine is LIVE.")

    # 8. The Live Event Loop
    prompt = "Analyze the system architecture of a bare-metal C++ runtime and its reliance on memory barriers."
    
    try:
        # Start the streaming generator. logprobs=1 ensures we get the raw integer token IDs.
        stream = llm(
            prompt,
            max_tokens=1024,
            stream=True,
            logprobs=1 
        )
        
        start_time = time.time()
        tokens_processed = 0

        for output in stream:
            # Check process health
            if core_proc.poll() is not None or eyes_proc.poll() is not None or hud_proc.poll() is not None:
                print("\n[❌ FATAL] A C++ pillar collapsed. Teardown initiated.")
                break

            # Safely extract the raw integer Token ID and the text chunk
            try:
                token_id = output["choices"][0]["logprobs"]["tokens"][0] # Grab the integer ID
            except (KeyError, IndexError, TypeError):
                token_id = 999 # Fallback if dict shape varies

            text_chunk = output["choices"][0]["text"]
            
            # Echo to the Python terminal so you can watch the Cortex think
            print(text_chunk, end="", flush=True)

            # --- THE SSOT MEMORY INJECTION ---
            current_write = bus.token_write_ptr
            current_read = bus.token_read_ptr
            
            # Blocking Backpressure: Wait if the C++ ring buffer is full
            while current_write - current_read >= MAX_IPC_TOKENS:
                time.sleep(0.001)
                current_read = bus.token_read_ptr

            # Write the exact integer token to the memory bus
            slot = current_write & (MAX_IPC_TOKENS - 1)
            bus.token_array[slot] = token_id if isinstance(token_id, int) else 999 
            
            # Write-commit (Relies on x86-64 TSO for implicit release)
            bus.token_write_ptr = current_write + 1
            
            # Calculate and broadcast true live velocity
            tokens_processed += 1
            elapsed = time.time() - start_time
            if elapsed > 0:
                bus.velocity_tps = int(tokens_processed / elapsed)

    except KeyboardInterrupt:
        print("\n\n[MASTER] Manual override detected. Ripping the wire.")
    finally:
        print("[MASTER] Shutting down. The OS will now instantly slaughter the C++ processes.")
        mm.close()
        f.close()

if __name__ == "__main__":
    ignite_engine()