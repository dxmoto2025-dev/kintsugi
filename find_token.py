"""
find_token.py — Looks up exact token IDs by matching strings against the
GGUF's real tokenizer.ggml.tokens vocabulary. Same source decode_tokens.py
reads, just searched instead of indexed — verified IDs, not guessed from
memory.

Run: python find_token.py
"""
import struct

GGUF_PATH = r"K:\models\Meta-Llama-3-8B-Instruct.Q4_K_M.gguf"

# Everything needed to properly close the user turn and open the assistant
# turn, completing the Llama-3-Instruct chat template.
SEARCH_FOR = [
    "<|eot_id|>",
    "assistant",
    "<|begin_of_text|>",
    "<|start_header_id|>",
    "<|end_header_id|>",
    "user",
]


def read_str(f):
    length = struct.unpack('<Q', f.read(8))[0]
    return f.read(length).decode('utf-8', errors='replace')


def skip_val(f, vtype):
    if vtype in (0, 1, 7): f.read(1)
    elif vtype in (2, 3): f.read(2)
    elif vtype in (4, 5, 6): f.read(4)
    elif vtype == 8: read_str(f)
    elif vtype in (10, 11, 12): f.read(8)
    elif vtype == 9:
        et = struct.unpack('<I', f.read(4))[0]
        count = struct.unpack('<Q', f.read(8))[0]
        for _ in range(count): skip_val(f, et)


tokens = []
with open(GGUF_PATH, 'rb') as f:
    f.read(4)
    f.read(4)
    struct.unpack('<Q', f.read(8))[0]
    n_kv = struct.unpack('<Q', f.read(8))[0]
    for _ in range(n_kv):
        key = read_str(f)
        vtype = struct.unpack('<I', f.read(4))[0]
        if key == 'tokenizer.ggml.tokens':
            et = struct.unpack('<I', f.read(4))[0]
            count = struct.unpack('<Q', f.read(8))[0]
            for _ in range(count):
                tokens.append(read_str(f))
            break
        else:
            skip_val(f, vtype)

print("=" * 62)
print(f"  Loaded {len(tokens)} tokens from GGUF vocabulary")
print("=" * 62)

for target in SEARCH_FOR:
    matches = [i for i, t in enumerate(tokens) if t == target]
    if matches:
        print(f"  \"{target}\"  ->  token_id = {matches[0]}"
              + (f"  (WARNING: {len(matches)} matches, showing first)" if len(matches) > 1 else ""))
    else:
        # Fall back to substring search in case of BPE-prefix variants (e.g. "Ġassistant")
        partial = [(i, t) for i, t in enumerate(tokens) if target.strip("<|>").lower() in t.lower()][:5]
        print(f"  \"{target}\"  ->  NO EXACT MATCH. Closest candidates:")
        for i, t in partial:
            print(f"      id={i}  ->  {repr(t)}")

print("=" * 62)
