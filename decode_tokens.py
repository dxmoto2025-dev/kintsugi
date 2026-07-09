import struct

GGUF_PATH = r"K:\models\Meta-Llama-3-8B-Instruct.Q4_K_M.gguf"

def read_str(f):
    length = struct.unpack('<Q', f.read(8))[0]
    return f.read(length).decode('utf-8', errors='replace')

def skip_val(f, vtype):
    if vtype in (0,1,7): f.read(1)
    elif vtype in (2,3): f.read(2)
    elif vtype in (4,5,6): f.read(4)
    elif vtype == 8: read_str(f)
    elif vtype in (10,11,12): f.read(8)
    elif vtype == 9:
        et = struct.unpack('<I', f.read(4))[0]
        count = struct.unpack('<Q', f.read(8))[0]
        for _ in range(count): skip_val(f, et)

tokens = []
with open(GGUF_PATH, 'rb') as f:
    f.read(4); f.read(4)
    struct.unpack('<Q', f.read(8))[0]
    n_kv = struct.unpack('<Q', f.read(8))[0]
    for _ in range(n_kv):
        key   = read_str(f)
        vtype = struct.unpack('<I', f.read(4))[0]
        if key == 'tokenizer.ggml.tokens':
            et    = struct.unpack('<I', f.read(4))[0]
            count = struct.unpack('<Q', f.read(8))[0]
            for _ in range(count):
                tokens.append(read_str(f))
            break
        else:
            skip_val(f, vtype)

def show(label, ids):
    print("=" * 62)
    print(f"  {label}")
    print("=" * 62)
    words = []
    for tid in ids:
        text = tokens[tid] if tid < len(tokens) else f"<id {tid}>"
        words.append(text)
        print(f"  token_id={tid:6d}  -> {repr(text)}")
    sentence = "".join(w.replace("\u0120", " ") for i, w in zip(ids, words) if i != 128009)
    print(f"  ASSEMBLED: \"{sentence.strip()}\"")
    print()

show("ENCODED INPUT: \"what is 2 plus 2\"", [12840, 374, 220, 17, 5636, 220, 17])
show("FULL GENERATED RESPONSE", [791, 4320, 311, 220, 17, 5636, 220, 17, 374, 220, 19, 13, 128009])
