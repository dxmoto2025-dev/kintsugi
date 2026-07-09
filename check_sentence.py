"""
check_sentence.py — NOT a general tokenizer. Checks whether each word in a
candidate test sentence exists as a single clean token in the real GGUF
vocabulary (checking both the bare word and the Ġ-prefixed "preceded by a
space" form, since decode_tokens.py already showed the vocab uses that
convention: 'Ġdown', 'Ġnotified', etc.).

This exists so we can pick a REAL test sentence with verified token IDs
instead of guessing how Llama-3's actual BPE tokenizer would split it.
Words that DON'T resolve to a single clean token get flagged so we pick
a different word rather than silently guessing wrong.

Run: python check_sentence.py
"""
import struct

GGUF_PATH = r"K:\models\Meta-Llama-3-8B-Instruct.Q4_K_M.gguf"

# Candidate test sentence — first word has no leading space (nothing
# precedes it), every other word is checked with the Ġ (space) prefix.
CANDIDATE_WORDS = ["What", "is", "the", "capital", "of", "France"]


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

# Build a reverse lookup: token string -> id (first occurrence)
lookup = {}
for i, t in enumerate(tokens):
    if t not in lookup:
        lookup[t] = i

print("=" * 62)
print(f"  Checking candidate sentence: \"{' '.join(CANDIDATE_WORDS)}\"")
print("=" * 62)

resolved_ids = []
all_clean = True

for idx, word in enumerate(CANDIDATE_WORDS):
    # First word: check bare form only (nothing precedes it).
    # Every subsequent word: check the space-prefixed Ġ form first.
    candidates = [word] if idx == 0 else [f"\u0120{word}", word]
    found = None
    for c in candidates:
        if c in lookup:
            found = c
            break
    if found:
        tid = lookup[found]
        resolved_ids.append(tid)
        marker = "Ġ" if found.startswith("\u0120") else "(bare)"
        print(f"  \"{word}\"  ->  token_id={tid}  [{marker}]")
    else:
        all_clean = False
        resolved_ids.append(None)
        print(f"  \"{word}\"  ->  NO CLEAN MATCH (would need multi-token split)")

print("=" * 62)
if all_clean:
    print(f"  ALL WORDS CLEAN. Token sequence: {resolved_ids}")
else:
    print("  Some words did NOT resolve cleanly — pick a different word or sentence.")
print("=" * 62)
