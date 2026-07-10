# Kintsugi / Jarvis — Project Context

Bare-metal C++ LLM inference engine. Llama-3-8B-Instruct, raw GGUF, Windows,
MinGW/g++. No frameworks, no ggml/llama.cpp linked in — every kernel
(dequant, attention, RoPE, tokenizer) is hand-built from first principles
and checked against real reference source, not reimplemented from memory.

Collaborators: Hamdan (relays + tests on real hardware), Kintsugi (a
separate terminal-only AI instance, blind to source, generates hypotheses),
Claude (reads actual code, verifies claims against ground truth). If you're
reading this as a new Claude Code session: you're now the one with direct
file access none of us had before — use it to verify, not just to edit.

## Read this before anything else: the actual operating discipline

This project's real asset isn't any single fix — it's a standing rule that
produced all of them: **a claim about code is not evidence until the actual
code is read.** Applies to logs, to prior-session summaries, to Kintsugi's
hypotheses, to your own assumptions. Concretely:
- A terminal log saying something succeeded doesn't confirm *why* — grep
  the actual source for the exact log string before trusting it matched
  the code you think ran. This caught a real stale-binary/version mismatch
  more than once.
- Before using an unfamiliar Windows API flag combination, check real docs.
  This caught a genuine silent-failure bug (see gotchas below).
- When a run "improves," compute the real delta against the last comparable
  baseline before calling it a win. Several apparent wins turned out to be
  within normal run-to-run noise once actually calculated.
- Structural verification (brace-balance checks, grep for old code fully
  removed, ordering checks for dependencies) happens on every edit, not
  just at the end.

## What's built and verified (the proven core)

- **GGUF parser** — real metadata (rope_freq_base, rms_norm_eps pulled from
  the file, not hardcoded), full tensor index, `tokenizer.ggml.tokens` and
  `tokenizer.ggml.merges` genuinely extracted (128,256 vocab, 280,147 merges).
- **Q4_K / Q6_K dequant** — verified byte-for-byte against real ggml-quants.c.
  One real bug found and fixed: Q6_K was missing `is = l/16` in its scale
  index, silently corrupting every Q6_K tensor including the LM head itself.
- **GQA attention, RoPE, SwiGLU FFN** — verified against llama.cpp's actual
  conventions, not textbook assumptions.
- **Per-layer persistent KV caches** — one independent cache per transformer
  layer. Earlier design used one shared cache reset per-layer, which caused
  real cross-layer attention contamination; now structurally impossible.
- **mmap weight vault** — `MapViewOfFile` over the GGUF, real ~5x decode
  speedup over raw file streaming (OS page cache serves repeat touches).
- **Full BPE tokenizer, built from scratch, all self-tested at every boot:**
  UTF-8 decoder, codepoint classifier (37 boundary cases), the real 7-branch
  LLAMA3 pretokenizer regex (verified against llama.cpp's actual
  `unicode_regex_split_custom_llama3`), byte-to-unicode table (verified
  against real GPT-2 source — byte 32 correctly resolves to U+0120, the Ġ
  seen in every decode this project has ever produced), the merge loop
  (lowest-rank-wins, merge-all-occurrences — both independently Python-
  verified before being written in C++), native detokenizer.
- **Real sampling** — temperature + top-p, replacing pure argmax. Near-zero
  temperature is a hard special case that collapses to exact argmax, used
  as the regression check against every proven-correct greedy result.
- **Checkpoint 2** — watermark-gated `SetProcessWorkingSetSize` trim at the
  prefill→decode boundary. Only fires above a real measured threshold
  (currently 3072MB), caps toward that ceiling rather than flooring to
  zero. This is the proven, load-bearing memory mechanism — holds a flat
  ~9.4–10s/step decode under real pressure, repeatedly, across dozens of runs.
- **LM-head buffer materialization** — `output.weight` (~411MB) copied once
  into a dedicated arena buffer at boot; every sweep reads it directly
  instead of depending on mmap page-cache warmth. Residency proven via
  direct `QueryWorkingSetEx` checks, not inferred from timing.

**Known-good regression prompts** (use these to confirm a fresh session's
changes haven't broken anything real):
- "What is the capital of France" → "The capital of France is Paris."
- "what is 2 plus 2" → "The answer to 2 plus 2 is 4."
- "what is the day after monday?" → "...the day after Monday is Tuesday!"

## What's a safety net, not the core — don't over-index on it

`kintsugi_page_reservoir.exe` pre-reserves large-page memory at boot, before
fragmentation sets in, for the engine to optionally use first. It is
**additive** — if it's not running, the engine falls straight through to
its own proven fallback (the checkpoint-2 + LM-head path above), unchanged.
Real open items here, not yet resolved:
- Currently uses `Local\` namespace (session-scoped). This only works when
  both programs run in the same interactive session — it does NOT work
  under the SYSTEM/Task-Scheduler deployment the program's own header still
  describes. Real regression from an earlier `Global\` version, traded
  away to dodge a privilege issue. Worth resolving deliberately, not by
  default.
- This machine's real large-page ceiling has been measured as low as
  256MB *even on a genuinely fresh boot*, not just after hours of uptime —
  the original "grab it early" theory is only partially confirmed.
- Engine-side size-matching (probe the same candidate list the reservoir
  tried, use whatever size actually maps) was fixed this session but only
  has one confirming run so far.

## Known gotchas — don't rediscover these the hard way

- **`GetLastError()` evaluation order**: `std::cerr << "literal" << GetLastError()`
  is a real bug. The streamed literal is itself an I/O call and can reset
  the thread's last-error state before `GetLastError()` is evaluated.
  Capture into a local variable immediately after the failing call, before
  any I/O touches it. This exact bug was reintroduced once already by not
  following this pattern in new code — check new error-handling code
  against it explicitly.
- **`FILE_MAP_LARGE_PAGES`**: since Windows 10 1703, `MapViewOfFile` maps
  small pages by default *even for a mapping created with `SEC_LARGE_PAGES`*
  unless this flag is also passed to the map call. Omitting it fails
  silently — reports success, delivers zero actual benefit.
- **Console output**: use `std::cout` with plain UTF-8 literals +
  `SetConsoleOutputCP(CP_UTF8)`, not `std::wcout` + wide literals.
  `SetConsoleOutputCP` only affects the narrow code page; combined with
  emoji requiring UTF-16 surrogate pairs, `wcout` breaks and can exit the
  program almost immediately with garbled output.
- **`VirtualLock` on this machine**: fails in the large majority of
  observed runs (error 1453, fragmentation) on this specific 7.87GB
  machine. Large Pages succeed occasionally (correlates with fresh boots,
  not guaranteed). Don't assume the arena is pinned without checking the
  actual boot log for that specific run.
- **Arena math**: `AllocateAligned`'s bounds check (`current_offset +
  padding + size > total_capacity`) is the real safety net for any memory
  source, including the reservoir. It only protects correctly if
  `total_capacity` reflects the *true* mapped size — never assume a
  requested size was what was actually granted.

## Genuinely open, not yet started

- Multi-token portal output (only the first generated token round-trips
  through `SiliconGatePortal` today; the rest is console-only).
- Persistent KV cache vs. rolling-summary memory across separate wake
  cycles — real undecided fork, not just an implementation detail.
- Sensing/watcher layer for proactive triggering — the "change vs.
  significance" threshold is a genuinely unsolved design question, not
  just unbuilt.
- The 32 transformer layers dominate total touched-bytes-per-cycle by
  roughly 10x over the LM head (~58GB vs ~5.6GB across one full wake
  cycle) — the LM-head materialization proved the *technique*, but the
  layers are where the technique would actually matter most, and that's
  a real arena-redesign-scale undertaking, not a quick follow-on.
