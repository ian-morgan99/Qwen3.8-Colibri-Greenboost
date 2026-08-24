# FreeToken Tuning Policy & Performance Matrix

## Policy

Whenever we add or discover a new `ft serve` option (a flag, backend choice, or
environment knob), we MUST do two things:

1. **Add it to the GUI launcher** — `tools/ft-gui-launch.sh` (zenity form), so it
   is selectable at start without editing scripts.
2. **Benchmark it and record the result in the matrix below**, so future choices
   are data-driven rather than guessed.

Benchmark procedure: run `tools/ftsmoke.py` against the running server, plus a
fixed 256-token generation prompt, and record decode tok/s, TTFT, and VRAM.

## Performance Matrix — Huihui Qwen3.8-27B-abliterated NVFP4 (RTX 5090)

| Date       | nvfp4-backend | attention | ctx      | max-req | decode tok/s | TTFT  | VRAM   | Notes                    |
|------------|---------------|-----------|----------|---------|--------------|-------|--------|--------------------------|
| 2026-08-23 | triton        | fi (auto) | 262144   | 4       | ~65          | ~0s   | 28 GiB | baseline; smoke test PASS |

(Add a row for every new option tested. Keep the best-performing config noted here.)

**Current best:** baseline row above.

## GUI options currently exposed

- NVFP4 backend: triton / flashinfer / marlin / auto
- Attention backend: auto (fi) / fi / fa3 / triton
- Port
- Context length: default (262k) / 8192 … 131072
- Max concurrent requests: 1 / 2 / 4 / 8
- Reasoning parser: qwen3 (default) / auto / off / deepseekv32 / gpt_oss / glm /
  minimax / minimax_m3 / muse_glimmer / gemma4
- Temperature: blank = model generation_config default; a value switches the
  server to `--sampling-defaults none` (per-request `temperature` in the API
  body then governs). NOTE: FreeToken has no server-side default-temperature
  flag; temperature is otherwise always per-request.
- KV cache strategy: radix (default) / naive
- Weight dtype: bfloat16 (default) / float16 / float32 / auto
- MoE CPU layers: none (all GPU) / fraction or layer count — MoE-only, no
  effect on this dense model; kept for the future 2.4T MoE target

**Not supported by FreeToken (cannot be exposed):**
- *Reasoning budget / effort* — no such flag exists. Reasoning length is
  controlled per-request via the chat template / prompt, not the server.
- *KV cache quantisation* — no `--kv-cache-dtype` flag; KV is fp/bf16 only.
- *Number of experts* — fixed by the checkpoint architecture; only MoE CPU
  offload (`--moe-cpu-layers`) and expert caching are tunable, and neither
  applies to dense models.

## T1/T2 memory routing & pre-2.4T checklist

FreeToken routes intelligently between VRAM (T1) and host RAM (T2); there is
no T3 (NVMe) tier. For MoE models:

- `--moe-backend auto` picks the offload family, choosing `hybrid` when an
  `ft bench bw` profile recommends it.
- `--moe-cache-auto` sizes the GPU expert cache from free VRAM (KV gets
  `--kv-reserve-tokens` as a floor).
- `--expert-load auto` falls back from parallel to serial RAM reads when RAM
  is tight.
- `--moe-hybrid-max-fetch -1` splits decode-step cache misses between PCIe
  fetch and CPU compute using the benched bandwidth ratio.

**Before the 2.4T model arrives:**
- [ ] Run `ft bench bw --model <path>` once to create the machine bandwidth
      profile so hybrid auto-routing works optimally.
- [ ] Serve with `--moe-backend auto --moe-cache-auto`.
- [ ] Note: ~440 GB Q1_0 exceeds 91 GB host RAM — even with hybrid offload,
      expect heavy CPU compute of misses. Evaluate mgoin pruned75 (~436 GB)
      vs smaller quants; no T3 tier exists to spill to NVMe.

## 2.4T feasibility analysis (written before download completes)

Hardware: RTX 5090 32 GB (T1) + 91 GB RAM (~61 GB usable, T2) + 3.7 TB NVMe.
Target: unsloth UD-Q1_0 GGUF, **397 GB total** across 10 shards.

**Verdict: it will NOT fit in T1+T2 — shortfall is ~305 GB.** FreeToken has
no T3/NVMe tier, so a full-resident load is impossible on this box as-is.

Why it might still work partially:
- Only ~95B params are active per token (A95B). At Q1_0's ~1 bit/param that's
  ~12 GB of expert weights touched per token — but WHICH experts rotate with
  the token, and across a 64-layer model the union touched over a sequence
  approaches the full 397 GB without caching.
- Hybrid offload (`--moe-backend hybrid`) streams misses over PCIe while
  computing others on CPU; but misses would have to come from disk once RAM
  overflows — which FreeToken cannot do.

Paths that could make it work, ranked:

1. **Smaller quant (best).** A ~90 GB quant (e.g. ~Q1_s or a pruned variant)
   fits T1+T2 fully resident-ish with expert cache. Watch for unsloth UD-Q2_K_XL
   or RedHatAI NVFP4 *pruned* releases. Pruned75 at 436 GB still too big;
   need ≤~85 GB to leave room for KV + OS.
2. **Convert GGUF → FTW with --moe-backend offload** (`ft checkpoint`): banks
   experts for streaming, but still requires T2 residency of the banks — same
   397 GB problem.
3. **LM Studio fallback:** llama.cpp CAN mmap from NVMe (pseudo-T3), giving
   maybe 1–3 tok/s — usable proof-of-life only.
4. **Not viable:** any FreeToken native run of the full 397 GB file.

Recommendation: let the current download finish (it doubles as LM Studio
mmap test material), but plan the real FreeToken target as a ≤90 GB quant or
pruned NVFP4 release. Revisit when RedHatAI/mgoin publish smaller variants.

## GreenBoost T3 question — source-verified answer (2026)

**Does the GreenBoost shim give us T1/T2/T3 (NVMe tier) instead of just
T1/T2? No.** The original Colibrì/GreenBoost design
(see `docs/QWEN38_WORKSTATION_FEASIBILITY.md`) specified an L3/NVMe tier and
simulated it (L1 hit 4.30%, L2 hit 71.98%, L3 fetch 22.99% stalling misses),
but that tier was **never implemented in FreeToken**. Verified from the
installed source (`freetoken-env/.venv/.../freetoken/`):

| Component | What the code actually does |
|---|---|
| `moe/host_banks.py` | `HostBank` = **anonymous** mmap (`mmap.mmap(-1, asize)`) — address space only, filled once at load via O_DIRECT preadv, then `cudaHostRegister`'d (pin-after-fill). Not file-backed; no runtime disk reads during inference. |
| `moe/host_banks.py` `HostResidency.PAGEABLE` / `lock()` | Exists as an enum value but movement paths are explicitly "not implemented"; `lock()` raises `NotImplementedError`. |
| `checkpoint/ftw.py` | FTW format is a **load-time** optimization: chunked multi-threaded O_DIRECT reads into host banks; banks must be RAM-resident thereafter. The file-backed `mmap` in `_map()` is only used by the non-O_DIRECT reader path to memcpy into banks — still load-time. |
| `moe/offload_cache.py` | GPU-side LRU slot cache over experts backed by host banks; `policy_ids = {"lru": 0}` — no prefetch/paging policy beyond LRU. |
| `moe/expert_banks.py` | Low-RAM fallback drops parallel→serial build but banks still must fit resident RAM (`_host_ram_fits_parallel` checks MemAvailable vs banks + one shard). |
| `engine/cache_budget.py` | Auto-sizing splits **GPU** VRAM between MoE slots and KV pages only; host RAM is assumed sufficient for banks, never budgeted as a tier. |

Conclusion: FreeToken's offload is strictly a two-tier system — pinned host
RAM (T2) feeding a GPU expert slot cache (T1). There is no file-backed mmap,
no runtime disk paging, no pageable-bank eviction. A 397 GB model cannot run
natively regardless of shim/config. To get a real T3 you would have to write
one: the most plausible patch is a file-backed-mmap variant of `HostBank`
(the PAGEABLE plumbing partially exists) plus a paging policy in
`OffloadMoeCache` — a genuine contribution upstream, but not something a
flag enables today.

## Research findings (2025–2026 survey, incl. local-LLM second opinion)

Second opinion from local Qwen3.8-8B confirmed the core math and added
corrections:

- **12 GB/token is a lower bound** — add attention/shared/router/embedding/KV
  overhead → plan for **13–18 GB/token** depending on context.
- Union-of-experts argument holds for **batched/prefill** workloads (batch of
  B tokens can touch up to min(E, B·k) experts per layer), not single-token
  decode.
- PCIe 5.0 x16 (~50 GB/s) caps pure weight-streaming decode at ~4 tok/s with
  0% cache hit; DRAM-compute path caps at ~5–10 tok/s.
- Single NVMe cold random serving: **0.1–1 tok/s**; warm page-cache hot set:
  0.5–3 tok/s. RAID0 helps little beyond ~3 tok/s (Linux buffered I/O is the
  limiter, per KTransformers community data).
- Real-world datapoint: DeepSeek-671B Q4 via KTransformers gets 3–6 tok/s
  with all experts in RAM, 1.5–3 tok/s paging from Gen5 NVMe. KTransformers
  ≈2× faster than llama.cpp in MoE-offload scenarios.

Techniques worth tracking / adopting where possible:

| Technique | Relevance to us |
|---|---|
| Fiddler-style CPU compute of misses (send activations to CPU, not weights to GPU) | FreeToken hybrid already does this — validate it's activation-shipping, not weight-copying |
| Routing-aware caching (frequency/entropy-based, not plain LRU) | Not exposed by ft flags; engine-internal |
| KV quantization FP8/FP4 + attention-weighted eviction | ft has no kv-cache-dtype flag; revisit on engine updates |
| Prefix caching | Available: `--cache-type radix` — already in GUI |
| Speculative decoding (EAGLE/MTP) | Check `ft serve --help` for draft-model support when 2.4T arrives |
| Expert pruning/merging/distillation | The only realistic route to ≤90 GB full-resident |
| Mixed-precision residency (hot experts high-bit, cold low-bit) | Unsloth Dynamic quants already do this statically |

Revised strategy ranking for 2.4T:
1. Smaller/pruned/merged quant that fits T1+T2 fully (only true fix).
2. llama.cpp/KTransformers mmap pseudo-T3 as proof-of-life (expect 0.5–3 tok/s).
3. FTW conversion only helps if we later get an mmap/disk tier in FreeToken.

## Not yet exposed (dense model — likely irrelevant)

MoE flags (`--moe-cache-*`, `--expert-load`, `--moe-cpu-layers`) apply to MoE
models only; the huihui 27B is dense. Revisit when serving the 2.4T MoE.
