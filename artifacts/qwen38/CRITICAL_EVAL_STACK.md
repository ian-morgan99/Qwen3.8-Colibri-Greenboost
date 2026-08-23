# Critical Stack Evaluation: FreeToken / GreenBoost / TurboQuant / vLLM-Moet

**Date:** 2026-08-23 · **Status:** Verified against freetoken 0.1.2 source (`/tmp/freetoken`) and live `ft serve` instance
**Goal:** Confirm we are maximising performance on RTX 5090 (32 GB VRAM / 91 GB RAM) for Qwen3.8-2.4T-A95B, using only technically plausible mechanisms.

---

## 1. Verdict up front

| Component | Role in final stack | Action |
|---|---|---|
| **FreeToken** | **The runtime.** Native `qwen3_5_moe`, offload/hybrid MoE backends, LRU expert cache, prefill copy overlap, CUDA graphs, pinned memory, OpenAI+Anthropic APIs | Keep as-is; tune flags (§4) |
| **vLLM-Moet bespoke engine** | Fully superseded | Drop from execution path (plan history only) |
| **GreenBoost** | Mostly superseded by FreeToken internals; only *greenboost-netd* transfer telemetry remains novel | Reduce to thin telemetry sidecar (Phase E) |
| **TurboQuant / SpectralQuant** | Not present anywhere in FreeToken. Remains our own post-baseline research idea | Defer to Phase 8, after correctness |
| **Colibrì heatmap/lookahead/packing** | Superseded by FreeToken's built-in LRU stats + cache sizing | Reuse only as tuning-input analysis |

The current plan (FREETOKEN_REVISED_PLAN.md Phases A–E) is the right architecture. This document adds the concrete optimisation levers we were missing.

---

## 2. What FreeToken gives us natively (verified in source)

### 2.1 MoE serving path — this is where all the performance lives

- **Backends** (`moe/`): `offload` (default auto), `hybrid`, `cpu`, `fused`. Auto-selects `offload`; upgrades to `hybrid` when a `ft bench bw` profile shows CPU bandwidth > 2× PCIe.
- **GPU expert cache** (`offload_cache.py`): unified slot pool, **LRU policy**, per-layer stats accumulated device-side (`lru_stats`). This *is* the "expert heatmap" Colibrì was going to build — it already exists.
- **Prefill copy overlap**: two-buffer DMA/compute overlap during prefill (`--disable-moe-prefill-overlap` to turn off). On by default. Requires cache ≥ 2×num_experts slots.
- **CPU executor** for hybrid: splits experts per layer between PCIe-fetch and CPU-compute based on benched bandwidth ratio (`--moe-hybrid-max-fetch`).
- **Pinned memory** (`kernel/pinned.py::alloc_pinned_tensor`): GreenBoost Phase 7 goal already implemented.

### 2.2 Attention & engine

- FlashInfer attention backend auto-selected (`attention_backend='fi'`), paged KV + radix cache (`hybrid_radix`), GDN/mamba and SWA pools for qwen3_5_moe hybrid layers.
- **CUDA graph capture** in engine init (`cuda_graph_bs`, `cuda_graph_max_bs`) — decode steps run graph-replayed.
- AOT prebuilt kernel table covers every served (model, format) pair at TP=1 bf16 — including `Qwen/Qwen3.5-35B-A3B{,-FP8}`, `Qwen3.6-35B-A3B{,-FP8}`, `nvidia/Qwen3.6-35B-A3B-NVFP4`. Cache misses fall back to JIT which needs nvcc (see §5 blocker).

### 2.3 Quantisation support for qwen3_5_moe (the 2.4T path)

- **FP8 block-scale** (`fused_fp8_block.py`): fp8 weights + bf16 128×128 block scales — matches `*-FP8` checkpoints.
- **NVFP4** (`fused_nvfp4.py`, `nvfp4_backends.py`): three GEMM backends selectable via `--nvfp4-backend`:
  - `marlin` (sm80–99 + vLLM)
  - **`flashinfer b12x` (sm120+ & CUDA≥13) ← RTX 5090 fast path. Verified importable in our venv (flashinfer 0.6.17, `_launch_sm120_w4a16_moe` loads OK).**
  - `triton` inline-dequant (portable fallback)
- Plain NVFP4 and nvidia/modelopt MIXED_PRECISION (per-layer W4A16-NVFP4/FP8 maps) both handled by `qwen3_5_moe/config.py`.

### 2.4 Checkpoint loading

- `ft checkpoint --model <hf> --out <ftw> [--shard-gib 8]` packs safetensors into FTW banks → dramatically faster startup vs re-parsing HF shards each boot.
- `--expert-load auto|serial|parallel`: parallel bank read is fast but needs whole-shard buffer RAM; auto falls back to serial under memory pressure. With 91 GB RAM and a ~1 TB Q1 GGUF irrelevant here (GGUF unusable — see §3), parallel should hold for FP8/NVFP4 safetensors.

---

## 3. Hard constraint: the GGUF download cannot serve via FreeToken

`GGUF_ARCH_TO_REGISTRY` supports **gemma4 only**. The 397 GB unsloth UD-Q1_0 download is exclusively for the **LM Studio GUI** path (llama.cpp runtime). For FreeToken serving of the 2.4T we need an **HF safetensors checkpoint**:

- Best case: official/nvidia **NVFP4** release (~4 bits ≈ 1.2–1.4 TB still too big for 91 GB RAM — needs the mixed-precision form or aggressive offload math re-check).
- Realistic near-term: **Qwen3.8-2.4T-A95B-FP8-class release if one exists**, else evaluate whether offload can stream enough experts at Q1-equivalent sizes.
- **Action item:** survey HF for safetensors releases of the 2.4T before planning further; do not assume the GGUF investment transfers.

---

## 4. Optimisation levers to add to the plan (concrete, plausible)

Ordered by expected impact:

1. **`ft bench bw --dtype nvfp4,bf16` first thing on the 5090.** Enables `--moe-backend auto` → hybrid selection with real numbers instead of the default offload. Zero risk, one command.
2. **FTW conversion before every serve session** (`ft checkpoint`). Startup currently re-reads raw HF shards; FTW banks make restarts minutes not hours (our live 35B load has been running >60 min largely due to cold HF-cache reads + JIT).
3. **Prebuilt kernel cache wheel.** Build `freetoken-kernel-cache` from `/tmp/freetoken/freetoken-kernel-cache/` (needs nvcc once) or set `FREETOKEN_KERNEL_CACHE_DIR` to a populated dir. Eliminates per-shape JIT stalls at startup. Alternative short-term: keep the JIT cache warm across sessions (it persists in the venv/build dirs).
4. **NVFP4 with `--nvfp4-backend flashinfer`** (b12x sm120 kernel verified importable). When a 2.4T NVFP4/MIXED_PRECISION checkpoint lands, this is the fastest routed-expert GEMM available on Blackwell consumer.
5. **Cache sizing:** let `--moe-cache-auto` size the GPU expert cache after `--kv-reserve-tokens`; then hand-tune with `--moe-cache-size` using ftmon telemetry (hit-rate proxy = fetch counts in stats). Larger cache ⇒ fewer PCIe fetches ⇒ directly higher decode TPS.
6. **Keep prefill overlap ON** (default). Only disable if VRAM pressure forces cache below 2×experts.
7. **`--expert-load parallel`** explicitly once RAM headroom is confirmed (91 GB host vs bank+buffer requirement) — removes serial-read startup penalty.
8. **CUDA graphs:** leave defaults; consider explicit `--cuda-graph-bs` list matching expected concurrency (1–4 local users) to trim capture time.
9. **Phase E telemetry:** ftmon JSONL already captures throughput/KV/requests. Add greenboost-netd only if we later need cross-host transfer metrics — otherwise drop it.

## 5. Known blockers

| Blocker | Impact | Mitigation |
|---|---|---|
| No nvcc (apt CUDA=12.0, need 13) | JIT compile fails on cache miss; slow first-load | Build kernel-cache wheel inside Docker (docker available); or user-assisted sudo install of CUDA 13 toolkit |
| No safetensors 2.4T checkpoint identified yet | FreeToken cannot serve the GGUF | Survey HF (unsloth/nvidia/modelopt) for FP8/NVFP4 safetensors of Qwen3.8-2.4T-A95B |
| Live 35B smoke test still loading (>60 min, GPU 98%) | Blocks ftsmoke checks 2–4 | Wait; next serve will be far faster with FTW + warm JIT cache |

## 6. Bottom line

We are **not** leaving performance on the table architecturally — FreeToken already implements what GreenBoost/TurboQuant/vLLM-Moet were going to build, with better integration (device-side LRU stats, graph capture, pinned DMA, Blackwell NVFP4 kernels). The wins left are operational: bench-bw calibration, FTW pre-conversion, kernel-cache wheel, correct NVFP4 backend pinning, and telemetry-driven cache sizing. The single biggest open risk is checkpoint availability for the 2.4T in a FreeToken-compatible format.
