# Revised Backend Plan: FreeToken-Based Serving for Qwen3.8-2.4T-A95B

Date: 2026-08-23. Supersedes the execution-foundation sections of
`backend_colibri_vllm_moe_detailed_plan.md` (kept as reference for the Colibrì-derived
orthogonal modules and telemetry list, most of which FreeToken already provides).

## 1. Why the pivot

New information from the flashml-org/freetoken repo (verified in source):

1. **Native `qwen3_5_moe` support** — `python/freetoken/models/qwen3_5_moe/` implements
   the exact architecture family of Qwen3.8-2.4T-A95B (`model_type=qwen3_5_moe_text`,
   `Qwen3_5MoeForCausalLM`), including GDN linear attention (`gdn.py`, `gdn_kernels.py`,
   `gdn_reference.py`) and quantized linear layers (`quant_linear.py`). Registered in
   `models/register.py`.
2. **The hard problems are already solved** — what our plan proposed to build over
   Phases 0–9 exists as configuration flags:
   - Expert tiering VRAM↔RAM: `--moe-backend offload|hybrid|cpu|fused`
   - GPU hot-set expert cache: `--moe-cache-auto` / `--moe-cache-size` / `--moe-cache-rate`
   - PCIe bandwidth calibration: `ft bench bw` (produces hardware profiles; auto-selects
     hybrid when CPU/PCIe ratio ≥ 2.0)
   - Per-layer CPU decode split: `--moe-cpu-layers`; fetch limiting: `--moe-hybrid-max-fetch`
   - Prefill miss-streaming: `--moe-prefill-hit-d2d` (CUDA ≥ 13)
   - Elastic VRAM reallocation between expert cache and KV: `--memory-ratio`,
     `--kv-reserve-tokens`
3. **Serving surface**: `ft serve` exposes OpenAI `/v1/*` and Anthropic `/v1/messages`
   APIs on port 1919, plus `GET /v1/stats` (throughput, latency, VRAM, pool occupancy) —
   covering most of our mandatory telemetry list out of the box.
4. **GGUF is gemma4-only** in FreeToken's loader (`GGUF_ARCH_TO_REGISTRY`). The UD-Q1_0
   GGUF we are downloading for LM Studio cannot be used here; FreeToken needs HF
   safetensors (FP8 or NVFP4).

## 2. Host prerequisites (verified)

| Requirement | Status |
|---|---|
| Linux x86_64 + NVIDIA GPU | ✅ RTX 5090, 32 GB |
| Driver r580+ (CUDA 13) | ✅ 595.84 |
| Python 3.12 + uv | ✅ |
| System RAM ≥ 64 GB (expert offload arena) | ✅ 91 GB |
| CUDA 13 toolkit with `nvcc` on PATH | ❌ **not installed** — required for JIT kernels |
| Disk for FP8/NVFP4 checkpoint | ⚠️ 3.1 TB free; FP8 ≈ 2.5 TB, NVFP4 smaller |

## 3. Implementation phases

### Phase A — Environment (half a day)
1. Install CUDA 13 toolkit (nvcc on PATH).
2. `uv venv && uv pip install "freetoken[accel]"` (PyPI) or `-e ".[accel]"` from a
   source clone if we need patches.
3. Sanity: `ft bench bw` runs and produces an offload/hybrid profile for this box.

### Phase B — Architecture validation on a small sibling model (1 day)
Before touching 2.4T:
1. `ft serve --model Qwen/Qwen3.5-35B-A3B` (same qwen3_5_moe code path, ~35 GB).
2. Verify: server ready on :1919, `/v1/models`, streaming chat completion,
   `/v1/stats` reporting.
3. Record baseline tokens/s with default offload backend.

### Phase C — Model acquisition (parallel, long pole)
FreeToken loads HF safetensors directly (`--model` accepts an HF repo id). Options in
order of preference:
1. **NVFP4 build** (e.g. nvidia NVFP4 release when available for Qwen3.8-2.4T) — smallest
   dense/shared footprint, best fit for 32 GB.
2. **Qwen/Qwen3.8-2.4T-A95B-FP8** (~2.5 TB download; fits disk, experts stream from RAM).
3. The LM Studio GGUF (UD-Q1_0, downloading now at 3 MB/s into `models-download/`)
   remains the **GUI/fallback path only**.

### Phase D — 2.4T bring-up and tuning (2–3 days)
1. `ft serve --model <checkpoint> --moe-backend auto --moe-cache-auto`
2. Tune in order: `--memory-ratio` → `--moe-cpu-layers` → `--moe-hybrid-max-fetch` →
   `--kv-reserve-tokens`. Use `/v1/stats` to watch pool occupancy and hit rates.
3. Acceptance criteria: stable decode ≥ 5 tok/s interactive on cold cache; no OOM across
   a 8k-token session.

### Phase E — GreenBoost / Colibrì integration (later, optional)
Only after Phase D acceptance: pinning strategy for the offload arena, heatmap-guided
`--moe-cpu-layers` selection, transfer telemetry via greenboost-netd. These layer onto
FreeToken's flags rather than requiring engine surgery.

## 4. What we keep from the old plan

- **Telemetry requirements** (§5 of old plan): map to `/v1/stats` + greenboost counters;
  anything not exposed becomes a small sidecar reader, not engine patches.
- **Exact-routing principle** (§2.2): FreeToken's cache decides residency, never routing —
  consistent with our design constraint.
- **Colibrì heatmap/lookahead/packing** (§3.1): deferred to Phase E as tuning inputs.

## 5. Risks

| Risk | Mitigation |
|---|---|
| 2.4T unproven at FreeToken scale (docs show ≤ 35B-A3B siblings) | Phase B validates code path; Phase D gates on measured tok/s before further investment |
| FP8 checkpoint is ~2.5 TB download | Start early (Phase C); NVFP4 alternative if released |
| CUDA 13 JIT compile time on first load | One-time cost; precompile during Phase A |
| 1-bit GGUF quality (LM Studio path) | Treat as smoke-test only; FreeToken path preserves quality |
