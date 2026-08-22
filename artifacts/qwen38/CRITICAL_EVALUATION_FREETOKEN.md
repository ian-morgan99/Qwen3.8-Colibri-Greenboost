# Critical Evaluation: Qwen3.8-2.4T Serving Plan vs FreeToken

Date: 2026-08-22. Basis: `backend_colibri_vllm_moe_detailed_plan.md` (268 lines, Phases 0–9),
`QWEN38_WORKSTATION_FEASIBILITY.md`, and flashml-org/freetoken @ HEAD.

## Verdict

**The bespoke vLLM-Moet plan as written will not deliver a working 2.4T model on this
workstation in reasonable time. FreeToken is the better path, with one caveat: it does not
ingest GGUF for this architecture — you must feed it safetensors (FP8 or NVFP4), not the
UD-Q1_0 GGUF currently downloading for LM Studio.**

## Plan feasibility (vLLM-Moet build)

1. **The plan's own go/no-go gate fails.** Dense + shared-expert BF16 footprint is
   ~88 GiB vs 32 GB VRAM (RTX 5090). The plan acknowledges this and recommends INT4/NF4
   dense quantization — i.e. the plan only works *after* you build a quantization pipeline
   that doesn't exist yet.
2. **ExpertWeightProvider tiering (VRAM/RAM/NVMe) is the whole project**, and it is
   research-grade: 92 layers × 512 experts × top-10, hybrid linear attention
   (full_attention_interval=4, GDN/conv), 262k context. Every one of the 14 mandatory
   telemetry metrics implies instrumentation vLLM does not expose today.
3. **Timeline risk**: Phases 0–9 with a from-scratch expert cache/prefetcher is months of
   work with a real chance the bandwidth math (PCIe/NVMe random-read for cold experts at
   decode) never reaches usable tokens/s at 1-bit quantization.

## FreeToken evaluation

Strengths (verified in source):
- Native `qwen3_5_moe` model package (`python/freetoken/models/qwen3_5_moe/`: model, moe,
  weight, gdn_reference) — same architecture family as Qwen3.8-2.4T-A95B
  (`model_type=qwen3_5_moe_text`, `Qwen3_5MoeForCausalLM`). Registered in
  `models/register.py`.
- Already implements exactly what the plan proposes to build: bandwidth-adaptive CPU–GPU
  co-execution (q* policy), LRU expert caching, double-buffered prefill streaming, elastic
  VRAM reallocation, FTW fast-weight format.
- Docs list Qwen3.5/3.6 MoE checkpoints incl. FP8 and NVFP4 variants — the NVFP4 path is
  the one that fits a 32 GB card for the dense/shared components.

Caveats / risks:
- **GGUF support is gemma4-only** (`GGUF_ARCH_TO_REGISTRY` in
  `models/gguf/config.py` maps only `gemma4`). The unsloth UD-Q1_0 GGUF being downloaded
  for LM Studio **cannot** be loaded by FreeToken. For FreeToken you need the HF
  safetensors repo (Qwen/Qwen3.8-2.4T-A95B-FP8, or an NVFP4 build such as nvidia's).
- Qwen3.8-2.4T is ~28× larger than the 35B-A3B models FreeToken documents; expert-cache
  hit rates and first-token latency on cold experts are unproven at this scale.
- 1-bit GGUF quality vs FP8/NVFP4: FreeToken's path costs more disk/RAM but preserves
  quality; the UD-Q1_0 LM Studio path is a quality gamble regardless of runtime.

## Recommendation

1. **Short term (LM Studio GUI)**: continue the UD-Q1_0 download (in progress,
   `models-download/`), accept 1-bit quality, validate that the model runs at all.
2. **Real serving path**: install FreeToken (`pip`), point it at
   `Qwen/Qwen3.8-2.4T-A95B-FP8` or an NVFP4 build, and benchmark. This replaces Phases
   1–9 of the plan with configuration + benchmarking.
3. **Keep the vLLM-Moet plan only as a fallback** if FreeToken's q* policy can't sustain
   acceptable decode throughput at 2.4T scale.
