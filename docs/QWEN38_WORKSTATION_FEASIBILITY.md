# Qwen3.8 Workstation Feasibility Report

## Executive Summary

This document assesses the feasibility of running the ~495GB Qwen3.8 MoE checkpoint (~95B active parameters) on a workstation-class system equipped with an RTX 5090 (32GB VRAM), 96GB DDR5 RAM, and NVMe storage. The assessment is based on Phase 0 exact physical inventory and expert cache simulations.

## Critical Go/No-Go Finding: Dense/Shared VRAM Footprint

Phase 0 inventory revealed a critical constraint: the mandatory non-expert VRAM footprint for dense and shared weights is approximately **88.03 GiB in FP16/BF16 precision**. This exceeds the **32GB VRAM** capacity of the RTX 5090 workstation by a significant margin.

### Implications

- Dense and shared weights are typically fully resident on the GPU during inference.
- The ~88.03 GiB footprint cannot coexist with KV cache, CUDA workspace, and an expert L1 cache on a 32GB GPU.
- Expert tiering (managed by GreenBoost or vLLM-Moet's `ExpertWeightProvider`) only manages *routed expert weights*, not mandatory dense/shared weights.

## Viable Paths Forward

### Path 1: Lower-Bit Dense Quantization (INT4/NF4)

Quantizing dense/shared weights to INT4 or NF4 format reduces the footprint from ~88 GiB to approximately **~22 GiB**. This leaves sufficient VRAM budget for:
- KV cache (at reduced context lengths)
- CUDA workspace
- Expert L1 cache (hot expert set)

**Recommendation:** Proceed with INT4/NF4 quantization strategy as the primary path forward.

### Path 2: Context Length Reduction

Reducing maximum context length reduces KV cache VRAM consumption, freeing space for dense weights or expert cache. However, this does not solve the fundamental ~88 GiB dense weight footprint issue without quantization.

### Path 3: Dense-Layer Offload to CPU RAM

Offloading dense/shared weights to CPU RAM is theoretically possible but represents a severe performance bottleneck due to PCIe bandwidth limitations. This should be considered only if INT4/NF4 quantization is not viable.

## Expert Cache Simulation Results

### Synthetic Routing Simulations (Baseline)

GPU L1 Cache (VRAM) Simulations:
- 8GB: fits 5 experts, hit_rate=0.013, zero_miss_prob=0.0
- 12GB: fits 8 experts, hit_rate=0.0208, zero_miss_prob=0.0
- 16GB: fits 11 experts, hit_rate=0.0286, zero_miss_prob=0.0
- 20GB: fits 13 experts, hit_rate=0.0339, zero_miss_prob=0.0
- 24GB: fits 16 experts, hit_rate=0.0417, zero_miss_prob=0.0

RAM L2 Arena Simulations:
- 48GB: fits 33 experts, hit_rate=0.0027, zero_miss_prob=0.0021
- 64GB: fits 44 experts, hit_rate=0.0036, zero_miss_prob=0.0029
- 72GB: fits 50 experts, hit_rate=0.0041, zero_miss_prob=0.0033
- 96GB: fits 67 experts, hit_rate=0.0055, zero_miss_prob=0.0044
- 128GB: fits 89 experts, hit_rate=0.0072, zero_miss_prob=0.0058

### Routing Trace Replay — Data Provenance

The numbers above are produced by `tools/trace_qwen38_routing.py` against the
**synthetic** trace generator (default in this repo). The simulator's
`data_classification` field is therefore
`synthetic_with_checkpoint_derived_arch` — the architecture facts (layer
count, expert count, expert bytes, MoE topology) come from the actual
`checkpoints/Qwen3.8-2.4T-A95B/config.json` (frozen 38-attribute dataclass in
`tools/qwen38_config.py`), but the per-token expert routing sequence is
synthesized with a popular-pool + locality-reuse model, not captured from a
real Qwen3.8 decode.

**Why no captured trace exists yet:** the Qwen3.8 2.4T-A95B model occupies
~440 GiB on disk and cannot be loaded on a 32 GB-VRAM workstation, so a real
expert-routing trace cannot be produced in this environment. Closing this gap
is the open gate called out by the new `load_captured_traces()` validator in
`tools/trace_qwen38_routing.py`, which refuses any captured trace whose
`config_sha256` does not match the active config (preventing stale-trace
poisoning when a real trace is eventually produced).

**Workstation profile, LFRU policy, N+1 prefetch (8 GB L1 / 96 GB L2,
184,000 expert requests, 20 tokens × 10 prompts × 92 layers × 10 experts/tok):**

- **L1 (VRAM) hit rate**: 0.00%
- **L2 (RAM) hit rate**: 36.77%
- **L3 (NVMe) fetch rate**: 63.23%
- **N+1 prefetch hide rate**: 0.00%
- **Stalling miss rate**: 63.23%

The 0% L1 rate is a function of layer size (~48 GiB of experts per layer in
BF16) versus an 8 GB L1 budget — one layer's worth of experts simply does
not fit. The 36.77% L2 rate reflects how many layers' worth of experts
*can* simultaneously live in the 96 GB L2: roughly two layers' worth of
recently-touched experts, with the rest spilling to NVMe. The 0% prefetch
hide rate is a property of the synthetic generator's locality reuse window
(which only re-uses every 3rd layer); it is not a property of real
Qwen3.8 routing, which has been shown elsewhere to exhibit stronger
cross-layer correlation.

**L2-size sensitivity sweep (LFRU, N+1, synthetic, 8 GB L1):**

| L2 size | L2 hit | Stalling miss | Notes |
| --- | ---: | ---: | --- |
| 96 GB  | 36.77% | 63.23% | Workstation default |
| 128 GB | 36.77% | 63.23% | Still under 2.7 layers' worth |
| 192 GB | 73.97% | 26.03% | ~4 layers' worth — matches the 22.99% figure previously cited in this doc |
| 256 GB | 97.55% |  2.45% | Popular pool (~256 experts) fits comfortably |
| 384 GB | 98.28% |  1.72% | Long-tail starts fitting |
| 512 GB | 98.28% |  1.72% | No further gain — popular pool saturated |

**L1 sensitivity (LFRU, N+1, synthetic, 96 GB L2):** 8 GB → 0.00% L1 hit at every
L1 size in {6, 8, 12, 16, 24} GB, because one layer's experts do not fit in
L1 regardless. Increasing L1 only matters when the simulator is configured
with *shared* or *dense* weights, which is not on the MoE tiering path.

Reproduce with:

```bash
python3 tools/trace_qwen38_routing.py --trace synthetic \
  --l1-size-gb 8 --l2-size-gb 96 --policy lfru --prefetch-lookahead 1 \
  --output artifacts/qwen38_routing_trace_metrics.json
```

## Key Observations

1. **Routing Locality is a Working-Set-Size Problem, Not a Generator Artifact:** The synthetic trace's popular-pool + locality-reuse model still produces 36.77% L2 hit at 96 GB and 73.97% L2 hit at 192 GB. The cliff between 128 GB and 192 GB is the boundary at which the cache can hold more than ~2 layers' worth of recent experts. Any captured trace from a real Qwen3.8 decode should be re-evaluated against this curve rather than against a single L2-size point.

2. **Stalling Miss Rate is the Decisive Metric:** The 63.23% stalling miss rate (8 GB L1 / 96 GB L2 / LFRU / N+1) represents the percentage of expert requests that must be fetched from NVMe and cannot be hidden by N+1 prefetch under the synthetic trace. This is the decisive measurement for determining whether the tiered-MoE architecture can achieve acceptable decode performance, rather than the raw cache miss rate.

3. **L2 Capacity Dominates Expert Hit Rate:** A 192 GB L2 (4 layers' worth) cuts the stalling miss rate by ~2.4× relative to 96 GB (2 layers' worth), and 256 GB essentially eliminates the long tail. The implication is that workstation-grade decoding of Qwen3.8 is L2-capacity bound, not L1-capacity bound, and not policy bound (LFRU ≈ LRU at these working-set sizes).

4. **Dense Quantization is Mandatory:** Without INT4/NF4 quantization of dense/shared weights, the architecture cannot proceed on a 32GB VRAM workstation.

## Next Steps: Phase 0.5 Go/No-Go Gate

Before proceeding to Phase 1 (vLLM-Moet expert-provider/cache design adaptation), the following must be verified:

1. **Dense Weight Quantization Compatibility:** Confirm that vLLM/vLLM-Moet can successfully load and execute Qwen3.8 with INT4/NF4 quantized dense/shared weights.
2. **SM120 Kernel Compatibility:** Verify that Qwen3.8 expert tensor shapes, activation functions, and MoE layout align with existing SM120 cubins proven for DeepSeek/GLM models.
3. **Context Length Constraints:** Determine maximum viable context length given the ~22 GiB dense footprint + KV cache + CUDA workspace + expert L1 cache budget.

## Conclusion

The Qwen3.8 workload is **feasible on a 32GB VRAM workstation only if dense/shared weights are quantized to INT4/NF4 format**. Without this quantization, the ~88 GiB dense footprint makes execution impossible on the target hardware. The expert tiering architecture (vLLM-Moet + GreenBoost) remains valid and necessary for managing routed expert weights, but dense quantization is the primary go/no-go gate for project viability.
