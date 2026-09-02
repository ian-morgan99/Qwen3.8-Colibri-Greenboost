# Detailed Backend Plan: vLLM-Moet Execution Foundation with Colibrì, GreenBoost, and TurboQuant/SpectralQuant

> **CANONICAL IMPLEMENTATION PLAN**
>
> - **Status:** Phase 0 / Phase 0.5 — checkpoint inventory and routing-trace provenance are in place, but **do not start Phase 1** until the following gates are PASS (see `artifacts/qwen38-phase-status.json` for machine-readable state):
>   - checkpoint-derived Phase 0 inventory (`PASS` — see `artifacts/qwen38_phase0_inventory_report.md`)
>   - routing trace provenance with `config_sha256` gate (`PASS` for synthetic; `BLOCKED` for captured — see `artifacts/qwen38_routing_trace_metrics.json`)
>   - dense execution format chosen (`PASS` — INT4/NF4 mandatory per `docs/QWEN38_WORKSTATION_FEASIBILITY.md` "Viable Paths Forward → Path 1: Lower-Bit Dense Quantization"; the 88.03 GiB FP16/BF16 dense footprint cannot fit on a 32GB VRAM GPU)
>   - SM120 expert-kernel compatibility (`BLOCKED` — `artifacts/kernel-compatibility.json` still says `requires_verification`)
> - **Source of truth for numbers:** `tools/qwen38_config.py` (frozen 38-attribute dataclass) and the artifacts it regenerates (`artifacts/qwen38-layout.json`, `artifacts/qwen38_phase0_inventory_report.md`, `artifacts/qwen38_routing_trace_metrics.json`). Do not hard-code model dimensions, expert counts, dense footprint, or per-expert byte sizes anywhere else — link the SoT artifact instead.
> - **Supersedes:** the older `backend_colibri_vllm_moe_plan_revised.md` (already deleted; this document is the only active implementation plan).
> - **Companion docs:** `docs/architecture/QWEN38_CHECKPOINT_DERIVED.md` (architecture), `docs/QWEN38_WORKSTATION_FEASIBILITY.md` (feasibility numbers), `docs/WORKLOAD_AWARE_EXPERT_TIERING_ATLAS_DESIGN.md` (tiering design), `docs/T3_DISK_TIER_PLAN.md` (L3 disk tier).

## 1. Executive Summary

This document provides a detailed, fleshed-out plan for implementing the backend inference runtime for the Qwen3.8-2.4T-A95B model (~95B active parameters, 512 experts) on a workstation-class system (RTX 5090 32GB VRAM, 96GB DDR5 RAM, NVMe storage).

> **Provenance** (read `docs/architecture/QWEN38_CHECKPOINT_DERIVED.md` for full derivation): `~95B active parameters` and `512 experts` are **checkpoint-derived** from `checkpoints/Qwen3.8-2.4T-A95B/config.json` (config_sha256 `89391ac8f44227959cb4b89df5c94d0b78d5686bc102988ca2ca4447fc4b84f1`); see `tools/qwen38_config.py` for the authoritative 38-attribute frozen dataclass.

The architecture is built on a clear engine boundary:
- **Execution Foundation**: vLLM-Moet (proven expert-cache architecture, SM120 kernels)
- **Colibrì-Derived Work**: Heatmap tracking, lookahead, expert packing, learned residency (orthogonal storage/cache modules)
- **GreenBoost**: Pinned memory, transfer, allocation, telemetry tuning (orthogonal transfer/memory optimization)
- **TurboQuant / SpectralQuant**: Optional compact representations after baseline correctness
- **GGUF**: Optional ingestion/export path, not runtime architecture (parallel investigation only)

This approach turns the project from a "fairly ambitious multi-runtime integration project" into a "much more tractable new-model enablement plus generalized expert-storage project."

---

## 2. Architecture and Engine Boundary

### 2.1 ExpertWeightProvider Abstraction
Expert weights must sit behind an `ExpertWeightProvider`/expert-residency abstraction. **Do not extend PagedAttention for expert tiering**, as PagedAttention is fundamentally the KV-cache side of vLLM. Mixing expert storage management into PagedAttention couples two unrelated memory lifecycles and will make long-context tuning much harder.

### 2.2 Exact Routing Baseline
The router should choose the model-correct experts; the cache manager should decide where those experts live. Memory-aware expert substitution should be a later experimental mode because it changes model semantics.

**Baseline flow:**
```
router → exact expert IDs → ExpertWeightProvider → VRAM/RAM/NVMe
```

**Not:**
```
router + cache state → different expert IDs
```

### 2.3 Expert Hierarchy Architecture
```
                 Qwen3.8 router
                       │
                 exact expert IDs
                       │
                       ▼
             ExpertWeightProvider
              /        |        \
             /         |         \
       GPU L1      RAM L2      NVMe L3
       hot set     arena       pack store
             \         |         /
              \        |        /
                 SM120 MoE kernel
```

---

## 3. Orthogonal Modules

### 3.1 Colibrì-Derived Work
- **Heatmap Tracking**: Maintain real-time statistics on expert activation frequency across tokens and layers.
- **Lookahead**: Implement router lookahead to anticipate upcoming expert demands based on context patterns.
- **Expert Packing**: Group frequently co-activated experts to minimize NVMe I/O and optimize contiguous storage.
- **Learned Residency**: Implement dynamic expert tier assignment based on usage patterns and workload heatmaps.

### 3.2 GreenBoost
- **Pinned Memory**: Implement pinned host arenas supporting high-throughput asynchronous DMA and overlap with GPU compute (not CUDA mapped zero-copy access).
- **Transfer**: Optimize NVMe→RAM and RAM→GPU data movement pipelines.
- **Allocation**: Manage memory pools efficiently across VRAM, RAM, and NVMe tiers.
- **Telemetry Tuning**: Provide low-overhead metrics collection for transfer efficiency and bottlenecks.

### 3.3 TurboQuant / SpectralQuant
- **Optional Compact Representations**: Apply after baseline correctness is proven.
- **Tensor-Type Dependent Quantization**: Validate actual tensor inventory before applying quantization formats, as routed experts, attention matrices, embeddings, router weights, and shared experts may not all receive the same effective bits/weight.

### 3.4 GGUF / llama.cpp
- **Parallel Investigation Only**: GGUF remains an optional ingestion/export path, not runtime architecture.
- **Do Not Splice into Hot Path**: Only consider if later evidence shows vLLM cannot support the released weights efficiently.

---

## 4. Detailed Implementation Phases (Phases 0-9)

### Phase 0 — Exact Physical Inventory and Compatibility Gate
**Goal**: Prove the tensor numbers and establish hard compatibility gates before any runtime implementation begins.

**Tasks:**
1. Parse every Qwen3.8 tensor from the released checkpoint or GGUF.
2. Classify each tensor as: dense/shared / router / routed-expert / KV-related.
3. Record stored dtype and byte size for each tensor.
4. Calculate expert size per layer.
5. Calculate the exact permanently resident non-expert footprint (dense/shared/router/embeddings).
6. Calculate the active expert bytes per layer/token.
7. **Compute mandatory non-expert VRAM footprint**: `VRAM = mandatory dense/shared + router + KV(min context) + CUDA workspace + expert cache`
8. Produce the **maximum available expert-cache size** (not just simulate 8/12/16GB pools). If this number is only 1–3GB, the architecture needs dense-layer offload or lower-bit dense kernels before proceeding.
9. Generate 48/64/72GB RAM-arena simulations to model L2 constraints.
10. Run routing traces on representative coding prompts if the model can be executed anywhere.
11. Estimate **stalling cache miss rate** and **milliseconds of exposed expert-transfer latency per generated token**, not just average expert hit rate or synthetic zero-miss metrics (critical: a cache with high expert hits can still perform badly if the misses that occur are not hidden by N+1/N+2 asynchronous prefetch).
    > **Provenance**: Phase 0 traces are **synthetic** (placeholder routing patterns) on top of **checkpoint-derived** architecture. They are sufficient to drive simulator calibration but are NOT a substitute for a captured-on-real-requests trace; see `artifacts/qwen38-phase-status.json` "routing_trace_provenance_phase0_5" gate.
12. **Output Qwen3.8→vLLM-Moet compatibility gate matrix**: `tensor → shape → dtype → expert role → existing kernel compatible? → conversion required?`
    - This is a **go/no-go gate**. If Qwen's K/N dimensions do not match one of the existing SM120 cubins, the project needs a kernel extension before cache work matters.

**Critical Go/No-Go Finding**: The mandatory non-expert VRAM footprint is **~88.03 GiB in FP16/BF16**, which exceeds the 32GB VRAM of the RTX 5090 workstation. This means the dense/shared portion alone cannot fit alongside KV, CUDA workspace, and an expert L1 cache on a 32GB GPU. The architecture **requires dense-layer offload or lower-bit dense kernels** before expert tiering can proceed.

> **Provenance**: `~88.03 GiB` is **computed_from_checkpoint** (arithmetic on `checkpoint_derived` fields: 92 layers × dense params per layer in FP16/BF16). It is the minimum VRAM required to hold the non-expert portion of the model, independent of expert tiering. See `docs/QWEN38_WORKSTATION_FEASIBILITY.md` "Critical Go/No-Go Finding" and `tools/qwen38_config.py` for the per-layer breakdown.

**Deliverable**: Phase 0 report with six mandatory answers before touching runtime code:
- exact mandatory non-expert VRAM footprint;
- actual free VRAM available for the L1 expert pool at 8K/32K/64K context;
- expert tensor shape/kernel compatibility with existing `vLLM-Moet` SM120 kernels;
- bytes per routed expert and active expert bytes per layer;
- simulated stalling cache miss rates and exposed transfer latency for candidate GPU/RAM cache sizes;
- whether existing expert-pack conversion can represent Qwen3.8 without changing model semantics beyond the separately measured quantisation loss.

---

### Phase 1 — Reuse the Existing vLLM-Moet Expert-Provider/Cache Design
**Goal**: Adapt model loader and expert indexing for Qwen3.8 using the proven vLLM-Moet architecture. No GGUF execution layer.

**Tasks:**
1. Ingest upstream checkpoint using vLLM-Moet's existing loader mechanisms.
2. Transform routed experts into a compact runtime representation.
3. Persist expert packs in a format compatible with SM120 MoE kernels.
4. Implement `ExpertWeightProvider` interface with VRAM/L1, RAM/L2, and NVMe/L3 backends.
5. Integrate exact expert ID routing: `router → exact expert IDs → ExpertWeightProvider`.
6. Validate basic inference correctness against an untiered reference model.

**Deliverable**: Functional vLLM-Moet expert-provider/cache design adapted for Qwen3.8 with exact routing.

---

### Phase 2 — Exact L1 GPU + L2 RAM Execution
**Goal**: Prove tiered execution correctness and quantization quality against reference implementations for a small subset or representative layers.

**Tasks:**
1. Implement GPU L1 hot set residency for currently activated experts (up to 10 per token + shared expert).
2. Implement RAM L2 arena for warm experts that do not fit in VRAM.
3. Ensure tiered execution is **bit-identical to the same quantized representation running fully resident**.
4. Ensure the quantized representation separately passes **quality equivalence/regression thresholds** against the original model.
5. Measure baseline L1 hit rate, L2 hit rate, and L3 fetch rate without prefetching.
6. Validate SM120 MoE kernel execution with tiered expert weights.

**Deliverable**: Working L1 GPU + L2 RAM execution with proven bit-identical equivalence to quantized reference and quality equivalence to original model.

---

### Phase 3 — L3 Persistent NVMe Expert Packs
**Goal**: Implement cold expert storage on NVMe with efficient packaging and metadata management.

**Tasks:**
1. Design contiguous expert records for NVMe storage to optimize sequential I/O.
2. Implement metadata/version checks for expert pack compatibility.
3. Build persistent quantization cache for L3 weights.
4. Implement NVMe→RAM prefetch pipelines for cold experts.
5. Measure L3 fetch rate and bytes NVMe→RAM/token metrics.

**Deliverable**: L3 persistent NVMe expert packs with contiguous records and metadata/version checks.

---

### Phase 4 — CUDA-Graph-Safe Miss Recovery and Batched Fetch
**Goal**: Reuse the proven vLLM-Moet replay architecture where possible to handle cache misses efficiently.

**Tasks:**
1. Implement CUDA-graph-safe miss recovery mechanisms.
2. Batch expert fetch operations to minimize NVMe I/O latency overhead.
3. Ensure miss recovery does not break CUDA graph execution for hit cases.
4. Measure replay percentage and unique expert misses/step metrics.
5. Optimize batched fetch to overlap with computation where possible.

**Deliverable**: CUDA-graph-safe miss recovery and batched fetch infrastructure.

---

### Phase 5 — Router Lookahead and Asynchronous Prefetch
**Goal**: Anticipate expert demands and prefetch warm/cold experts before they are needed.

**Tasks:**
1. Start with evidence-based prefetch: next-use prediction from recent expert recurrence; persistent hot-set prefetch; layer-wise transition statistics; co-activation sets.
2. Only later implement learned/predictive lookahead (router lookahead based on context patterns is too magical as exact router results for future layers depend on hidden state not yet computed).
3. Build asynchronous prefetch pipeline for warm experts (RAM L2) and cold experts (NVMe L3).
4. Optimize I/O overlap with computation to hide NVMe latency.
5. Measure useful-prefetch ratio and wasted-prefetch bytes metrics.
6. Tune prefetch depth and lookahead window based on telemetry data.

**Deliverable**: Router lookahead and asynchronous prefetch infrastructure.

---

### Phase 6 — Workload Heat/Persistent Pinning
**Goal**: Implement dynamic expert tier assignment based on usage patterns and workload heatmaps.

**Tasks:**
1. Implement heatmap tracking for expert activation frequency across tokens and layers.
2. Develop learned residency policies to move experts between VRAM/RAM/NVMe based on usage.
3. Implement persistent pinning for high-heat experts to reduce cache churn.
4. Measure cache churn and expert-set Jaccard similarity between adjacent tokens/layers.
5. Tune heat thresholds and tier assignment policies based on telemetry.

**Deliverable**: Workload heat tracking and persistent pinning with dynamic tier assignment.

---

### Phase 7 — GreenBoost Transfer and Memory Optimisation
**Goal**: Optimize pinned memory, transfer, allocation, and telemetry for maximum efficiency.

**Tasks:**
1. Implement pinned host arenas supporting high-throughput asynchronous DMA and overlap with GPU compute (not CUDA mapped zero-copy access).
2. Optimize NVMe→RAM and RAM→GPU data movement pipelines.
3. Manage memory pools efficiently across VRAM, RAM, and NVMe tiers.
4. Implement low-overhead telemetry collection for transfer efficiency and bottlenecks.
5. Measure bytes RAM→GPU/token and GPU idle time waiting for experts metrics.

**Deliverable**: GreenBoost transfer and memory optimization infrastructure.

---

### Phase 8 — Alternate Quant Formats / TurboQuant / SpectralQuant
**Goal**: Explore optional compact representations after baseline correctness is proven.

**Tasks:**
1. Validate actual tensor inventory and tensor-type dependent quantization requirements.
2. Evaluate selective CPU/GPU residency for dense/shared weights if they exceed 32GB VRAM.
3. Implement native low-bit representations for non-critical tensors.
4. Test TurboQuant and SpectralQuant compact representations on a subset of experts.
5. Measure quality degradation vs. memory savings tradeoffs.

**Deliverable**: Validated alternate quant formats or residency strategies for dense stacks.

---

### Phase 9 — Experimental Cache-Aware Routing
**Goal**: Implement memory-aware expert substitution as an experimental, off-by-default feature.

**Tasks:**
1. Develop cache-aware routing logic that considers memory tiers when selecting experts.
2. Ensure this feature is **off by default** to avoid changing model semantics prematurely.
3. Allow fallback to exact expert IDs if cache-aware substitution degrades quality.
4. Measure impact of cache-aware routing on stalling cache miss rate and overall decode performance.

**Deliverable**: Experimental cache-aware routing module, off by default.

---

## 5. Critical Telemetry Requirements (Mandatory)

The following metrics must be implemented and monitored throughout all phases. The **primary** metrics (#1–#4) are the ones that actually determine decode tok/s, TTFT, and whether Phase 1+ is meeting its SLO; the **secondary** metrics (#5–#14) are still required for capacity planning and diagnosis but should be read in light of the primary ones.

**Primary — exposed-wait / direct user-impact metrics**

1. **stalling cache miss rate** (critical: drives decode performance more strongly than headline token→expert hit percentage; measures milliseconds of exposed expert-transfer latency per generated token)
2. **GPU idle time waiting for experts** (the actual wall-clock cost the user pays for a miss)
3. **p50/p95 expert-fetch latency** (end-to-end NVMe→RAM→GPU; what the GPU is waiting on)
4. **replay percentage** (how often the cache must re-fetch a previously-resident expert; indicates churn that can mask as a hit)

**Primary — pressure / efficiency metrics**

5. **bytes NVMe→RAM/token** (NVMe pressure; primary L3 capacity signal)
6. **bytes RAM→GPU/token** (GPU pressure; primary L1 capacity signal)
7. **useful-prefetch ratio** and **wasted-prefetch bytes** (prefetch quality; pairs 1:1)
8. **decode tok/s and TTFT** (the only metrics the end user actually feels)

**Secondary — locality and counter-metrics (still required, but not the primary SLO)**

9. **L1 hit rate** (VRAM hot set) — derived from #1 + #5
10. **L2 hit rate** (RAM arena) — derived from #1 + #5
11. **L3 fetch rate** (NVMe pack store) — derived from #5
12. **unique expert misses/step** (compositional, used for prefetch batching)
13. **cache churn** (L1 evictions per second)
14. **expert-set Jaccard similarity between adjacent tokens/layers** (router stability, used by Phase 5 lookahead)

> **Anti-pattern warning:** do not optimise for a high L1/L2 hit rate while exposed-wait (#1, #2) regresses. The "zero-miss-replay" headline metric from the vLLM-Moet paper is *exactly* the sort of derivative counter that hides decode stalls — prefer the exposed-wait metrics above when reasoning about user-visible performance.

---

## 6. Overall Assessment and Next Steps

The big idea is right: the VRAM/RAM/NVMe hierarchy, Colibrì-inspired locality, GreenBoost support, and measurement-first approach are all solid.

What must be maintained is the **engine boundary**:
> **Use vLLM-Moet as the execution foundation, adapt its proven expert-cache architecture to Qwen3.8, and borrow Colibrì's storage/cache ideas. Do not splice llama.cpp/GGUF into the hot path unless later evidence shows vLLM cannot support the released weights efficiently.**

This turns the project from a fairly ambitious multi-runtime integration project into a much more tractable **new-model enablement plus generalized expert-storage project**.

The immediate next step is to execute **Phase 0 — Exact Physical Inventory** to prove the tensor numbers before proceeding with implementation.