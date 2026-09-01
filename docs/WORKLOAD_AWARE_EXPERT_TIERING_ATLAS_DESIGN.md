# Workload-Aware Expert Tiering for Qwen3.8 / GreenBoost

**Status:** canonical research/implementation design for workload-conditioned T1/T2/T3 expert residency

**Primary target:** Qwen3.8-2.4T-A95B on RTX 5090 32 GB + 96 GB RAM + NVMe

**Related project:** `alesha-pro/atlas` (Weight Atlas)

**Scope boundary:** this document is about **where exact routed experts reside and when they are prefetched**. It must not change router semantics, substitute experts, prune experts, or redefine model correctness.

---

## 1. Why this exists

The current tiered-runtime design already assumes:

- T1 / VRAM = hottest/current experts;
- T2 / RAM = warm experts;
- T3 / NVMe = cold experts;
- exact router decisions remain authoritative;
- asynchronous movement attempts to hide misses;
- historical expert frequency can seed placement.

That is necessary but incomplete. A single global hotness map assumes expert demand is stationary across workloads.

Weight Atlas demonstrates a useful measurement principle: run the same checkpoint over distinct workload domains and preserve the domain label through activation capture. Atlas currently separates English, code and agent traces and measures domain-dependent residual/FFN behavior. The extension proposed here is to apply the same methodology to **MoE router/expert behavior**.

The hypothesis is:

> `P(expert | layer, workload)` may be materially more concentrated than `P(expert | layer)`.

If true, the Bridge can identify the workload before inference, GreenBoost can seed T1/T2 with a workload-specific working set, and live router telemetry can then adapt that prior during the request.

This is especially important for a 512-expert/layer model whose full expert bank is much larger than available RAM/VRAM.

---

## 2. Architectural ownership

### Qwen3.8-Colibri-Greenboost owns

- exact router trace capture;
- workload-conditioned expert statistics;
- workload profile artifacts;
- T1/T2/T3 placement policy;
- expert admission/eviction/promotion/demotion;
- asynchronous prefetch;
- expert transfer telemetry;
- cache simulator/replay;
- runtime profile learning;
- correctness/throughput benchmarks.

### VSCode-LMStudio-Bridge owns

- user-intent/task classification;
- stable workload taxonomy;
- endpoint capability negotiation;
- attaching a workload hint to requests when supported;
- request/session identity;
- outcome/intent-satisfaction telemetry;
- optional feedback that a workload classification was wrong or task intent changed.

The Bridge must **not** choose experts.

### LMStudioSupport owns

- adoption/integration guidance for LM Studio-backed GreenBoost runtimes;
- compatibility with its existing expert-placement work (#18);
- backend/profile plumbing where the runtime is actually an LM Studio/llama.cpp backend;
- validation that this optimisation can be disabled and produces the same exact-routing baseline.

### Weight Atlas is a reference methodology

Do not vendor or fork Atlas merely to implement this feature. Reuse its methodology:

- explicit calibration domains;
- real captured activations rather than synthetic claims;
- per-domain analysis;
- provenance on every measurement;
- measured fragility/behavior rather than heuristic labels.

---

## 3. Non-negotiable correctness invariants

1. The model router remains authoritative.
2. Workload profiles may affect residency/prefetch only.
3. A cache miss must fetch the router-selected expert, never a resident alternative.
4. Approximate/cache-aware routing remains a separate experimental mode and OFF by default.
5. Workload hints are advisory. An incorrect hint may reduce performance but must not change output semantics.
6. Live router evidence supersedes the prior when they disagree.
7. Every placement optimisation must have a deterministic OFF/control mode.
8. Synthetic routing traces may test code but may not qualify a workload profile.
9. Every expert identity is layer-scoped and representation-scoped.
10. Intent-success/quality regression can veto a faster profile.

---

## 4. Workload taxonomy

Start deliberately small and versioned.

```text
unknown
chat
coding_implementation
code_review
verification
bug_debugging
tool_orchestration
architecture_reasoning
research
summarisation
vision
```

The taxonomy must be shared with the Bridge through a versioned protocol. Do not create free-form workload strings in GreenBoost.

A later version may add subtypes only after measurements show they predict expert locality.

Example:

```json
{
  "taxonomy": "bridge-workload-v1",
  "workload": "code_review",
  "confidence": 0.91
}
```

Unknown/low-confidence classification must fall back to generic placement.

---

## 5. What to capture

For every exact router decision, capture at minimum:

```text
model revision
runtime revision
quant/representation
request id (opaque)
workload taxonomy/version
workload label
workload confidence
phase
position/token bucket
layer id
ordered expert ids
router scores/weights
shared-expert use
prefill/decode marker
```

Recommended phase values:

```text
prefill
decode
reasoning
tool_selection
post_tool
verification
```

If the runtime cannot reliably distinguish reasoning/tool phases, keep only prefill/decode initially rather than guessing.

Do not persist prompt text in expert telemetry.

---

## 6. Statistics to derive

### 6.1 Marginal expert demand

Per layer and workload:

```text
P(expert | layer, workload)
```

Weighted and unweighted forms:

- selection frequency;
- sum of router weights;
- rank distribution within top-k;
- unique-request frequency.

A globally frequent expert touched by one pathological request should not dominate the workload prior, hence unique-request frequency is useful.

### 6.2 Conditional entropy

Calculate:

```text
H(E | L)
H(E | L, W)
```

where:

- `E` = expert identity;
- `L` = layer;
- `W` = workload.

Primary research question:

```text
entropy_reduction = H(E | L) - H(E | L, W)
```

Large positive reduction means workload classification materially predicts expert demand.

### 6.3 Working-set coverage curves

For every layer/workload, rank experts by predicted value and calculate cumulative routed mass at N resident experts:

```text
coverage(N) for N = 8,16,32,48,64,96,128,160,192,256
```

Also calculate **byte-budget coverage** because different packed expert representations may not be uniform.

The key decision artifact is not merely `100/512 experts cached`; it is:

> what percentage of real routed expert accesses/weight mass is covered by the expert set fitting the actual T1/T2 byte budget?

### 6.4 Reuse distance

For `(layer, expert)` record token/step reuse distance:

- p50;
- p90;
- p95;
- p99;
- request-boundary recurrence.

This informs T1 versus T2 retention and hysteresis.

### 6.5 Co-activation

Capture expert sets selected together for a token/batch and derive:

- pair co-occurrence;
- small frequent expert bundles;
- conditional `P(E_j | E_i, layer, workload)`;
- batch-union working sets for prefill.

This enables grouped prefetch and contiguous expert-pack layout experiments.

### 6.6 Transition prediction

Measure whether current routing predicts near-future routing:

```text
P(E at layer N+1 | routing state at N, workload)
P(E at token t+1 | recent routing window, workload)
```

Do not call oracle lookahead implementable prefetch. Separate:

- oracle upper bound;
- workload-prior predictor;
- recent-router EWMA predictor;
- transition-table predictor.

### 6.7 Phase drift

Compare expert distributions for:

- prefill versus decode;
- before/after tool results;
- verification versus coding generation.

A single request may change expert population enough to justify a phase-local profile.

---

## 7. Placement value model

Do not rank experts on frequency alone.

Start with a transparent score:

```text
placement_value =
    predicted_access_probability
  * miss_cost_ms
  * compute_acceleration_factor
  * expected_reuse_factor
  * quality_sensitivity_factor
  / residency_bytes
```

Initially set `quality_sensitivity_factor = 1.0` until a trustworthy sensitivity measurement exists.

### Candidate components

`predicted_access_probability`
- weighted blend of workload prior and live-router heat.

`miss_cost_ms`
- T3→T2 + T2→T1 expected exposed latency for this expert/pack.

`compute_acceleration_factor`
- relative gain from GPU versus CPU execution if execution tier differs.

`expected_reuse_factor`
- derived from reuse-distance and recent recency.

`residency_bytes`
- actual packed runtime bytes, not parameter-count estimate.

### Optional later Atlas-inspired sensitivity factor

Weight Atlas measures per-tensor quantisation fragility. If a comparable expert sensitivity experiment is added, use it to avoid placing/representing a high-impact fragile expert in a way that harms quality.

Do not mix quantisation policy into phase 1 of workload-aware placement.

---

## 8. Three scheduling time scales

### 8.1 Slow loop: offline profile learning

Timescale: many requests / hours / days.

Inputs:

- captured exact router traces;
- workload label;
- hardware transfer measurements;
- cache simulator output;
- request outcome/intent success where available.

Outputs:

```text
profiles/<model>/<taxonomy>/<workload>.json
```

Profiles are immutable/versioned once published.

### 8.2 Medium loop: request initialization

At request start:

1. receive workload hint;
2. validate taxonomy/model/profile compatibility;
3. choose generic fallback if confidence/provenance is insufficient;
4. seed T2 from workload working set where needed;
5. seed T1 only within a bounded startup budget;
6. do not block indefinitely waiting for profile warmup;
7. record startup bytes/time and whether they were later useful.

### 8.3 Fast loop: live adaptation

During inference:

- maintain per-layer EWMA heat;
- update recency/reuse statistics;
- promote experts whose observed demand exceeds prior;
- demote stale experts with hysteresis;
- allow live evidence to override the workload prior;
- prefetch likely future experts with a strict bandwidth budget.

The prior is a starting point, not a lock.

---

## 9. Prior/posterior blending

A simple initial policy:

```text
heat = alpha(t) * workload_prior + (1 - alpha(t)) * live_heat
```

where `alpha(t)` decays with observed routed decisions.

Example experimental schedule:

```text
first routed decisions: alpha ~ 0.8
warm-up window:         alpha -> 0.4
after sufficient live evidence: alpha -> 0.1
```

Do not hardcode those values as final defaults. Benchmark them.

Confidence can scale initial alpha:

```text
alpha0 = profile_strength * workload_hint_confidence
```

If the runtime detects strong divergence between prior and live routing, rapidly decay the prior and record `workload_profile_mismatch`.

---

## 10. T1/T2/T3 policy

### T1: VRAM

Use for:

- currently required experts;
- highest-value imminent experts;
- experts with high GPU acceleration benefit;
- very high reuse probability;
- always-resident dense/router/shared/state tensors according to the existing memory plan.

T1 is scarce. Do not fill it entirely with historical experts before the request starts.

Recommended budget classes:

```text
T1_FIXED       dense/router/shared/runtime state
T1_ACTIVE      experts required now
T1_PREDICTIVE  small bounded next-expert set
T1_HOT         remaining long-lived hot-expert capacity
```

### T2: RAM

This is the primary workload-conditioned working set.

Use for:

- high cumulative workload coverage;
- experts expensive to fetch from NVMe;
- likely soon but not worth VRAM;
- backing for T1 promotion.

The 96 GB workstation should reserve OS/runtime headroom and calculate T2 by bytes dynamically rather than assume a fixed 70 GB.

### T3: NVMe

All other exact experts remain available.

Requirements:

- contiguous/efficient per-expert packs where practical;
- validated offsets/lengths;
- async reads;
- measured direct-I/O/mmap/page-cache alternatives;
- no semantic difference from T1/T2.

---

## 11. Admission, eviction and hysteresis

Start with deterministic policies.

Candidate admission rule:

```text
admit to T2 if expected_saved_stall_ms over horizon > transfer/admission cost
```

Candidate T1 promotion rule:

```text
promote if:
  live/workload value > promote_threshold
  AND expected residence horizon justifies H2D cost
  AND destination budget available/evictable
```

Demote only below a lower threshold:

```text
demote_threshold < promote_threshold
```

This hysteresis prevents oscillation.

Minimum comparison policies:

- LRU;
- LFU;
- LFRU;
- global learned pinning;
- workload prior + live LFRU;
- workload prior + live EWMA/hysteresis.

---

## 12. Prefetch design

### 12.1 Workload-start prefetch

Seed only a bounded amount.

Measure:

- startup bytes;
- startup latency;
- fraction used within first N layers/tokens;
- bytes evicted unused.

### 12.2 Layer lookahead

Use only information actually available before expert use.

Evaluate:

- router-computed next-layer prefetch if architecture/runtime permits;
- transition-table prediction;
- current-token co-activation bundle.

### 12.3 NVMe→RAM lead time

T3 prefetch should target a longer horizon than T2→T1 because storage latency is larger.

Separate queues and budgets for:

```text
NVMe -> RAM
RAM  -> VRAM
```

Avoid T3 prefetch starving demand I/O.

### 12.4 Wasted prefetch control

Track:

```text
prefetch_precision
prefetch_recall
useful_prefetch_bytes
wasted_prefetch_bytes
prefetch_bandwidth_share
```

Disable/degrade predictor when value is negative.

---

## 13. Bridge-to-runtime workload hint protocol

The runtime must work perfectly without the Bridge. Therefore hints are optional metadata.

Preferred semantic object:

```json
{
  "schema": "greenboost.workload-hint.v1",
  "taxonomy": "bridge-workload-v1",
  "workload": "verification",
  "confidence": 0.93,
  "intent_revision": 4,
  "request_role": "verifier"
}
```

Do not send raw user intent/prompt text.

### Transport

Because endpoints differ, support one or more adapter-level carriers:

1. backend-native request extension when supported;
2. namespaced HTTP header for local trusted endpoints;
3. side-channel/session control API for persistent verifier runtimes.

Do **not** inject the hint into user-visible model prompt text merely to reach the runtime; that wastes tokens and can alter model behavior.

Candidate local header:

```text
X-GreenBoost-Workload-Hint: <compact signed/versioned value>
```

If a generic OpenAI endpoint strips unknown fields/headers, omit the hint and run generic mode.

---

## 14. Runtime capability negotiation

Bridge endpoint metadata should expose something like:

```json
{
  "supportsWorkloadHint": true,
  "workloadHintSchema": "greenboost.workload-hint.v1",
  "workloadTaxonomy": "bridge-workload-v1"
}
```

The Bridge must not assume every `openai-compatible` endpoint understands GreenBoost hints.

The runtime must ignore unknown taxonomy/schema versions safely.

---

## 15. Outcome feedback

Placement should ultimately optimise useful work, not only cache hit rate.

Where privacy policy permits, the Bridge may send bounded outcome metadata after a request:

```json
{
  "schema": "greenboost.workload-outcome.v1",
  "request_id": "opaque-id",
  "intent_outcome": "satisfied",
  "tool_success": true,
  "user_correction_followed": false,
  "wall_ms": 123456
}
```

No prompt text, code or private tool output belongs in this feedback.

Use this only for offline evaluation/profile selection at first. Do not online-train placement from a single success/failure event.

---

## 16. Quality and intent-satisfaction gate

A faster placement profile is not automatically better.

For each profile compare:

```text
intent success / benchmark correctness
TTFT
prefill tok/s
decode tok/s
wall time
T1/T2/T3 hit rates
exposed stall ms/token
bytes NVMe->RAM/token
bytes RAM->VRAM/token
```

If exact routing and representation are unchanged, output should ordinarily remain numerically/semantically equivalent apart from nondeterminism. Still keep task-quality and intent-success metrics because scheduling bugs, stale buffers, wrong representation generations or unintended policy interactions can produce silent corruption.

A profile that improves throughput but increases intent failure must not be promoted.

---

## 17. Profile artifact schema

Suggested initial schema:

```json
{
  "schema": "greenboost.expert-workload-profile.v1",
  "model": {
    "id": "...",
    "revision": "...",
    "representation": "...",
    "layout_hash": "..."
  },
  "workload": {
    "taxonomy": "bridge-workload-v1",
    "name": "verification"
  },
  "training": {
    "trace_count": 0,
    "request_count": 0,
    "token_count": 0,
    "captured": true,
    "source_commit": "..."
  },
  "per_layer": {
    "0": {
      "experts": [
        {
          "expert_id": 17,
          "selection_probability": 0.0,
          "router_mass": 0.0,
          "request_frequency": 0.0,
          "reuse_p50": 0,
          "reuse_p95": 0,
          "bytes": 0
        }
      ],
      "coverage_curve": {
        "32": 0.0,
        "64": 0.0,
        "96": 0.0,
        "128": 0.0
      }
    }
  },
  "validation": {
    "held_out_trace_count": 0,
    "simulated_t2_hit_rate": 0.0,
    "simulated_exposed_stall_ms_per_token": 0.0
  }
}
```

Profiles must embed enough identity to reject use with the wrong model/quant/layout.

---

## 18. Required tooling

### `capture_router_trace`

Capture exact layer-scoped router decisions from a real runtime.

### `build_workload_profile`

Input:

- captured traces;
- workload labels;
- expert byte manifest;
- hardware bandwidth/latency measurements.

Output workload profile.

### `replay_tier_policy`

Replay traces under alternative cache sizes/policies without model inference.

It must simulate by bytes and transfer timing, not just entry count.

### `compare_workload_profiles`

Generate:

- conditional entropy table;
- coverage curves;
- profile overlap/Jaccard;
- cross-workload misclassification penalty;
- expected T1/T2/T3 hit/stall differences.

### `profile_runtime`

Run live requests and compare predicted versus actual cache/residency behavior.

---

## 19. Calibration corpus

Do not use only generic English/code/agent labels if the goal is Bridge optimisation.

Create a privacy-safe reproducible corpus with at least:

### Coding implementation

- bounded feature implementation;
- refactor;
- test writing;
- multi-file change.

### Code review

- inspect diff;
- identify correctness issues;
- security/reliability review;
- architecture fit.

### Verification

- intent contract + diff + tests -> PASS/REVISE;
- short structured output.

### Debugging

- logs + failure + source slices;
- iterative diagnosis.

### Tool orchestration

- tool choice/arguments;
- multi-step repository navigation.

### Reasoning/research

- architecture comparison;
- long evidence synthesis.

All fixtures should be non-sensitive and versioned.

---

## 20. Experiments that answer whether the idea is worthwhile

### Experiment A: does workload reduce routing entropy?

For each layer:

```text
H(E|L) vs H(E|L,W)
```

Decision:

- negligible reduction -> keep global/live caching as primary;
- strong reduction -> workload prior justified.

### Experiment B: fixed T2 byte budget

Replay the exact same traces under:

1. global LFRU;
2. global learned pin set;
3. correct workload prior + LFRU;
4. wrong workload prior + LFRU;
5. oracle workload prior.

Measure exposed stall, not just hit rate.

### Experiment C: warm-start cost

Compare zero seeding versus bounded T2/T1 seeding at request start.

Find the point where preload cost is repaid.

### Experiment D: prior decay

Sweep blend/decay parameters and measure mismatch recovery.

### Experiment E: verifier workload

Treat verification as a first-class target because:

- output is short;
- workload is relatively structured;
- persistent service permits stable profiling;
- higher expert locality could make the 2.4T verifier materially more useful.

---

## 21. Metrics and promotion criteria

### Mandatory runtime metrics

```text
workload hint + confidence
profile id/version
profile matched/fallback
T1 hit rate / byte rate
T2 hit rate / byte rate
T3 demand rate / byte rate
NVMe->RAM bytes/token
RAM->VRAM bytes/token
exposed expert wait ms/token
GPU idle waiting for experts
promotions/demotions
evictions
prefetch precision/recall/useful/wasted bytes
prior/live divergence
TTFT
prefill tok/s
decode tok/s
wall time
```

### Profile promotion requires

- held-out captured traces;
- improvement over global baseline in exposed stall or wall time;
- no correctness regression;
- no material intent-success regression;
- wrong-hint behavior degrades gracefully;
- profile can be disabled instantly;
- model/layout identity checks pass.

---

## 22. Failure modes

### Wrong workload classification

Mitigation:

- confidence threshold;
- generic fallback;
- fast prior decay;
- live router override.

### Distribution shift

Mitigation:

- version profiles;
- recent-vs-historical divergence telemetry;
- rebuild rather than mutate silently.

### T1 thrash

Mitigation:

- T1 fixed/active/predictive/hot sub-budgets;
- hysteresis;
- minimum residence time;
- transfer-cost-aware admission.

### T2 over-seeding

Mitigation:

- byte/time startup budget;
- track unused seeded bytes;
- preload only the coverage knee, not the entire historical set.

### Prefetch saturation

Mitigation:

- demand I/O priority;
- bandwidth caps;
- predictor kill switch;
- wasted-byte telemetry.

### Stale profile applied to wrong quant/layout

Mitigation:

- strict model revision + layout hash + representation identity.

### Silent memory corruption / stale transfer completion

Coordinate with tier-transfer pinning/lifetime work; never infer correctness from throughput.

---

## 23. Implementation phases for a coding agent

### Phase 0 — provenance and baseline

1. Finish exact captured router traces per existing issue #2.
2. Ensure model/expert dimensions and bytes come from authoritative artifacts.
3. Establish global LRU/LFRU replay baseline.
4. Measure target-machine NVMe/RAM/PCIe transfer characteristics.

**Gate:** no workload learning from synthetic traces.

### Phase 1 — workload-labelled capture

1. Add workload taxonomy fields to trace schema.
2. Build fixed calibration fixtures.
3. Capture real traces for at least `coding_implementation`, `code_review`, `verification`, `bug_debugging`, `tool_orchestration`.
4. Validate no prompt contents are stored.

### Phase 2 — offline analysis

1. Implement conditional entropy.
2. Implement working-set coverage curves.
3. Implement reuse distance/co-activation.
4. Generate immutable workload profiles.
5. Add cross-workload overlap and wrong-profile penalty report.

### Phase 3 — simulator

1. Extend tier replay to accept workload profiles.
2. Simulate by bytes and measured timing.
3. Compare global vs workload priors.
4. Report exposed stall ms/token and bandwidth.
5. Identify profiles that cannot beat global baseline and do not implement them live yet.

### Phase 4 — runtime medium-loop seeding

1. Add optional workload hint input.
2. Load compatible profile.
3. Seed bounded T2 set.
4. Add profile/fallback telemetry.
5. No T1 speculative seeding yet unless proven cheap.

### Phase 5 — live prior/posterior scheduler

1. Add live per-layer heat.
2. Blend workload prior with observed heat.
3. Add hysteretic promotion/demotion.
4. Add mismatch detection and fast decay.
5. Preserve exact router choices.

### Phase 6 — predictive prefetch

1. Add transition/co-activation predictor.
2. Separate T3→T2 and T2→T1 horizons.
3. Cap bandwidth.
4. Benchmark against no-prefetch and oracle upper bound.

### Phase 7 — Bridge integration

1. Implement endpoint capability for workload hints.
2. Send taxonomy/version/workload/confidence out-of-band.
3. Never add hint tokens to normal model prompt.
4. Add outcome feedback only after privacy-safe schema is accepted.

### Phase 8 — quality/outcome-aware promotion

1. Feed benchmark/intent-success outcomes into offline reports.
2. Promote only profiles that improve end-to-end utility.
3. Keep profile updates human/audit visible initially.

---

## 24. Acceptance criteria

- [ ] Real captured router traces are labelled with a versioned workload taxonomy.
- [ ] No synthetic trace can qualify a workload profile.
- [ ] `H(E|L)` and `H(E|L,W)` are reported for the target model.
- [ ] Per-workload expert coverage curves are produced for realistic T1/T2 byte budgets.
- [ ] Reuse distance and co-activation are measured.
- [ ] A deterministic replay compares global LFRU against workload-conditioned policies using measured transfer costs.
- [ ] At least one workload demonstrates a material reduction in exposed expert wait or wall time on held-out traces before live default enablement.
- [ ] Runtime workload hints are optional and advisory.
- [ ] Wrong/missing hints degrade to generic/live behavior without correctness change.
- [ ] Live router evidence can override/decay the workload prior.
- [ ] Placement never changes router-selected expert IDs.
- [ ] T1/T2/T3 telemetry is sufficient to attribute gains/regressions.
- [ ] Workload profiles are model/quant/layout version-bound.
- [ ] Bridge integration sends only bounded metadata, not prompt content.
- [ ] Profile promotion includes intent/task success, not throughput alone.
- [ ] The entire feature has a deterministic OFF mode reproducing the exact baseline.

---

## 25. Final design principle

The desired system is not:

> keep globally popular experts resident.

It is:

> use the task as a prior, use exact live routing as evidence, and continuously spend scarce VRAM/RAM on the experts with the highest expected reduction in exposed work — without ever changing what the model chose to execute.

That creates a three-level adaptive system:

```text
Bridge intent/workload
        ↓
offline workload prior
        ↓
request-start T2/T1 seeding
        ↓
live exact router observations
        ↓
posterior heat / promote / demote / prefetch
        ↓
T1 VRAM | T2 RAM | T3 NVMe
        ↓
performance + intent outcome
        ↓
offline profile evaluation
```

For the 2.4T verifier specifically, this should be tested early: verification is a structured, repeatable workload with short output, making it one of the best candidates for highly concentrated expert locality to turn a marginal 2.4T workstation runtime into a practically useful background reviewer.