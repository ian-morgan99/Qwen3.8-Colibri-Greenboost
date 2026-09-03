# Validation Layers for the Qwen3.8 Loader Path

**Status:** Authoritative. Defines the three-layer validation stack that any
loader, simulator, or downstream tool that consumes
`checkpoints/Qwen3.8-2.4T-A95B/config.json` or its shards must respect.

**Source of truth:**
- Layer 1 (cross-field) — `tools/validate_architecture.py`
- Layer 2 (byte-range) — `tools/inventory_checkpoint.py`
- Layer 3 (typed errors) — `tools/loader_errors.py`
- Regression tests — `tools/test_loader_errors.py`,
  `tools/test_malformed_checkpoints.py`,
  `tools/verify_issue2_routing_provenance.py`,
  `tools/verify_sm120_kernel_compat.py`
- CI wiring — `.github/workflows/regression.yml`

## Why three layers, in this order

The Qwen3.8-2.4T-A95B loading path has to defend against three qualitatively
different classes of failure, and the layers are ordered so that the most
specific failure (a per-tensor byte-range violation) is detected *after* the
config has already been cross-field-validated, but *before* any layout or
ranking decision is made on top of the loaded data.

The order is also a stability contract: every layer raises a small, fixed
set of typed error names (Layer 3). Downstream code branches on those names,
not on the layer that produced them. A future contributor who reverses the
order, or who has a layer raise a non-canonical error, will break every
caller that imports `loader_errors`.

## The three layers

### Layer 1 — cross-field architecture invariants (`validate_architecture.py`)

Validates *the config*, not the shards. Asserts that the model dimensions
cohere: `head_dim × num_heads == hidden_size`, `top_k ≤ num_experts`,
`num_local_experts ≤ num_experts`, `num_hidden_layers > 0`, etc.

- **On failure:** `InvalidModelConfig`, `CrossFieldInconsistent` (Layer 3).
- **Hardware needed:** no — pure config-only.
- **Tested by:** `validate_architecture` CI step (last of 6).
- **Why it runs last in CI:** it depends on the *names* of the typed errors
  produced by Layers 2 and 3, so a regression in the contract is caught
  before this step runs.

### Layer 2 — per-tensor byte-range invariants (`inventory_checkpoint.py`)

Validates *the shards on disk*. For every tensor in every shard's header,
asserts:

1. `data_offsets[0] >= 0` and `data_offsets[1] > data_offsets[0]`
2. `data_offsets[1] <= shard_file_size`
3. `prod(shape) * DTYPE_BYTES[dtype] == data_offsets[1] - data_offsets[0]`
4. Header is parseable JSON with the required keys (`__metadata__` is
   optional; `data_offsets` and `dtype` are required)

The `DTYPE_BYTES` table covers both the Python-style names (`float32`,
`int64`, `bool`) and the safetensors on-disk short codes (`F32`, `I64`,
`BOOL`, `BF16`, …) per
https://huggingface.co/docs/safetensors/index#format — because the
on-disk format uses the uppercase short codes and the Python-side type
hints may use either form.

- **On failure:** `InvalidTensorShape`, `TensorOutOfFileBounds`,
  `MalformedShardHeader` (Layer 3).
- **Hardware needed:** no — pure file-IO and arithmetic.
- **Tested by:** `test_malformed_checkpoints` CI step (5 of 6), with 14
  adversarial cases × 9 error codes.
- **Why it runs after `test_loader_errors`:** the byte-range validator
  *uses* the typed error names, so the test that proves the names exist
  must run first.

### Layer 3 — the typed-error contract (`loader_errors.py`)

Nine `@dataclass(frozen=True)` error codes. Every layer raises these;
nothing else does. This is the public API that downstream code (Colibri
integration, future Tier-1 / Tier-2 ranking code, the metrics pipeline)
imports.

The full set is:

| Code | Layer(s) that raise it | Meaning |
|---|---|---|
| `InvalidModelConfig` | 1 | A field in `config.json` is malformed or out of range |
| `CrossFieldInconsistent` | 1 | A derived quantity doesn't match its inputs (e.g. `head_dim × num_heads != hidden_size`) |
| `InvalidTensorShape` | 2 | A tensor's declared shape * dtype cannot fit in its declared byte range |
| `TensorOutOfFileBounds` | 2 | A tensor's `data_offsets` extend past the shard's file size |
| `MalformedShardHeader` | 2 | A shard's safetensors header is unparseable or has required fields missing |
| `IntegerOverflow` | 1, 2 | A multiplication in the validation step overflowed Python's `int` (note: Python ints don't overflow — this means the input shape * dtype is not representable in any reasonable sense, e.g. >2^62 elements) |
| `ExpertPackRangeError` | 1, 2 | A packed-expert layout references a slot outside `[0, num_experts)` |
| `DuplicateExpert` | 1 | Two tensor names map to the same `(layer, expert_id)` slot |
| `OrphanTensor` | 2 | A tensor in a shard has no matching slot in the declared layout |

(The last two are reserved for the future Tier-1/Tier-2 ranking code; they
are not raised by current layers, but they are part of the contract so that
the contract doesn't have to be amended when the ranking code lands.)

## CI step ordering — and why

`.github/workflows/regression.yml` runs six steps in this order:

1. `test_no_duplicate_artifacts` — surface accidental duplicate paths
   (commit `158e38a`).
2. `verify_issue2_routing_provenance` — prove the routing simulator uses
   real Qwen3.8-2.4T-A95B parameters (issue #2, hardware-free ACs).
3. `verify_sm120_kernel_compat` — hold the SM_120 kernel gate in
   `requires_verification` state (issue #4).
4. `test_loader_errors` — prove the nine typed errors exist with the
   correct fields.
5. `test_malformed_checkpoints` — prove the byte-range validator (Layer 2)
   produces the right Layer 3 codes on adversarial inputs.
6. `validate_architecture` — prove Layer 1's cross-field math, using
   the Layer 3 names from step 4.

The rationale for the ordering: each step's correctness depends on the
contract from earlier steps. A regression in Layer 3 (step 4) is caught
*before* Layer 2 (step 5) exercises the broken contract, and both are
caught *before* Layer 1 (step 6) tries to import a now-renamed error
class. This means the failure point in CI tells the operator exactly
which contract is broken.

## How to add a new check

If the new check raises an existing Layer 3 code, add the corresponding
adversarial test in the appropriate `tools/test_*.py` file. CI order
does not change.

If the new check needs a *new* Layer 3 code (because it's a new class of
failure), the procedure is:

1. Add the new `@dataclass(frozen=True)` to `tools/loader_errors.py`.
2. Add a `test_*` case in `tools/test_loader_errors.py` that proves the
   new code is constructible.
3. Add the adversarial test that exercises the new code in the appropriate
   `tools/test_*.py` file (typically `test_malformed_checkpoints.py`).
4. Bump the assertion in `verify_issue2_routing_provenance.py` and
   `validate_architecture.py` if they were enumerating error codes.
5. Update this document with the new code's meaning and which layer(s)
   raise it.

Do **not** add a new check that raises a non-canonical exception. The
contract is that *only* Layer 3 names are user-visible; everything else
is a programming error.

## How to add a new layer

Don't, unless there is a clean qualitative boundary that the three
current layers don't cover (e.g. KV-cache invariants, or runtime tensor
allocation). If a new layer is needed, it goes between Layer 1 and the
shard load — *not* between Layer 2 and Layer 3, because Layer 2 already
*is* the layer that depends on Layer 3's protocol. New layers must:

- Raise only Layer 3 names.
- Be hardware-free by default.
- Have a corresponding CI step that runs after `test_loader_errors` and
  before `validate_architecture`.

## Related documents

- `docs/architecture/QWEN38_CHECKPOINT_DERIVED.md` — the architectural
  facts that Layer 1's invariants are derived from.
- Issue #2 (Phase 0 routing simulator) — the consumer of the routed-expert
  layout that Layer 2 produces.
- Issue #4 (SM_120 kernel compatibility) — the gate that blocks Phase 1
  until hardware verification populates `artifacts/kernel-compatibility.json`.
- Issue #6 (Colibri loader hardening) — the upstream work that consumes
  the Layer 3 contract.
