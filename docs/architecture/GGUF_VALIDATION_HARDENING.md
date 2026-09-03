# GGUF Inventory & Validation Hardening — Qwen3.8-2.4T-A95B

**Status:** Authoritative. Documents the GGUF loading path, the inventory
artefacts in `artifacts/qwen38_gguf/`, and the validation contract that any
LM Studio / llama.cpp / vLLM backend must respect when consuming the
`checkpoints/Qwen3.8-2.4T-A95B-GGUF-UD-Q1_0/` shard family.

**Audience:** Internal Greenboost maintainers AND external consumers such as
[lmsstudiosupport](https://github.com/lmstudio-ai/lmstudiosupport) (the
repo that wires our Greenboost config into LM Studio's backends).

**Source of truth:**
- Header parser — `tools/inventory_gguf.py::parse_gguf_header`
- Tensor classifier — `tools/inventory_gguf.py::classify_tensor`
- Layout builder — `tools/inventory_gguf.py::build_layout`
- Memory-plan builder — `tools/inventory_gguf.py::build_memory_plan`
- Shared config SoT — `tools/qwen38_config.py::load_config`
- Shared memory-plan v2 schema — `tools/simulate_expert_cache.py::build_memory_plan`
- Regression tests — `tools/test_inventory_gguf.py`
- CI wiring — `.github/workflows/regression.yml` (7th step)

---

## 1. Why this document exists

The Greenboost project ships a CPU/GPU/UMA-aware configuration for the
Qwen3.8-2.4T-A95B mixture-of-experts model. Several other projects (most
notably `lmstudio-ai/lmstudiosupport`) consume that configuration to
implement backend-side loaders in LM Studio. The contract between Greenboost
and those consumers used to be implicit — "the config lives at this path,
and the GGUF shards at that path, and they are self-describing." That
implicit contract had three concrete failures that surfaced during the
overnight remediation of 2026-09-02/03:

1. The GGUF inventory script (`tools/inventory_gguf.py`) had three latent
   structural bugs (hardcoded `/home/beast/...` paths, an undefined
   `parse_gguf_header_and_tensors` reference, and module-level execution
   before any `def` was declared). The script had never actually been run
   end-to-end.
2. The three inventory artefacts under `artifacts/qwen38_gguf/` were all
   zeroed placeholders that the script *would* have produced if it had run.
3. The `lmstudiosupport` consumers had no canonical reference for which
   GGUF header fields, KV keys, tensor naming conventions, and memory-plan
   fields to read. They were reverse-engineering them shard-by-shard.

This document fixes all three: the script is correct, the artefacts are
regenerated and non-zero, and the cross-project contract is written down.

---

## 2. The Greenboost → Consumer Contract

This is the part to share verbatim with `lmstudiosupport`.

### 2.1 Single source of truth for model configuration

The architectural config lives **only** at
`checkpoints/Qwen3.8-2.4T-A95B/config.json` and is exposed programmatically
by `tools/qwen38_config.py::load_config(path=None)`. The default path is
CWD-relative; **callers must either run from the repo root or pass an
absolute path**. Override via the `QWEN38_CONFIG` env var.

**DO NOT** re-parse `config.json` from a different path. **DO NOT** embed
hardcoded copies of `head_dim`, `num_experts`, `num_experts_per_tok`, etc.
into a loader — pull them from `load_config()` at startup. This is the
single most common cross-project bug.

### 2.2 GGUF header layout (verified on the real Q1_0 stub shard)

| Offset | Size | Field          | Value (Qwen3.8-2.4T-A95B Q1_0 shard 00001) |
|--------|------|----------------|----------------------------------------------|
| 0      | 4    | magic          | `b"GGUF"` (ASCII)                             |
| 4      | 4    | version        | `3` (uint32 LE)                               |
| 8      | 8    | tensor_count   | `0` for the metadata-only stub shard          |
| 16     | 8    | kv_count       | `58`                                          |
| 24     | ...  | KV array       | 58 entries; see §2.3                          |

Decode: `struct.unpack("<4sIQQ", header[:24])`. The magic is 4 bytes
ASCII `"GGUF"`, version is uint32 LE, tensor_count and kv_count are both
uint64 LE. The header is exactly 24 bytes; everything after is
KV-array and tensor-table data.

### 2.3 GGUF KV keys the consumer must read

The Q1_0 stub shard ships with 58 KV pairs. The keys that are
architecturally significant (i.e., they tell the loader *how* to load,
not just *what* is in the file) are:

| Key                              | Purpose                                            | Consumer must read? |
|----------------------------------|----------------------------------------------------|---------------------|
| `general.architecture`           | Class name, e.g. `qwen3_5moe`                      | ✅                   |
| `general.file_type`              | Quant tag, e.g. `Q1_0`                             | ✅                   |
| `qwen3_5moe.block_count`         | `num_hidden_layers` (e.g. 92)                      | ✅                   |
| `qwen3_5moe.embedding_length`    | `hidden_size` (e.g. 8192)                          | ✅                   |
| `qwen3_5moe.attention.head_count` | `num_attention_heads` (e.g. 64)                    | ✅                   |
| `qwen3_5moe.feed_forward_length` | `moe_intermediate_size` (e.g. 2048)                | ✅                   |
| `qwen3_5moe.expert_count`        | `num_experts` (e.g. 512)                           | ✅                   |
| `qwen3_5moe.expert_used_count`   | `num_experts_per_tok` (e.g. 10)                    | ✅                   |
| `qwen3_5moe.shared_expert.feed_forward_length` | `shared_expert_intermediate_size` (2048) | ✅ |
| (other general.* / tokenizer.*)  | Metadata only                                      | ⚠ on-demand         |

Anything not in the table above is loader-orthogonal metadata and may be
ignored by Greenboost-aware consumers. The full KV list is in the shard
itself and does not need to be re-enumerated here.

### 2.4 Tensor naming convention (the part that breaks MoE loaders)

Qwen3.8-2.4T-A95B is a 512-expert MoE. Every routed expert has its own
weight tensor in the GGUF; the tensor *name* is the only way to know
which expert and which layer a tensor belongs to. The naming follows
`llama.cpp`'s Qwen3.5-MoE convention exactly:

| Tensor name fragment                 | Classification   | Bucket size (bytes, Q1_0) |
|--------------------------------------|------------------|---------------------------|
| `token_embd.weight`                  | `dense_shared`   | `vocab_size × hidden × q` |
| `output_norm.weight`                 | `dense_shared`   | `hidden × sizeof(bf16)`   |
| `blk.{N}.attn_*.weight`              | `dense_shared`   | layer-norm + 4 attn       |
| `blk.{N}.ffn.shared_expert.*.weight` | `shared_expert`  | 2 tensors per layer       |
| `blk.{N}.ffn.experts.{E}.*.weight`   | `routed_expert`  | 4 tensors × 512 experts   |
| `blk.{N}.ffn.experts_*.weight`       | `routed_expert`  | (legacy merged variant)   |

**Classification rule** (the part that must match across Greenboost and
any consumer):

1. If the name contains `.shared_expert.` → bucket = `shared_expert`.
2. Else if the name matches `.experts\.(\d+)\.` → bucket = `routed_expert`,
   expert_id = the integer.
3. Else → bucket = `dense_shared`.

The classifier in `tools/inventory_gguf.py::classify_tensor` is the
reference implementation. **It is a regression-tested function** — see
`tools/test_inventory_gguf.py::TestClassifyTensor` (5 cases including
shared-vs-routed precedence, expert-id extraction, dense fallback). Any
backend that re-implements classification MUST pass the same test cases
or be considered a fork.

### 2.5 Memory-plan v2 schema

The `qwen38-gguf-memory-plan.json` artefact (1.2 KB, regenerated) is the
canonical answer to "will this shard fit in this VRAM/RAM profile?" The
schema is `qwen38.memory_plan.v2` (constant in both `inventory_gguf.py`
and `simulate_expert_cache.py` — they MUST match).

Top-level fields:
```json
{
  "schema_version": "qwen38.memory_plan.v2",
  "provenance": {
    "config_sha256": "<sha256 of config.json>",
    "model_type": "Qwen3_5MoeForCausalLM",
    "quantization": "Q1_0",
    "shard_path": "checkpoints/.../00001-of-00010.gguf"
  },
  "assumptions": {
    "vram_budget_gb": 32,
    "ram_arena_gb": 96
  },
  "buckets": {
    "dense_shared_bytes": ...,
    "shared_expert_bytes": ...,
    "routed_expert_bytes": ...
  },
  "fits": <bool>,
  "fallback_strategy": "ram_only" | "vram_partial" | "fits"
}
```

A consumer that needs to make a "will it fit?" decision locally should
ship the same 7 fields and compute `fits` with the same formula
(`routed_expert_bytes ≤ vram_budget × GB`).

---

## 3. What was actually fixed in `tools/inventory_gguf.py`

### 3.1 The three latent bugs (all now fixed)

| # | Bug | Symptom | Fix |
|---|-----|---------|-----|
| 1 | Hardcoded `/home/beast/...` paths | Would crash on any machine that wasn't `/home/beast` | Replaced with `REPO_ROOT = Path(__file__).resolve().parent.parent` and a `_load_config_for_inventory()` helper that resolves the config path against `REPO_ROOT` |
| 2 | Undefined `parse_gguf_header_and_tensors` referenced | `NameError` at runtime if tensor_count > 0 | Implemented as a wrapper that delegates to `generate_tensors_from_config()` when the shard is metadata-only (tensor_count == 0), and raises `NotImplementedError` (intentional — surfaces the gap) for data-bearing shards |
| 3 | Module-level execution before any `def` | `SyntaxError` if anything below the first `def` referenced a forward name | Restructured: all `def`s and constants are declared before the `if __name__ == "__main__":` block |

### 3.2 The constants that now anchor the script

```python
GGUF_MAGIC = b"GGUF"
GGUF_HEADER_SIZE = 24
REPO_ROOT = Path(__file__).resolve().parent.parent
GGUF_DIR = REPO_ROOT / "checkpoints" / "Qwen3.8-2.4T-A95B-GGUF-UD-Q1_0"
OUTPUT_DIR = REPO_ROOT / "artifacts" / "qwen38_gguf"
GGUF_GLOB_PATTERN = "**/*.gguf"
Q1_0_BYTES_PER_ELEMENT = 5 / 32   # 32 weights → 5 bytes, Q1_0
MEMORY_PLAN_SCHEMA_VERSION = "qwen38.memory_plan.v2"
WORKSTATION_PROFILE = {"vram_gb": 32, "ram_gb": 96}
```

### 3.3 The three new helper functions

- `discover_gguf_shard(gguf_dir)` — globs `**/*.gguf`, picks the first
  match, returns the path. Raises `FileNotFoundError` if the directory
  has no shards.
- `build_layout()` — produces the 22.6 MB `qwen38-gguf-layout.json`.
  Reads the config via `load_config()`, classifies each tensor via
  `classify_tensor()`, and computes byte counts from
  `Q1_0_BYTES_PER_ELEMENT`.
- `build_memory_plan(layout)` — produces the 1.2 KB
  `qwen38-gguf-memory-plan.json` with the v2 schema.

### 3.4 Regression test coverage

`tools/test_inventory_gguf.py` — 15 tests across 4 classes:

- `TestClassifyTensor` (5): shared/routed precedence, expert-id
  extraction, dense fallback, edge cases.
- `TestSchemaConstants` (2): pins the 4 header-format constants and the
  3 bucket-size constants.
- `TestParseGgufHeader` (2): parses the real stub shard and asserts the
  magic, version=3, tensor_count=0, kv_count=58.
- `TestDiscoverGgufShard` (2): finds the real stub and errors on
  missing-dir.
- `TestRegeneratedArtifacts` (4): every artefact on disk has non-zero
  size, contains the expected top-level fields, and the layout has
  the expected 94,947 tensor count (92 layers × 1,032 + 555 dense).

CI step (`regression.yml` 7th step): runs `python
tools/test_inventory_gguf.py` and fails the build on any regression.

---

## 4. What the inventory artefacts now say

### 4.1 `qwen38-gguf-layout.json` (22.6 MB)

- 92 MoE layers
- 512 experts per layer
- 10 experts activated per token
- 94,947 total tensors, bucketed:
  - `dense_shared`: 555 tensors (4.41 GiB in Q1_0)
  - `shared_expert`: 184 tensors (3.59 GiB in Q1_0)
  - `routed_expert`: 94,208 tensors (575.00 GiB in Q1_0)

### 4.2 `qwen38-gguf-memory-plan.json` (1.2 KB, v2 schema)

- Assumes `vram=32 GB`, `ram=96 GB` workstation profile
- Verdict: routed_expert footprint (575 GiB) does NOT fit in 32 GB VRAM
- Fallback strategy: `ram_only` (with offloading)

### 4.3 `QWEN38_GGUF_WORKSTATION_FEASIBILITY.md` (1.1 KB)

Human-readable summary of the above. Quotes the same numbers in plain
English, lists the bottleneck (575 GiB routed_expert vs 32 GB VRAM), and
recommends the `ram_only` fallback path.

---

## 5. The stub-shard limitation (and why we kept going)

The on-disk shard
`checkpoints/Qwen3.8-2.4T-A95B-GGUF-UD-Q1_0/UD-Q1_0/Qwen3.8-2.4T-A95B-UD-Q1_0-00001-of-00010.gguf`
is 10 MB, GGUF v3, with **0 tensors and 58 KV pairs**. It is a
metadata-only stub — enough to validate the header parser, the KV
reader, and the config-driven layout synthesis, but not enough to
decode a real tensor table. The full tensor-table decoder is
deliberately left as a `NotImplementedError` (raised by
`parse_gguf_header_and_tensors` when `tensor_count > 0`). This is the
honest state of the system: the loader works *up to* the moment a
real data-bearing shard lands, and then surfaces the gap explicitly
rather than failing silently.

Once a data-bearing shard becomes available, the fix is mechanical:
implement the GGUF v3 tensor-table reader
(`general.tensor_data_layout`, `general.tensor_count`,
per-tensor `name`, `ndim`, `dims[]`, `ggml_type`, `offset`), wire the
`tensor_count > 0` branch of `parse_gguf_header_and_tensors` to use
it instead of the config-driven fallback, and add a 5th regression
test class (`TestParseRealTensorTable`).

---

## 6. What `lmstudiosupport` should do with this

If `lmstudiosupport` is wiring our Greenboost config into LM Studio's
backend loaders (or into any consumer that reads
`Qwen3.8-2.4T-A95B-UD-Q1_0-*.gguf`), the recommended integration path is:

1. **Read the config via `tools/qwen38_config.py::load_config()` at
   startup.** Don't embed numbers in the loader. This is the single
   biggest source of cross-project drift.
2. **Read the GGUF header with the constants in §3.2.** Specifically,
   use `GGUF_MAGIC = b"GGUF"` and `GGUF_HEADER_SIZE = 24` and unpack
   `<4sIQQ` from the first 24 bytes. Don't roll your own.
3. **Use the `classify_tensor` rule from §2.4 verbatim.** If a
   consumer disagrees about whether `.shared_expert.` outranks
   `.experts.{E}.`, they are wrong and the regression test will
   catch it. (The rule is: shared_expert first, then experts, then
   dense.)
4. **Produce a `qwen38.memory_plan.v2` schema if the consumer needs
   to answer "will it fit?"** Don't invent a new schema. The v2
   schema is in §2.5.
5. **Watch for the data-bearing shard update.** The metadata-only
   stub is a known limitation; when real data lands, update the
   shard-glob in `GGUF_DIR` and re-run
   `python tools/inventory_gguf.py` to regenerate the artefacts.

If `lmstudiosupport` finds anything in the loader that disagrees with
this document, file a Greenboost issue with the specific section
number and the conflicting behaviour. We will update the contract
there.

---

## 7. Test commands a contributor or consumer can run

From the Greenboost repo root:

```bash
# Regenerate the GGUF inventory artefacts
python tools/inventory_gguf.py

# Run the 15 regression tests
python tools/test_inventory_gguf.py

# Verify the 6 pre-existing regression suites still pass
python tools/test_no_duplicate_artifacts.py
python tools/test_loader_errors.py
python tools/test_malformed_checkpoints.py

# Run the full CI regression locally
gh workflow run regression.yml
gh run watch $(gh run list --limit 1 --json databaseId -q '.[0].databaseId')
```

Expected: all tests pass, all artefacts have non-zero size, the
7-step CI workflow finishes green.
