# Hand-off to `lmstudio-ai/lmstudiosupport`

**TL;DR for backend implementors:** Read the **Greenboost → Consumer
Contract** below. Then run `python tools/inventory_gguf.py` from the
Greenboost repo root to regenerate the canonical artefacts. Don't
re-implement header parsing, tensor classification, or memory-plan
schema — they are already correct and regression-tested.

---

## Background

`Qwen3.8-2.4T-A95B-UD-Q1_0` is the Q1_0-quantised GGUF family for
Greenboost's 2.4 T-parameter mixture-of-experts model. The shards
live in `checkpoints/Qwen3.8-2.4T-A95B-GGUF-UD-Q1_0/` (10 shards,
`.gguf` extension, 00001–00010). The Greenboost repo ships a
loader-side inventory that produces three artefacts under
`artifacts/qwen38_gguf/`.

If `lmstudiosupport` is wiring the Greenboost config into LM Studio
backends, this note describes the contract your loader should match
and the tests it should pass.

## Greenboost → Consumer Contract

### 1. Configuration: one path, one loader

```python
from tools.qwen38_config import load_config
cfg = load_config()  # MUST run from repo root, or pass absolute path
```

Do not embed `head_dim`, `num_experts`, `num_experts_per_tok`, etc.
in the loader. Pull them from `load_config()` at startup. Override
via `QWEN38_CONFIG` env var.

The full config surface is in `tools/qwen38_config.py` (frozen
dataclass with `to_dict()` for serialisation).

### 2. GGUF header (verified on the real Q1_0 stub shard)

24-byte header:

| Offset | Size | Field          | Value             |
|--------|------|----------------|-------------------|
| 0      | 4    | magic          | `b"GGUF"`         |
| 4      | 4    | version        | `3` (uint32 LE)   |
| 8      | 8    | tensor_count   | `0` (metadata-only stub) |
| 16     | 8    | kv_count       | `58`              |

Decode with: `struct.unpack("<4sIQQ", header[:24])`.

### 3. KV keys the loader must read

- `general.architecture` = `qwen3_5moe`
- `general.file_type` = `Q1_0`
- `qwen3_5moe.block_count` = 92
- `qwen3_5moe.embedding_length` = 8192
- `qwen3_5moe.attention.head_count` = 64
- `qwen3_5moe.feed_forward_length` = 2048
- `qwen3_5moe.expert_count` = 512
- `qwen3_5moe.expert_used_count` = 10
- `qwen3_5moe.shared_expert.feed_forward_length` = 2048

All other KV pairs are loader-orthogonal metadata.

### 4. Tensor classification (the rule that breaks MoE loaders)

Qwen3.8-2.4T-A95B is a 512-expert MoE. Every routed expert has its own
weight tensor; the tensor *name* tells you which one. The reference
classifier in `tools/inventory_gguf.py::classify_tensor` is:

1. `.shared_expert.` → `shared_expert` bucket.
2. `.experts\.(\d+)\.` → `routed_expert` bucket, expert_id = N.
3. Otherwise → `dense_shared` bucket.

**Test cases that must pass** (5 in `tools/test_inventory_gguf.py::TestClassifyTensor`):

- `blk.0.attn_q.weight` → `("dense_shared", None)`
- `blk.5.ffn.shared_expert.w1.weight` → `("shared_expert", None)`
- `blk.5.ffn.experts.42.w1.weight` → `("routed_expert", 42)`
- `blk.5.ffn.experts_4.w1.weight` (legacy merged) → `("routed_expert", 4)`
- `token_embd.weight` → `("dense_shared", None)`

If your loader disagrees with any of these, your loader is wrong.

### 5. Memory-plan v2 schema (for "will it fit?")

The Greenboost `qwen38-gguf-memory-plan.json` artefact answers "will
this shard fit in this VRAM/RAM profile?" The schema is
`qwen38.memory_plan.v2`. Fields:

```json
{
  "schema_version": "qwen38.memory_plan.v2",
  "provenance": { "config_sha256", "model_type", "quantization", "shard_path" },
  "assumptions": { "vram_budget_gb", "ram_arena_gb" },
  "buckets": { "dense_shared_bytes", "shared_expert_bytes", "routed_expert_bytes" },
  "fits": <bool>,
  "fallback_strategy": "ram_only" | "vram_partial" | "fits"
}
```

If your consumer needs to make a "will it fit?" decision locally,
ship the same 7 fields. The `fits` rule: `routed_expert_bytes ≤
vram_budget_gb × 2^30`.

### 6. Currently-expected numbers (Q1_0, 32 GB VRAM, 96 GB RAM)

- Total tensors per shard: 94,947 (when data-bearing shards land)
- `dense_shared`: 555 tensors, 4.41 GiB
- `shared_expert`: 184 tensors, 3.59 GiB
- `routed_expert`: 94,208 tensors, **575.00 GiB** ← the bottleneck
- Verdict: does not fit in 32 GB VRAM, fallback = `ram_only`

The routed_expert footprint dominates because 512 experts × 4 tensors ×
92 layers × Q1_0 size is a lot of bytes. This is the architectural
reason Qwen3.8-2.4T needs either a much larger GPU, an MoE-aware
offloader, or a router that doesn't activate all 10 experts per token.

## Known limitation: the shard is a metadata-only stub

The on-disk shard
`checkpoints/Qwen3.8-2.4T-A95B-GGUF-UD-Q1_0/UD-Q1_0/Qwen3.8-2.4T-A95B-UD-Q1_0-00001-of-00010.gguf`
is 10 MB, GGUF v3, with **0 tensors**. It validates the header, the
KV reader, and the config-driven layout synthesis, but not a real
tensor table. When a data-bearing shard becomes available:

1. Re-run `python tools/inventory_gguf.py` from the Greenboost repo root.
2. The 4th regression test class (`TestParseRealTensorTable`) is the
   one to add — it should round-trip 10 known tensor names and
   assert that `tensor_count` matches the header.
3. The `qwen38-gguf-layout.json` will then list real tensor names
   instead of config-synthesised placeholders.

## How to verify your loader is Greenboost-compatible

From the Greenboost repo root:

```bash
# Regenerate the canonical artefacts
python tools/inventory_gguf.py

# Run the 15 regression tests (all must pass)
python tools/test_inventory_gguf.py

# Cross-check: does the loader's tensor list match the layout's
# `tensors` field? If yes, you're integrated correctly.
python -c "
import json
layout = json.load(open('artifacts/qwen38_gguf/qwen38-gguf-layout.json'))
print(f'Greenboost expects {layout[\"tensor_counts\"][\"total\"]} tensors')
print(f'  dense_shared: {layout[\"tensor_counts\"][\"dense_shared\"]}')
print(f'  shared_expert: {layout[\"tensor_counts\"][\"shared_expert\"]}')
print(f'  routed_expert: {layout[\"tensor_counts\"][\"routed_expert\"]}')
"
```

## Contact / questions

If anything in this contract is wrong, or if `lmstudiosupport` finds a
behaviour that disagrees with the artefacts on disk, please open a
Greenboost issue with the specific section number and the conflicting
behaviour. We will update the contract in
`docs/architecture/GGUF_VALIDATION_HARDENING.md` (the authoritative
version of this note).
