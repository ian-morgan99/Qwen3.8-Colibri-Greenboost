# Qwen3.8 Architecture (Checkpoint-Derived Facts)

**Status:** Authoritative. Read by `tools/qwen38_config.py` and every
downstream tool that needs layer / expert / per-expert byte counts.

**Source of truth:** `checkpoints/Qwen3.8-2.4T-A95B/config.json`
- **stable sha256:** `89391ac8f44227959cb4b89df5c94d0b78d5686bc102988ca2ca4447fc4b84f1`
  (drops the `transformers_version` field so the hash tracks the
  architecture rather than the writer library)
- **model_type:** `qwen3_5_moe_text`
- **architectures:** `["Qwen3_5MoeForCausalLM"]`

If the upstream `config.json` ever changes, rerun
`python3 tools/qwen38_config.py --json > artifacts/qwen38-config.facts.json`
and the stable hash will be updated. Every artefact that cites a fact from
this document must include the stable hash so reviewers can detect drift.

## Top-line architecture

| Field | Value | Notes |
|------|------|------|
| `num_hidden_layers` | 92 | Body transformer |
| `mtp_num_hidden_layers` | 1 | Multi-Token Prediction head |
| `num_experts` | 512 | Routed (MoE) experts per layer |
| `num_experts_per_tok` | 10 | Top-10 routing |
| `hidden_size` | 8192 | Token embedding width |
| `moe_intermediate_size` | 2048 | Per-expert FFN width |
| `shared_expert_intermediate_size` | 2048 | Per-layer always-on expert |
| `vocab_size` | 248,320 | Standalone LM head (not tied) |
| `num_attention_heads` | 64 | Full-attention layers |
| `num_key_value_heads` | 4 | GQA |
| `head_dim` | 256 | |
| `full_attention_interval` | 4 | Layer i is full attention iff `i % 4 == 0` |
| `max_position_embeddings` | 262,144 | RoPE theta 10000 |

**Layer mix:** 23 full-attention layers, 69 linear-attention (DeltaNet)
layers, 1 MTP layer.

## Linear-attention (DeltaNet / Mamba2) knobs

| Field | Value |
|------|------|
| `linear_conv_kernel_dim` | 4 |
| `linear_key_head_dim` | 128 |
| `linear_value_head_dim` | 128 |
| `linear_num_key_heads` | 16 |
| `linear_num_value_heads` | 128 |

## Storage layout (per layer)

The Qwen3.8 MoE design **packs all 512 experts into two tensors per layer**:

- `model.layers.N.mlp.experts.gate_up_proj` — shape `[hidden, 2*moe_int, num_experts]`,
  transposed in safetensors to `[num_experts, 2*moe_int, hidden]` (the
  safetensors index flattens this to `[num_experts * 2*moe_int * hidden]`).
- `model.layers.N.mlp.experts.down_proj` — shape `[num_experts, moe_int, hidden]`.

This is a per-tensor storage of 512 experts. Caching that treats "an
expert" as a unit must therefore evict at the per-expert granularity
(1/512 of a packed tensor) and re-evict individual experts; it is **not**
acceptable to evict an entire 7.7-GiB packed tensor on every miss.

The shared expert is **split into three tensors** (not fused):

- `model.layers.N.mlp.shared_expert.gate_proj.weight`
- `model.layers.N.mlp.shared_expert.up_proj.weight`
- `model.layers.N.mlp.shared_expert.down_proj.weight`

Plus a routing gate: `model.layers.N.mlp.gate.weight`
and a shared-expert gate: `model.layers.N.mlp.shared_expert_gate.weight`.

### Attention tensor layout

The body has 92 layers, of which 23 are full-attention and 69 are
linear-attention (driven by `full_attention_interval=4`).

- **Full-attention layer** (23 layers): 6 tensors — `q_proj`, `k_proj`,
  `v_proj`, `o_proj`, `k_norm.weight`, `q_norm.weight`.
- **Linear-attention layer** (69 layers): 9 tensors — `A_log`,
  `conv1d.weight`, `dt_bias`, `in_proj_a.weight`, `in_proj_b.weight`,
  `in_proj_qkv.weight`, `in_proj_z.weight`, `out_proj.weight`, and
  `norm.weight` (the additional 9th tensor is the layer-level RMSNorm
  used inside the linear-attention block).

A separate MTP layer (`mtp.layers.0`) follows the full-attention layout
(6 self_attn tensors) and adds its own `mlp.experts.{gate_up_proj,
down_proj}` (packed), `mlp.shared_expert.{gate,up,down}_proj`,
`mlp.shared_expert_gate.weight`, `mlp.gate.weight`, and the bare
`mtp.fc.weight`, `mtp.norm.weight`, `mtp.pre_fc_norm_embedding.weight`,
`mtp.pre_fc_norm_hidden.weight`.

## Per-expert byte math (BF16 reference)

Per expert (the smallest routing unit):

```
gate_up_proj params:  hidden * 2 * moe_int  = 8192 * 2 * 2048 = 33,554,432
down_proj    params:  moe_int * hidden      = 2048  * 8192 = 16,777,216
total params per expert:                    50,331,648
BF16 bytes per expert:                      100,663,296  (~96.00 MiB)
```

Per layer (512 experts, BF16):

```
512 * 100,663,296 = 51,539,607,552 bytes  (~48.00 GiB)
```

Active per token (10 of 512 experts per layer, 92 layers):

```
92 layers * 10 experts * 50,331,648 params = 46,305,116,160 params
BF16 bytes:                                92,610,232,320  (~86.25 GiB)
```

For Q1_0 GGUF (≈2.56 bits/weight) the same numbers scale to:

```
bytes per expert (Q1_0):   ~15.4 MiB
bytes per layer (Q1_0):    ~7.69 GiB
active per token (Q1_0):   ~13.81 GiB
```

**Important correction:** the previous `tools/trace_qwen38_routing.py`
hard-coded a 1.45 GiB-per-expert constant. That number is wrong by
roughly an order of magnitude. At BF16 a single expert is 96 MiB, not
1.45 GiB. The 1.45 GiB figure appears to have been the result of
confusing the per-layer packed tensor with a per-expert figure. Any
prior memory-plan numbers derived from that constant should be
re-examined; `tools/qwen38_config.py` exposes the correct math.

## Tensor counts (from `model.safetensors.index.json`)

| Category | Count | Notes |
|----------|------:|------|
| Embed + final norm | 3 | `embed_tokens.weight`, final norms |
| Full-attention `self_attn.*` | 144 | 6 tensors per full-attn layer × 23 + lm_head |
| Linear-attention `linear_attn.*` | 621 | 9 tensors per linear layer × 69 |
| Routed expert tensors `mlp.experts.*` | 186 | 184 body + 2 MTP |
| Shared expert tensors `mlp.shared_expert.*` | 279 | 276 body + 3 MTP |
| Routing gates | 186 | 92 + 1 MTP × 2 (gate, shared_expert_gate) |
| Other norms | 92 | input_layernorm + post_attention_layernorm per layer |
| MTP (other) | 3 | `mtp.fc`, `mtp.norm`, `mtp.pre_fc_norm_*` |
| LM head | 1 | `lm_head.weight` |
| **Total tensor refs** | **1,609** | matches index length |

## Provenance rules

Any tool that emits numbers derived from this document MUST embed:

1. `config_sha256` (the stable hash above)
2. `config_path` (absolute path to the checkpoint config)
3. `tool_name` and `tool_version` (or git commit of the tool that produced the numbers)
4. A clear `data_classification` of one of:
   - `checkpoint_derived` — directly read from config.json
   - `computed_from_checkpoint` — arithmetic on checkpoint fields
   - `synthetic` — placeholder until a real captured trace exists
   - `captured` — real measurements from a real model run

Synthetic outputs MUST NOT be labelled `captured`. The first time a real
trace is captured from a running Qwen3.8, this document will gain a
"Captured traces" section with the trace source, prompt set, capture
method, and reproduction instructions.
