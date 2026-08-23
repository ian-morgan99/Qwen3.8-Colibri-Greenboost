# Qwen3.8-2.4T-A95B Model Survey & FreeToken Compatibility

Date: 2026-08-23. Companion to `CRITICAL_EVAL_STACK.md`.

## Candidate checkpoints (all report `architectures: ["Qwen3_5MoeForCausalLM"]`, `model_type: qwen3_5_moe_text`)

| Repo | Format | Size | Notes |
|---|---|---|---|
| RedHatAI/Qwen3.8-2.4T-A95B-NVFP4 | compressed-tensors NVFP4 W4A16 | ~1425 GB | Best format match; ~5.5 days at 3 MB/s |
| mgoin/Qwen3.8-2.4T-A95B-pruned75-NVFP4 | NVFP4, 75% experts pruned | ~436 GB | ~1.7 days; quality tradeoff |
| mgoin/Qwen3.8-2.4T-A95B-pruned94-NVFP4 | NVFP4, 94% pruned | ~183 GB | Demo-scale only |
| Qwen official FP8 | FP8 | ~2496 GB | Too large for our WAN |
| RadixArk / Inferact / modal-labs NVFP4 | NVFP4 variants | similar | Same arch name |

## The blocker (now fixed)

`freetoken.models.register.get_model_spec("Qwen3_5MoeForCausalLM")` raised
`ValueError: Model architecture not supported` — the registry only had
`Qwen3_5MoeForConditionalGeneration` and `Qwen3_5ForConditionalGeneration`.
The mismatch is universal across every viable 2.4T checkpoint, and
transformers `AutoConfig` does not normalize the arch name.

**Fix (commit ecee0aa in /tmp/freetoken):** added to `_MODEL_REGISTRY`:

```python
"Qwen3_5MoeForCausalLM": ModelSpec(
    "freetoken.models.qwen3_5_moe",
    "Qwen3_5MoEForCausalLM",
),
```

Additive only; reuses the existing implementation. Verified:
- `get_model_spec("Qwen3_5MoeForCausalLM")` resolves to the correct ModelSpec
- class import `freetoken.models.qwen3_5_moe.Qwen3_5MoEForCausalLM` succeeds
- `tests/models/test_models_registry.py` passes

## Verified compatibility details

- **parse_config** (`qwen3_5_moe/config.py`): `text = getattr(hf_config, "text_config", hf_config)`
  handles the flat RedHatAI config. Full parse succeeds: 92 layers, 512 experts,
  top-10, hidden 8192, moe_intermediate 2048, GQA 64/4.
- **NVFP4 loader** (`models/weight.py` L538-541, L617-638): compressed-tensors
  detection requires `quant_method=="compressed-tensors"`, num_bits=4, type=float,
  group_size=16, strategy=tensor_group — matches RedHatAI config. MXFP4 (group 32) rejected.
- **Tensor layout**: `weight_packed` uint8 `[O, IN//2]` + `weight_scale` fp8-e4m3
  `[O, IN//16]` + `weight_global_scale` scalar (+ `input_global_scale`) — matches loader.
- **AOT kernels** (`kernel/aot_models.py`): `architecture`/`arch_aliases` fields are
  shape-table metadata only, NOT dispatch gates. The registry is the sole gate.
  No 2.4T entry → JIT compile required for the 2.4T model;
  `nvidia/Qwen3.6-35B-A3B-NVFP4` (23.4 GB) has prebuilt kernels and is the smoke target.

## Next steps

1. Real end-to-end test: `ft serve` + `tools/ftsmoke.py` against
   nvidia/Qwen3.6-35B-A3B-NVFP4 (needs VRAM freed).
2. Download decision: full RedHatAI NVFP4 (1425 GB, ~5.5 d) vs mgoin pruned75
   (436 GB, ~1.7 d) vs defer until GGUF finishes.
3. Build freetoken-kernel-cache wheel (CUDA 13 nvcc or docker).
