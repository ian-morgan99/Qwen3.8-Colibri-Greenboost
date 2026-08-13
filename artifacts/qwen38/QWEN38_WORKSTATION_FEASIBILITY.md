# Qwen3.8 workstation feasibility

Generated from: `simulated_checkpoint_metadata`

## Inventory

| Measurement | Result |
|---|---:|
| Layers | 60 |
| MoE layers | unknown |
| Experts observed in tensor names | 128 |
| Experts activated per token | 8 |
| Dense/shared tensors | 293.00 GiB |
| Shared expert tensors | 14.65 GiB |
| Routed expert tensors | 167.12 GiB |
| Total tensor storage | 460.77 GiB |

## Initial placement model

With the default 32 GiB VRAM and 96 GiB RAM planning assumptions:

- Dense/shared resident footprint: **293.00 GiB**
- Average routed expert size: **1.64 GiB**
- GPU expert capacity after dense weights: **0.00 GiB**
- Estimated GPU experts fitting: **0**
- Estimated RAM experts fitting: **54**
- Cold active-expert NVMe traffic per token: **1.036 seconds at the configured sequential read rate**

These are planning estimates. The report deliberately leaves MoE-layer count and
runtime hit rates unknown when the checkpoint does not expose enough metadata.

## Limitations and gate

- moe_layers is unknown unless represented explicitly in config or tensor names
- experts_activated_per_token is copied from config and is not inferred from generated tokens

This inventory does not establish inference correctness or RTX 5090 kernel support.
Those remain Phase 1 gate items before implementing the tiered runtime.
