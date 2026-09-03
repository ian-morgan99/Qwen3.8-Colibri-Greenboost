# Qwen3.8 GGUF Q1_0 Physical Inventory

Generated from: `checkpoints/Qwen3.8-2.4T-A95B-GGUF-UD-Q1_0/UD-Q1_0/Qwen3.8-2.4T-A95B-UD-Q1_0-00001-of-00010.gguf`

Schema version: `qwen38.memory_plan.v2`

## Inventory

| Measurement | Result |
|---|---:|
| Layers | 92 |
| MoE layers | 92 |
| Experts observed in tensor names | 512 |
| Experts activated per token | 10 |
| Dense tensors (Q1_0 est.) | 4.41 GiB |
| Shared expert tensors (Q1_0 est.) | 3.59 GiB |
| Routed expert tensors (Q1_0 est.) | 575.00 GiB |
| Total tensor storage (Q1_0 est.) | 583.00 GiB |

## Initial placement model

With the default 32 GiB VRAM and 96 GiB RAM planning assumptions:

- Dense + shared-resident footprint (Q1_0 est.): **8.00 GiB**
- Average routed expert size (Q1_0 est.): **1.12 GiB**
- GPU expert capacity after dense weights: **24.00 GiB**
- Estimated GPU experts fitting: **21**
- Estimated RAM experts fitting: **85**
- Cold active-expert NVMe traffic per token: **1.6043526785714286 seconds at 7.0 GiB/s sequential read rate**

These are planning estimates based on Q1_0 quantization (~1.56 bits/weight).
