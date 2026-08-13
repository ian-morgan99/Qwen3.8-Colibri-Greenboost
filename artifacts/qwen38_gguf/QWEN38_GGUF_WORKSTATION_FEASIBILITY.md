# Qwen3.8 GGUF Q1_0 Physical Inventory

Generated from: `/home/beast/Documents/VSCode/Qwen3.8/checkpoints/Qwen3.8-2.4T-A95B-GGUF-UD-Q1_0/UD-Q1_0/Qwen3.8-2.4T-A95B-UD-Q1_0-00001-of-00010.gguf`

## Inventory

| Measurement | Result |
|---|---:|
| Layers | 92 |
| MoE layers | 92 |
| Experts observed in tensor names | 0 |
| Experts activated per token | 10 |
| Dense/shared tensors (Q1_0 est.) | 0.00 GiB |
| Routed expert tensors (Q1_0 est.) | 0.00 GiB |
| Total tensor storage (Q1_0 est.) | 0.00 GiB |

## Initial placement model

With the default 32 GiB VRAM and 96 GiB RAM planning assumptions:

- Dense/shared resident footprint (Q1_0 est.): **0.00 GiB**
- Average routed expert size (Q1_0 est.): **0.00 GiB**
- GPU expert capacity after dense weights: **32.00 GiB**
- Estimated GPU experts fitting: **unknown**
- Estimated RAM experts fitting: **unknown**
- Cold active-expert NVMe traffic per token: **unknown seconds at 7 GiB/s sequential read rate**

These are planning estimates based on Q1_0 quantization (~1.56 bits/weight).
