# Qwen3.8 Colibri Greenboost

This repository starts with the Phase 1 checkpoint inventory described in issue #1.
The inventory is intentionally independent of the inference runtime: it measures the
released checkpoint before any vLLM, quantisation, or routing changes are introduced.

## Architecture inventory

Run the inventory against a local Hugging Face checkpoint directory:

```bash
python3 tools/inventory_checkpoint.py \
  --checkpoint /path/to/Qwen3.8
```

The checkpoint directory must contain `config.json` and either
`model.safetensors.index.json` or one or more `*.safetensors` files. The command
writes to canonical paths (no flags needed unless you want to override):

- `artifacts/qwen38-layout.json` — Phase 0 layout (canonical)
- `artifacts/qwen38-memory-plan.json` — Phase 0 memory plan (canonical)
- `docs/QWEN38_WORKSTATION_FEASIBILITY.md` — Phase 0 feasibility narrative (canonical)

The duplicate-namespace guard in `tools/test_no_duplicate_artifacts.py` fails
if any of those basenames appears at more than one location under the repo.

The generated report records whether estimates are based on tensor metadata or
actual safetensors headers. It does not download model weights or alter the
checkpoint.

## Scope

This first implementation is an exact inventory and planning tool. It does not
implement expert execution, expert substitution, approximate routing, or model
serving yet.