# FreeToken Tuning Policy & Performance Matrix

## Policy

Whenever we add or discover a new `ft serve` option (a flag, backend choice, or
environment knob), we MUST do two things:

1. **Add it to the GUI launcher** — `tools/ft-gui-launch.sh` (zenity form), so it
   is selectable at start without editing scripts.
2. **Benchmark it and record the result in the matrix below**, so future choices
   are data-driven rather than guessed.

Benchmark procedure: run `tools/ftsmoke.py` against the running server, plus a
fixed 256-token generation prompt, and record decode tok/s, TTFT, and VRAM.

## Performance Matrix — Huihui Qwen3.8-27B-abliterated NVFP4 (RTX 5090)

| Date       | nvfp4-backend | attention | ctx      | max-req | decode tok/s | TTFT  | VRAM   | Notes                    |
|------------|---------------|-----------|----------|---------|--------------|-------|--------|--------------------------|
| 2026-08-23 | triton        | fi (auto) | 262144   | 4       | ~65          | ~0s   | 28 GiB | baseline; smoke test PASS |

(Add a row for every new option tested. Keep the best-performing config noted here.)

**Current best:** baseline row above.

## GUI options currently exposed

- NVFP4 backend: triton / flashinfer / marlin / auto
- Attention backend: auto (fi) / fi / fa3 / triton
- Port
- Context length: default (262k) / 8192 … 131072
- Max concurrent requests: 1 / 2 / 4 / 8

## Not yet exposed (dense model — likely irrelevant)

MoE flags (`--moe-cache-*`, `--expert-load`, `--moe-cpu-layers`) apply to MoE
models only; the huihui 27B is dense. Revisit when serving the 2.4T MoE.
