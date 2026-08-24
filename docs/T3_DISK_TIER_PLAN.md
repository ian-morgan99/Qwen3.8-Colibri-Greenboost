# T3 / NVMe Tier — Detailed Implementation Plan

Status: **planned, not started**. Blocked on Option B (llama.cpp mmap
proof-of-life) giving a tok/s number that justifies the effort. All work in a
**separate clone + venv**; production venv untouched until patches proven.

## Goal

Make FreeToken serve models whose expert banks exceed host RAM by keeping
expert banks **file-backed and pageable**, reading expert rows from NVMe on
demand. Target: Qwen3.8-2.4T UD-Q1_0 (397 GB) on 32 GB VRAM + 91 GB RAM +
3.7 TB NVMe at 0.5–3 tok/s.

## Key architectural discovery (changes the original plan)

The CPU executor (`moe/cpu_executor.py`) reads expert banks via **raw host
pointers** (`_resolve_banks` → `_make_table(...).data_ptr()`), with **no
pinned-memory requirement**. The hybrid backend already splits each decode
step's misses between PCIe fetch (GPU) and CPU compute (overflow). This means:

> **A disk-tier layer can be served entirely by the existing CPU executor
> path — no new GPU movement paths needed.** The C++ GEMV kernel just reads
> through the pointer table; if the underlying pages are file-backed mmap,
> the kernel's touch faults them in from NVMe transparently.

This collapses the project from "surgery on hot GPU paths" to:

1. file-backed HostBank variant (no `cudaHostRegister`),
2. accept non-pinned layers in `set_bank_sources` when they're routed to CPU,
3. let the kernel page cache be the tier.

## Design

### New flag

```
--expert-disk-tier {off,auto,all}   default: off
--expert-disk-budget <GB>           soft RAM cap for page cache of bank regions
```

- `auto`: layers whose banks don't fit free RAM become disk-tier; rest stay pinned.
- `all`: every expert layer disk-tiered (max capacity, slowest).
- Budget enforced with periodic `madvise(MADV_DONTNEED)` on cold bank windows
  (the call already exists in `host_banks.py::release`).

### Component changes (upstream flashml-org/freetoken)

| # | File | Change | Est. size |
|---|------|--------|-----------|
| 1 | `moe/host_banks.py` | `HostBank(shape, dtype, path=None, offset=0)`: when `path` given, `mmap.mmap(fd, length, offset=aligned_off)` instead of anonymous; skip pin(); residency = new `HostResidency.DISK`. Add `advise_dontneed()` per bank. | ~40 lines |
| 2 | `checkpoint/ftw.py` | `load_ftw_banks(..., disk_tier=False)`: per-layer aligned entries map directly to their FTW shard region (per-layer layout is ALIGN-aligned by writer invariant — verified). Alphas stay pinned (tiny, GPU-resident anyway). Flat-layout entries need window mapping like today but without read_into. | ~60 lines |
| 3 | `moe/expert_banks.py` | Thread `disk_tier` through `load_expert_banks`; set `ExpertBanks.layer_residency[i] = "disk"` for disk layers. | ~20 lines |
| 4 | `moe/offload_cache.py` | Relax the `set_bank_sources` NotImplementedError: allow `"disk"` residency iff those layer ids ⊆ `cpu_layer_ids`. Disk layers never enter `_copy_src_ptrs`/copy plan. | ~15 lines |
| 5 | `engine/engine.py` | When `disk_tier != off`: force `decode_target="hybrid"` semantics for disk layers (CPU executor required); auto-set `--moe-cpu-layers` to cover all disk layers if not user-specified; wire budget advisor thread. | ~30 lines |
| 6 | `server/args.py` + `engine/config.py` | New flags. | ~15 lines |
| 7 | NEW `moe/disk_budget.py` | Background thread: watch `/proc/meminfo` MemAvailable; when under budget watermark, `MADV_DONTNEED` the least-recently-touched bank windows (track touches via `collect_decode_freq`, which already exists). | ~80 lines |

Total ≈ 260 lines across 7 files. No kernel/C++ changes.

### Why this works performance-wise

- Decode miss on a disk layer → CPU GEMV touches packed q4_0/nvfp4 rows →
  page fault → NVMe read (~4 GB/s Gen4, or ~7 GB/s cached).
- Per-token active bytes ≈ 12–18 GB spread over 64 layers ⇒ ~200–280 MB per
  layer per token. At even 1 GB/s effective NVMe throughput that's ~0.2 s/layer
  worst case... mitigated by:
  - **routing locality**: heavy-tailed expert popularity means the page cache
    holds hot experts well (this is why `auto` beats `all`);
  - **prefetch**: N+1 layer's likely experts can be `madvise(MADV_WILLNEED)`'d
    during layer N's compute (follow-up optimization);
  - **Q1_0 packing**: 1-bit weights mean each expert row is tiny — faults are
    small and frequent rather than large and stalling.

### Risks / open questions

1. **CUDA graph capture**: CPU executor decode is graph-captured. Page faults
   inside a captured host node are fine (they're just slow host memory reads),
   but first-capture will fault the whole touched working set — capture may
   take minutes on a cold cache. Mitigation: optional `--warmup-steps`.
2. **memlock**: file-backed mmaps we DON'T register bypass RLIMIT_MEMLOCK
   entirely — actually safer than today's pin path.
3. **Windows**: MADV_DONTNEED unavailable; budget thread degrades to no-op
   (page cache managed by OS). Fine.
4. **FTW conversion of the 397 GB GGUF** takes ~1–2 h disk-to-disk and needs
   400+ GB free (have 2.6 TB). Alternatively teach the GGUF loader path
   (`q4_0` native banks) to mmap — bigger change, phase 2.

## Execution order

1. **Phase 0 (now)**: Option B llama.cpp number → go/no-go.
2. **Phase 1**: clone upstream repo, separate venv, implement items 1–6,
   unit-test with a tiny synthetic FTW checkpoint (fabricate 4-layer model,
   verify: banks file-backed, inference correct vs pinned baseline, RSS stays low).
3. **Phase 2**: item 7 (budget thread) + `ft bench bw`-style microbenchmark
   of fault throughput on real NVMe.
4. **Phase 3**: convert 2.4T GGUF → FTW, run end-to-end, measure tok/s,
   write perf matrix row, submit patch series upstream.

## Success criteria

- Synthetic test: identical outputs pinned vs disk-tier; peak RSS < bank size.
- Real test: 2.4T loads and generates ≥0.5 tok/s sustained.
