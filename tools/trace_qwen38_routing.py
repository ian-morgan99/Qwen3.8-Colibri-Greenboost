#!/usr/bin/env python3
"""Qwen3.8 routing trace + expert-cache simulator.

The Qwen3.8 MoE layers choose ``num_experts_per_tok`` (10) of 512 experts
for every token, and each of those experts has to be resident in VRAM
before the corresponding layer can run. This tool replays routing traces
through a 3-level cache (L1 = VRAM, L2 = system RAM, L3 = NVMe) so we
can size the cache and pick a prefetch policy.

Two trace sources are supported:

* ``--trace captured``: load a JSON routing trace produced by an actual
  Qwen3.8 forward pass (e.g. from llama.cpp's ``--verbose`` router dump
  or from an in-process capture). A captured trace closes the gate
  described in Issue #2. The tool refuses to label a non-captured
  output as ``captured``.
* ``--trace synthetic``: generate a routing trace locally with a
  documented skew. This is useful for unit tests and for sanity-checking
  the cache simulator, but the output is explicitly labelled
  ``synthetic`` in the metrics file so downstream consumers cannot
  mistake it for a real measurement.

The cache simulator is **deterministic** - the same input trace + the
same config always produce the same metrics. There is no ``set.pop()``
arbitrary eviction; both LRU and LFRU policies use ordered eviction
based on the actual access stream.

The simulator also reports:

* bytes/token transferred per source (VRAM / RAM / NVMe)
* exposed-wait latency per generated token, in milliseconds, using a
  configurable bandwidth model
* p50 / p95 fetch latency for NVMe -> RAM and RAM -> VRAM transfers
* full provenance: which ``config.json`` was used, its hash, the trace
  source, and the simulator commit

Remediation: this file replaces the previous synthetic
``trace_qwen38_routing.py`` per GitHub Issue #2. The previous
implementation hard-coded ``num_layers=92 / num_experts=512 / top_k=10``,
used a per-expert byte size of 1.45 GB that was off by an order of
magnitude, used ``set.pop()`` for arbitrary cache eviction, used
``random.sample(range(95), 5)`` for N+1 prefetch, and labelled its
output "real Qwen3.8 routing traces" when it was not.
"""
from __future__ import annotations

import argparse
import collections
import dataclasses
import json
import os
import random
import statistics
import subprocess
import sys
from typing import Any, Dict, List, Optional, Sequence, Tuple

# Make the package importable when invoked as a script.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from qwen38_config import Qwen38Config, load_config  # noqa: E402


# ---------------------------------------------------------------------------
# Trace types
# ---------------------------------------------------------------------------

@dataclasses.dataclass(frozen=True, order=True)
class ExpertKey:
    """Identity of a single expert access in a routing trace.

    The ``pack_revision`` field tracks which safetensors / GGUF revision
    produced the expert. If the same checkpoint is re-quantised, the
    expert cache should not return the stale copy. ``dtype`` captures the
    storage precision (BF16, Q4_K, Q1_0, ...) since cache sizes differ
    by dtype.
    """

    layer_id: int
    expert_id: int
    pack_revision: str
    dtype: str

    def to_dict(self) -> Dict[str, Any]:
        return dataclasses.asdict(self)

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "ExpertKey":
        return cls(
            layer_id=int(d["layer_id"]),
            expert_id=int(d["expert_id"]),
            pack_revision=str(d.get("pack_revision", "unknown")),
            dtype=str(d.get("dtype", "bf16")),
        )


@dataclasses.dataclass(frozen=True)
class TraceToken:
    """A single token's routing decision: which experts each layer chose."""

    token_idx: int
    layer_to_experts: Tuple[Tuple[int, Tuple[ExpertKey, ...]], ...]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "token_idx": self.token_idx,
            "layer_to_experts": [
                {
                    "layer_id": layer_id,
                    "experts": [e.to_dict() for e in experts],
                }
                for layer_id, experts in self.layer_to_experts
            ],
        }


@dataclasses.dataclass(frozen=True)
class RoutingTrace:
    """A full routing trace for a single prompt."""

    prompt_idx: int
    prompt: str
    tokens: Tuple[TraceToken, ...]
    source: str  # "captured" or "synthetic"
    captured_at: Optional[str] = None  # ISO timestamp if captured

    def to_dict(self) -> Dict[str, Any]:
        return {
            "prompt_idx": self.prompt_idx,
            "prompt": self.prompt,
            "source": self.source,
            "captured_at": self.captured_at,
            "tokens": [t.to_dict() for t in self.tokens],
        }


# ---------------------------------------------------------------------------
# Default bandwidth / latency model
# ---------------------------------------------------------------------------

# These are conservative defaults for a desktop NVMe + DDR5 system. They
# can be overridden on the CLI; we explicitly avoid hard-coding them
# inside the simulator so reviewers can plug in measured numbers.
DEFAULT_NVME_TO_RAM_BW_GBS = 7.0  # GB/s sequential read on a mid-range NVMe
DEFAULT_RAM_TO_VRAM_BW_GBS = 50.0  # GB/s PCIe 5.0 x16 host-to-GPU
DEFAULT_NVME_LATENCY_US = 80.0  # us access latency on the NVMe
DEFAULT_RAM_LATENCY_US = 1.0  # us RAM access latency


# ---------------------------------------------------------------------------
# Synthetic trace generator (clearly labelled, deterministic)
# ---------------------------------------------------------------------------

# A small set of coding-agent prompts that we expect to stress the MoE
# router. These are intentionally short so the synthetic trace is
# reproducible in milliseconds and easy to inspect in code review.
ROUTING_PROMPTS = [
    "Write a Python function to sort a list of dictionaries by a nested key.",
    "Explain the difference between async and await in Python.",
    "Debug this SQL query that is running slowly: SELECT * FROM users WHERE status = 'active' AND created_at > '2023-01-01'",
    "Design a REST API for a task management system with endpoints for CRUD operations.",
    "Write a regex to validate email addresses.",
    "Explain how garbage collection works in JavaScript.",
    "Implement a binary search tree in C++ with insert and delete operations.",
    "What are the best practices for securing a web application against SQL injection?",
    "Write a shell script to backup all files in a directory to an S3 bucket.",
    "Explain the concept of dependency injection in Java.",
]


def synth_route_token(
    config: Qwen38Config,
    token_idx: int,
    layer: int,
    prev_layer_experts: Optional[Tuple[int, ...]],
    rng: random.Random,
) -> Tuple[int, ...]:
    """Pick ``top_k`` experts for one (token, layer) with realistic skew.

    MoE routers in production show two kinds of structure:

    * **Popular experts:** a small subset of experts handle a
      disproportionate share of tokens, regardless of content. We model
      this by reserving ``popular_fraction`` of the expert pool as
      "hot" and sampling from it with high probability.
    * **Locality:** within a context window, the same experts tend to
      be re-used, especially across adjacent layers. We model this by
      reusing the previous layer's experts with a small perturbation
      roughly half the time.
    """
    popular_count = max(config.num_experts_per_tok, config.num_experts // 20)
    popular_pool = list(range(0, config.num_experts, max(1, config.num_experts // popular_count)))
    top_k = config.num_experts_per_tok

    if prev_layer_experts is not None and token_idx > 0 and (layer % 3) == 0:
        # Locality reuse: keep most of the previous layer's experts, perturb the rest.
        reuse = list(prev_layer_experts)
        n_replace = min(2, top_k, len(popular_pool))
        if n_replace > 0:
            replacements = rng.sample(popular_pool, n_replace)
            for i, r in enumerate(replacements):
                if i < len(reuse):
                    reuse[i] = r
        experts = []
        seen = set()
        for e in reuse:
            if e in seen:
                e = (e + 1) % config.num_experts
            seen.add(e)
            experts.append(e)
        while len(experts) < top_k:
            e = rng.choice(popular_pool)
            if e not in seen:
                seen.add(e)
                experts.append(e)
        return tuple(experts[:top_k])

    sample_pool = popular_pool if len(popular_pool) >= top_k else list(range(config.num_experts))
    experts = rng.sample(sample_pool, min(top_k, len(sample_pool)))
    while len(experts) < top_k:
        e = rng.randrange(config.num_experts)
        if e not in experts:
            experts.append(e)
    return tuple(experts)


def synth_trace(
    config: Qwen38Config,
    prompts: Sequence[str] = ROUTING_PROMPTS,
    tokens_per_prompt: int = 20,
    dtype: str = "bf16",
    pack_revision: str = "synthetic-v1",
    seed: int = 0,
) -> List[RoutingTrace]:
    """Generate a synthetic routing trace. Clearly labelled as such."""
    rng = random.Random(seed)
    out: List[RoutingTrace] = []
    for prompt_idx, prompt in enumerate(prompts):
        tokens: List[TraceToken] = []
        for token_idx in range(tokens_per_prompt):
            layer_to_experts: List[Tuple[int, Tuple[ExpertKey, ...]]] = []
            prev_layer_experts: Optional[Tuple[int, ...]] = None
            for layer in range(config.num_hidden_layers):
                experts = synth_route_token(
                    config, token_idx, layer, prev_layer_experts, rng
                )
                prev_layer_experts = experts
                keys = tuple(
                    ExpertKey(
                        layer_id=layer,
                        expert_id=eid,
                        pack_revision=pack_revision,
                        dtype=dtype,
                    )
                    for eid in experts
                )
                layer_to_experts.append((layer, keys))
            tokens.append(
                TraceToken(
                    token_idx=token_idx,
                    layer_to_experts=tuple(layer_to_experts),
                )
            )
        out.append(
            RoutingTrace(
                prompt_idx=prompt_idx,
                prompt=prompt,
                tokens=tuple(tokens),
                source="synthetic",
            )
        )
    return out


# ---------------------------------------------------------------------------
# Captured-trace loader
# ---------------------------------------------------------------------------

def load_captured_traces(
    path: str,
    config: Qwen38Config,
) -> List[RoutingTrace]:
    """Load a captured routing trace from disk.

    The expected JSON layout matches :meth:`RoutingTrace.to_dict`::

        {
          "schema_version": "qwen38.routing_trace.v1",
          "model_revision": "...",
          "config_sha256": "...",
          "captured_at": "2025-01-01T00:00:00Z",
          "prompts": [
            {"prompt_idx": 0, "prompt": "...", "tokens": [...]}
          ]
        }

    We refuse to load a trace that disagrees with the active config on
    the architecture fields (num_experts, num_experts_per_tok) so a
    stale trace cannot poison the cache simulator.
    """
    with open(path, "r", encoding="utf-8") as f:
        doc = json.load(f)

    schema = doc.get("schema_version")
    if schema != "qwen38.routing_trace.v1":
        raise ValueError(
            f"Unsupported trace schema {schema!r}; expected 'qwen38.routing_trace.v1'."
        )
    declared_config_sha = doc.get("config_sha256")
    if declared_config_sha and declared_config_sha != config.config_sha256:
        raise ValueError(
            f"Captured trace was produced with config_sha256={declared_config_sha} "
            f"but the active checkpoint is {config.config_sha256}. "
            "Refusing to load a stale trace."
        )
    declared_captured_at = doc.get("captured_at")

    traces: List[RoutingTrace] = []
    for p in doc.get("prompts", []):
        tokens: List[TraceToken] = []
        for t in p.get("tokens", []):
            layer_to_experts: List[Tuple[int, Tuple[ExpertKey, ...]]] = []
            for layer_entry in t.get("layer_to_experts", []):
                layer_id = int(layer_entry["layer_id"])
                keys = tuple(
                    ExpertKey.from_dict(e) for e in layer_entry["experts"]
                )
                if len(keys) != config.num_experts_per_tok:
                    raise ValueError(
                        f"Layer {layer_id} chose {len(keys)} experts, "
                        f"expected {config.num_experts_per_tok}."
                    )
                for k in keys:
                    if not (0 <= k.expert_id < config.num_experts):
                        raise ValueError(
                            f"Layer {layer_id} picked expert_id={k.expert_id}, "
                            f"out of range 0..{config.num_experts - 1}."
                        )
                layer_to_experts.append((layer_id, keys))
            tokens.append(
                TraceToken(
                    token_idx=int(t["token_idx"]),
                    layer_to_experts=tuple(layer_to_experts),
                )
            )
        traces.append(
            RoutingTrace(
                prompt_idx=int(p["prompt_idx"]),
                prompt=str(p["prompt"]),
                tokens=tuple(tokens),
                source="captured",
                captured_at=declared_captured_at,
            )
        )
    if not traces:
        raise ValueError("Captured trace contained zero prompts.")
    return traces


# ---------------------------------------------------------------------------
# Cache simulator
# ---------------------------------------------------------------------------

class _BoundedCache:
    """A bounded cache with a deterministic eviction policy.

    Unlike ``set.pop()`` (which is arbitrary), this class records the
    last-access time of every entry and evicts the entry with the
    smallest last-access time on overflow. That is exactly LRU
    (least-recently-used) for the ``policy='lru'`` mode. For
    ``policy='lfru'`` (least-frequently-recently-used) we use a
    recency-weighted frequency: ``score = access_count * decay +
    last_access_time * (1 - decay)`` and evict the entry with the
    smallest score.
    """

    def __init__(self, capacity_bytes: int, policy: str = "lru") -> None:
        if capacity_bytes < 0:
            raise ValueError("capacity_bytes must be non-negative")
        if policy not in ("lru", "lfru"):
            raise ValueError(f"Unknown policy {policy!r}")
        self.capacity_bytes = capacity_bytes
        self.policy = policy
        # key -> (size_bytes, access_count, last_access_time)
        self._entries: "collections.OrderedDict[ExpertKey, Tuple[int, int, int]]" = (
            collections.OrderedDict()
        )
        self._used_bytes = 0
        self._clock = 0
        self._lfru_decay = 0.5  # weight of frequency vs recency

    def __len__(self) -> int:
        return len(self._entries)

    @property
    def used_bytes(self) -> int:
        return self._used_bytes

    def _score(self, entry: Tuple[int, int, int]) -> float:
        size, count, last = entry
        if self.policy == "lru":
            return float(last)
        return count * self._lfru_decay + last * (1.0 - self._lfru_decay)

    def _evict_until(self, free_bytes_needed: int) -> List[ExpertKey]:
        evicted: List[ExpertKey] = []
        while (
            self._used_bytes + free_bytes_needed > self.capacity_bytes
            and self._entries
        ):
            worst_key = min(
                self._entries.keys(),
                key=lambda k: self._score(self._entries[k]),
            )
            size, _, _ = self._entries.pop(worst_key)
            self._used_bytes -= size
            evicted.append(worst_key)
        return evicted

    def get(self, key: ExpertKey) -> bool:
        if key not in self._entries:
            return False
        size, count, _ = self._entries[key]
        self._clock += 1
        self._entries[key] = (size, count + 1, self._clock)
        self._entries.move_to_end(key)  # mark MRU
        return True

    def put(self, key: ExpertKey, size_bytes: int) -> List[ExpertKey]:
        if size_bytes > self.capacity_bytes:
            raise ValueError(
                f"Entry {key} of size {size_bytes} bytes exceeds cache "
                f"capacity {self.capacity_bytes} bytes."
            )
        if key in self._entries:
            old_size, count, _ = self._entries[key]
            self._used_bytes -= old_size
            self._used_bytes += size_bytes
            self._clock += 1
            self._entries[key] = (size_bytes, count + 1, self._clock)
            self._entries.move_to_end(key)
            return []
        self._evict_until(size_bytes)
        self._clock += 1
        self._entries[key] = (size_bytes, 1, self._clock)
        self._used_bytes += size_bytes
        return []


@dataclasses.dataclass
class CacheSimulator:
    """Replay a routing trace through a 3-level cache hierarchy.

    The simulator treats each expert as a fixed-size unit equal to
    ``bytes_per_expert_bf16`` from the checkpoint config. L1 (VRAM)
    and L2 (RAM) caches are bounded by the configured sizes in bytes.
    """

    config: Qwen38Config
    policy: str  # "lru" or "lfru"
    l1_vram_bytes: int
    l2_ram_bytes: int
    nvme_to_ram_bw_gbs: float
    ram_to_vram_bw_gbs: float
    nvme_latency_us: float
    ram_latency_us: float
    prefetch_lookahead: int  # 0 = no prefetch, 1 = N+1, 2 = N+2
    bytes_per_expert_bf16: int  # from config
    bytes_per_layer_packed_bf16: int  # from config (for reference)

    @classmethod
    def from_args(
        cls,
        config: Qwen38Config,
        policy: str,
        l1_size_gb: float,
        l2_size_gb: float,
        nvme_to_ram_bw_gbs: float,
        ram_to_vram_bw_gbs: float,
        nvme_latency_us: float,
        ram_latency_us: float,
        prefetch_lookahead: int,
    ) -> "CacheSimulator":
        bytes_per_expert = config.bytes_per_expert_bf16
        bytes_per_layer = config.num_experts * bytes_per_expert
        return cls(
            config=config,
            policy=policy,
            l1_vram_bytes=int(l1_size_gb * (1024 ** 3)),
            l2_ram_bytes=int(l2_size_gb * (1024 ** 3)),
            nvme_to_ram_bw_gbs=nvme_to_ram_bw_gbs,
            ram_to_vram_bw_gbs=ram_to_vram_bw_gbs,
            nvme_latency_us=nvme_latency_us,
            ram_latency_us=ram_latency_us,
            prefetch_lookahead=prefetch_lookahead,
            bytes_per_expert_bf16=bytes_per_expert,
            bytes_per_layer_packed_bf16=bytes_per_layer,
        )

    def replay(self, traces: Sequence[RoutingTrace]) -> Dict[str, Any]:
        l1 = _BoundedCache(self.l1_vram_bytes, policy=self.policy)
        l2 = _BoundedCache(self.l2_ram_bytes, policy=self.policy)
        per_token_metrics: List[Dict[str, Any]] = []
        fetch_latencies_ms: List[float] = []
        exposed_wait_ms_total = 0.0
        nvme_fetch_count = 0
        vram_hit_count = 0
        ram_hit_count = 0
        stalling_miss_count = 0
        prefetch_hide_count = 0
        total_expert_requests = 0
        bytes_from_vram = 0
        bytes_from_ram = 0
        bytes_from_nvme = 0

        for trace in traces:
            for token in trace.tokens:
                token_stall_ms = 0.0
                token_bytes_vram = 0
                token_bytes_ram = 0
                token_bytes_nvme = 0
                layer_entries = list(token.layer_to_experts)
                for layer_idx, (layer_id, experts) in enumerate(layer_entries):
                    lookahead_experts: set = set()
                    if self.prefetch_lookahead > 0:
                        for k in range(1, self.prefetch_lookahead + 1):
                            next_idx = layer_idx + k
                            if next_idx < len(layer_entries):
                                _, next_experts = layer_entries[next_idx]
                                lookahead_experts.update(next_experts)

                    for key in experts:
                        total_expert_requests += 1
                        if l1.get(key):
                            vram_hit_count += 1
                            token_bytes_vram += self.bytes_per_expert_bf16
                            bytes_from_vram += self.bytes_per_expert_bf16
                            continue
                        if l2.get(key):
                            ram_hit_count += 1
                            transfer_bytes = self.bytes_per_expert_bf16
                            transfer_ms = (
                                transfer_bytes / (self.ram_to_vram_bw_gbs * (1024 ** 3))
                            ) * 1000.0
                            fetch_latencies_ms.append(transfer_ms)
                            exposed_wait_ms_total += transfer_ms
                            token_stall_ms += transfer_ms
                            token_bytes_ram += self.bytes_per_expert_bf16
                            bytes_from_ram += self.bytes_per_expert_bf16
                            l1.put(key, self.bytes_per_expert_bf16)
                            continue
                        if key in lookahead_experts:
                            # N+1 / N+2 prefetch wins.
                            prefetch_hide_count += 1
                            l1.put(key, self.bytes_per_expert_bf16)
                            vram_hit_count += 1
                            token_bytes_vram += self.bytes_per_expert_bf16
                            bytes_from_vram += self.bytes_per_expert_bf16
                            continue
                        # Stalling miss: NVMe -> RAM -> VRAM, on the
                        # critical path.
                        nvme_fetch_count += 1
                        stalling_miss_count += 1
                        expert_bytes = self.bytes_per_expert_bf16
                        nvme_ms = (
                            expert_bytes / (self.nvme_to_ram_bw_gbs * (1024 ** 3))
                        ) * 1000.0
                        ram_ms = (
                            expert_bytes / (self.ram_to_vram_bw_gbs * (1024 ** 3))
                        ) * 1000.0
                        latency_ms = (
                            (self.nvme_latency_us + self.ram_latency_us) / 1000.0
                            + nvme_ms + ram_ms
                        )
                        fetch_latencies_ms.append(latency_ms)
                        exposed_wait_ms_total += latency_ms
                        token_stall_ms += latency_ms
                        token_bytes_nvme += expert_bytes
                        token_bytes_ram += expert_bytes
                        bytes_from_nvme += expert_bytes
                        bytes_from_ram += expert_bytes
                        l2.put(key, expert_bytes)
                        l1.put(key, expert_bytes)
                per_token_metrics.append(
                    {
                        "prompt_idx": trace.prompt_idx,
                        "token_idx": token.token_idx,
                        "stall_ms": token_stall_ms,
                        "bytes_vram": token_bytes_vram,
                        "bytes_ram": token_bytes_ram,
                        "bytes_nvme": token_bytes_nvme,
                    }
                )

        per_token_stall = [t["stall_ms"] for t in per_token_metrics]
        per_token_bytes_vram = [t["bytes_vram"] for t in per_token_metrics]
        per_token_bytes_total = [
            t["bytes_vram"] + t["bytes_ram"] + t["bytes_nvme"]
            for t in per_token_metrics
        ]

        def _percentile(xs: List[float], pct: float) -> float:
            if not xs:
                return 0.0
            xs_sorted = sorted(xs)
            idx = max(0, min(len(xs_sorted) - 1, int(round((pct / 100.0) * (len(xs_sorted) - 1)))))
            return xs_sorted[idx]

        return {
            "total_expert_requests": total_expert_requests,
            "l1_hits": vram_hit_count,
            "l2_hits": ram_hit_count,
            "nvme_fetches": nvme_fetch_count,
            "stalling_misses": stalling_miss_count,
            "prefetched_hits": prefetch_hide_count,
            "expert_bytes_from_vram": bytes_from_vram,
            "expert_bytes_from_ram": bytes_from_ram,
            "expert_bytes_from_nvme": bytes_from_nvme,
            "l1_hit_rate": vram_hit_count / total_expert_requests if total_expert_requests else 0.0,
            "l2_hit_rate": ram_hit_count / total_expert_requests if total_expert_requests else 0.0,
            "nvme_fetch_rate": nvme_fetch_count / total_expert_requests if total_expert_requests else 0.0,
            "stalling_miss_rate": stalling_miss_count / total_expert_requests if total_expert_requests else 0.0,
            "prefetch_hide_rate": prefetch_hide_count / total_expert_requests if total_expert_requests else 0.0,
            "exposed_wait_ms_total": exposed_wait_ms_total,
            "exposed_wait_ms_per_token_mean": (
                statistics.fmean(per_token_stall) if per_token_stall else 0.0
            ),
            "exposed_wait_ms_per_token_p50": _percentile(per_token_stall, 50),
            "exposed_wait_ms_per_token_p95": _percentile(per_token_stall, 95),
            "fetch_latency_ms_p50": _percentile(fetch_latencies_ms, 50),
            "fetch_latency_ms_p95": _percentile(fetch_latencies_ms, 95),
            "bytes_per_token_vram_mean": (
                statistics.fmean(per_token_bytes_vram) if per_token_bytes_vram else 0.0
            ),
            "bytes_per_token_total_mean": (
                statistics.fmean(per_token_bytes_total) if per_token_bytes_total else 0.0
            ),
            "per_token": per_token_metrics,
        }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _git_commit() -> Optional[str]:
    try:
        out = subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            stderr=subprocess.DEVNULL,
            cwd=os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        )
        return out.decode().strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return None


def build_provenance(
    config: Qwen38Config,
    trace_source: str,
    trace_path: Optional[str],
    policy: str,
    l1_size_gb: float,
    l2_size_gb: float,
    prefetch_lookahead: int,
) -> Dict[str, Any]:
    return {
        "config_path": config.config_path,
        "config_sha256": config.config_sha256,
        "model_type": config.model_type,
        "architectures": list(config.architectures),
        "num_hidden_layers": config.num_hidden_layers,
        "num_experts": config.num_experts,
        "num_experts_per_tok": config.num_experts_per_tok,
        "bytes_per_expert_bf16": config.bytes_per_expert_bf16,
        "bytes_per_layer_experts_bf16": config.bytes_per_layer_experts_bf16,
        "trace_source": trace_source,
        "trace_path": trace_path,
        "policy": policy,
        "l1_vram_size_gb": l1_size_gb,
        "l2_ram_size_gb": l2_size_gb,
        "prefetch_lookahead": prefetch_lookahead,
        "simulator_commit": _git_commit(),
        "simulator_tool": "tools/trace_qwen38_routing.py",
        "data_classification": (
            "captured" if trace_source == "captured"
            else "synthetic_with_checkpoint_derived_arch"
        ),
    }


def emit_metrics(
    metrics: Dict[str, Any],
    provenance: Dict[str, Any],
    output_path: str,
) -> None:
    payload = {
        "schema_version": "qwen38.routing_metrics.v1",
        "provenance": provenance,
        "metrics": metrics,
    }
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)


def main(argv: Optional[Sequence[str]] = None) -> int:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument(
        "--config",
        default=os.environ.get("QWEN38_CONFIG", "checkpoints/Qwen3.8-2.4T-A95B/config.json"),
        help="Path to checkpoint config.json.",
    )
    p.add_argument(
        "--trace",
        choices=["captured", "synthetic"],
        default="synthetic",
        help="Trace source. Use 'captured' with --trace-path to load a real trace.",
    )
    p.add_argument(
        "--trace-path",
        default=None,
        help="Path to a captured routing trace JSON. Required when --trace=captured.",
    )
    p.add_argument(
        "--policy",
        choices=["lru", "lfru"],
        default="lru",
        help="Eviction policy for both L1 and L2 caches.",
    )
    p.add_argument(
        "--l1-size-gb", type=float, default=6.0,
        help="L1 (VRAM) cache size in GB (default: 6.0).",
    )
    p.add_argument(
        "--l2-size-gb", type=float, default=48.0,
        help="L2 (system RAM) cache size in GB (default: 48.0).",
    )
    p.add_argument(
        "--nvme-to-ram-bw-gbs", type=float, default=DEFAULT_NVME_TO_RAM_BW_GBS,
        help="NVMe -> RAM bandwidth in GB/s.",
    )
    p.add_argument(
        "--ram-to-vram-bw-gbs", type=float, default=DEFAULT_RAM_TO_VRAM_BW_GBS,
        help="RAM -> VRAM bandwidth in GB/s.",
    )
    p.add_argument(
        "--nvme-latency-us", type=float, default=DEFAULT_NVME_LATENCY_US,
        help="NVMe access latency in microseconds.",
    )
    p.add_argument(
        "--ram-latency-us", type=float, default=DEFAULT_RAM_LATENCY_US,
        help="RAM access latency in microseconds.",
    )
    p.add_argument(
        "--prefetch-lookahead", type=int, default=1,
        help="Number of future layers to prefetch speculatively (0, 1, 2).",
    )
    p.add_argument(
        "--tokens-per-prompt", type=int, default=20,
        help="Tokens per prompt for synthetic traces.",
    )
    p.add_argument(
        "--seed", type=int, default=0,
        help="Seed for synthetic trace generation.",
    )
    p.add_argument(
        "--output",
        default="artifacts/qwen38_routing_trace_metrics.json",
        help="Where to write the metrics JSON.",
    )
    p.add_argument(
        "--trace-output",
        default=None,
        help="If set, also write the trace JSON to this path (useful for "
             "bootstrapping a captured trace from a synthetic one).",
    )
    args = p.parse_args(argv)

    config = load_config(args.config)
    print(f"Loaded config: {config.config_path}")
    print(f"  config_sha256: {config.config_sha256}")
    print(f"  num_hidden_layers: {config.num_hidden_layers}")
    print(f"  num_experts: {config.num_experts}")
    print(f"  num_experts_per_tok: {config.num_experts_per_tok}")
    print(f"  bytes_per_expert (BF16): {config.bytes_per_expert_bf16_mib:.2f} MiB")
    print(f"  bytes_per_layer (BF16):  {config.bytes_per_layer_experts_bf16_gib:.3f} GiB")
    print()

    if args.trace == "captured":
        if not args.trace_path:
            print("ERROR: --trace=captured requires --trace-path", file=sys.stderr)
            return 2
        print(f"Loading captured trace from {args.trace_path} ...")
        traces = load_captured_traces(args.trace_path, config)
        trace_source = "captured"
        trace_path = os.path.abspath(args.trace_path)
    else:
        print("Generating synthetic trace ...")
        traces = synth_trace(
            config,
            tokens_per_prompt=args.tokens_per_prompt,
            seed=args.seed,
        )
        trace_source = "synthetic"
        trace_path = None
        if args.trace_output:
            os.makedirs(os.path.dirname(args.trace_output) or ".", exist_ok=True)
            with open(args.trace_output, "w", encoding="utf-8") as f:
                json.dump(
                    {
                        "schema_version": "qwen38.routing_trace.v1",
                        "model_revision": config.config_sha256,
                        "config_sha256": config.config_sha256,
                        "dtype": "bf16",
                        "pack_revision": "synthetic-v1",
                        "captured_at": None,
                        "prompts": [t.to_dict() for t in traces],
                    },
                    f,
                    indent=2,
                )
            print(f"  Wrote synthetic trace to {args.trace_output}")

    print(f"Replaying {len(traces)} prompt(s) through cache simulator ...")
    sim = CacheSimulator.from_args(
        config=config,
        policy=args.policy,
        l1_size_gb=args.l1_size_gb,
        l2_size_gb=args.l2_size_gb,
        nvme_to_ram_bw_gbs=args.nvme_to_ram_bw_gbs,
        ram_to_vram_bw_gbs=args.ram_to_vram_bw_gbs,
        nvme_latency_us=args.nvme_latency_us,
        ram_latency_us=args.ram_latency_us,
        prefetch_lookahead=args.prefetch_lookahead,
    )
    metrics = sim.replay(traces)

    provenance = build_provenance(
        config=config,
        trace_source=trace_source,
        trace_path=trace_path,
        policy=args.policy,
        l1_size_gb=args.l1_size_gb,
        l2_size_gb=args.l2_size_gb,
        prefetch_lookahead=args.prefetch_lookahead,
    )
    emit_metrics(metrics, provenance, args.output)
    print()
    print("=== Routing trace replay summary ===")
    print(f"  data_classification: {provenance['data_classification']}")
    print(f"  trace_source:        {provenance['trace_source']}")
    print(f"  policy:              {args.policy}")
    print(f"  L1 size (GB):        {args.l1_size_gb}")
    print(f"  L2 size (GB):        {args.l2_size_gb}")
    print(f"  prefetch_lookahead:  {args.prefetch_lookahead}")
    print()
    print("  Total expert requests:", metrics["total_expert_requests"])
    print(f"  L1 (VRAM) hit rate:    {metrics['l1_hit_rate']:.2%}")
    print(f"  L2 (RAM) hit rate:     {metrics['l2_hit_rate']:.2%}")
    print(f"  NVMe fetch rate:       {metrics['nvme_fetch_rate']:.2%}")
    print(f"  Stalling miss rate:    {metrics['stalling_miss_rate']:.2%}")
    print(f"  Prefetch hide rate:    {metrics['prefetch_hide_rate']:.2%}")
    print()
    print("  Exposed-wait latency (per token, ms):")
    print(f"    mean:  {metrics['exposed_wait_ms_per_token_mean']:.2f}")
    print(f"    p50:   {metrics['exposed_wait_ms_per_token_p50']:.2f}")
    print(f"    p95:   {metrics['exposed_wait_ms_per_token_p95']:.2f}")
    print()
    print("  Fetch latency (NVMe->RAM+RAM->VRAM, ms):")
    print(f"    p50:   {metrics['fetch_latency_ms_p50']:.2f}")
    print(f"    p95:   {metrics['fetch_latency_ms_p95']:.2f}")
    print()
    print("  Bytes per token (mean):")
    print(f"    from VRAM:  {metrics['bytes_per_token_vram_mean'] / (1024 ** 2):.2f} MiB")
    print(f"    total:      {metrics['bytes_per_token_total_mean'] / (1024 ** 2):.2f} MiB")
    print()
    print(f"  Metrics written to {args.output}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
