#!/usr/bin/env python3
"""ftmon — FreeToken sidecar monitor for Qwen3.8-2.4T-A95B bring-up.

Reads FreeToken's control API (GET /v1/stats, GET /v1/requests) on 127.0.0.1:1919
and renders a live terminal dashboard plus an append-only JSONL telemetry log.
Zero engine patches: everything comes from the documented endpoints, so it works
unchanged across Phase B (35B-A3B smoke test) and Phase D (2.4T tuning).

Usage:
    ftmon.py [--url URL] [--interval S] [--log PATH] [--once] [--no-tui]

    --url       base URL of the FreeToken server (default http://127.0.0.1:1919)
    --interval  poll interval in seconds (default 2.0)
    --log       JSONL output path (default ./ftmon.jsonl next to this script)
    --once      print one snapshot and exit (for scripts / cron)
    --no-tui    log only; no ANSI dashboard (safe for nohup / CI)

Exit codes: 0 normal, 1 server unreachable on --once.

JSONL record shape (one per poll):
    {"ts": ..., "stats": <full /v1/stats doc>, "vram_gb": ..., "decode_tps": ...,
     "prefill_tps": ..., "kv_used_pct": ..., "active": ...}
"""

from __future__ import annotations

import argparse
import json
import os
import signal
import sys
import time
import urllib.error
import urllib.request
from datetime import datetime, timezone

GB = 1024 ** 3


def fetch_json(base_url: str, path: str, timeout: float = 5.0):
    req = urllib.request.Request(base_url.rstrip("/") + path)
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.load(resp)


def kv_used_pct(stats: dict) -> float | None:
    kv = stats.get("kv")
    if not kv or not kv.get("total_pages"):
        return None
    return 100.0 * kv["used_pages"] / kv["total_pages"]


def summarize(stats: dict) -> dict:
    vram = stats.get("vram_bytes") or 0
    return {
        "vram_gb": round(vram / GB, 2),
        "decode_tps": stats.get("throughput", {}).get("decode_tps", 0.0),
        "prefill_tps": stats.get("throughput", {}).get("prefill_tps", 0.0),
        "kv_used_pct": round(kv_used_pct(stats), 1) if kv_used_pct(stats) is not None else None,
        "active": stats.get("requests", {}).get("active", 0),
    }


def render_line(stats: dict, s: dict) -> str:
    model = (stats.get("model") or {}).get("id", "?")
    up = stats.get("uptime_s", 0)
    hh, rem = divmod(up, 3600)
    mm, ss = divmod(rem, 60)
    kv = f"{s['kv_used_pct']:.1f}%" if s["kv_used_pct"] is not None else "n/a"
    return (
        f"[{datetime.now().strftime('%H:%M:%S')}] {model} "
        f"up={hh:d}h{mm:02d}m  VRAM={s['vram_gb']:.1f}GB  "
        f"dec={s['decode_tps']:.1f}tok/s  pre={s['prefill_tps']:.1f}tok/s  "
        f"KV={kv}  active={s['active']}"
    )


def render_dashboard(stats: dict, s: dict, ok: bool, polls: int, errors: int) -> str:
    # ANSI: clear screen, home cursor.
    out = ["\x1b[2J\x1b[H"]
    title = "ftmon — FreeToken monitor"
    out.append(f"\x1b[1m{title}\x1b[0m  polls={polls} errors={errors}")
    if not ok:
        out.append("\x1b[31mserver unreachable\x1b[0m")
        return "\n".join(out)
    m = stats.get("model") or {}
    r = stats.get("requests") or {}
    t = stats.get("throughput") or {}
    kv = stats.get("kv")
    lines = [
        "",
        f"model      : {m.get('id', '?')}  (attn={m.get('attn', '?')}, moe={m.get('moe')})",
        f"context    : {m.get('ctx', '?'):>10,} tokens",
        f"uptime     : {stats.get('uptime_s', 0):,} s",
        f"VRAM       : {s['vram_gb']:.2f} GB",
        f"throughput : decode {t.get('decode_tps', 0):.1f} tok/s   prefill {t.get('prefill_tps', 0):.1f} tok/s",
        f"requests   : active={r.get('active', 0)} completed={r.get('completed', 0)} "
        f"p95={r.get('p95_ms', 0)}ms ttft_mean={r.get('ttft_mean_ms', 0)}ms",
        f"tokens     : prompt={r.get('prompt_tokens_total', 0):,} completion={r.get('completion_tokens_total', 0):,}",
    ]
    if kv:
        pct = 100.0 * kv["used_pages"] / max(1, kv["total_pages"])
        bar_w = 30
        filled = int(bar_w * kv["used_pages"] / max(1, kv["total_pages"]))
        lines.append(
            f"KV pool    : [{'#' * filled}{'.' * (bar_w - filled)}] "
            f"{kv['used_pages']:,}/{kv['total_pages']:,} pages ({pct:.1f}%) page_size={kv.get('page_size', 1)}"
        )
    else:
        lines.append("KV pool    : n/a")
    if stats.get("swa"):
        swa = stats["swa"]
        lines.append(f"SWA pool   : {swa['used_pages']:,}/{swa['total_pages']:,} pages")
    if stats.get("mamba"):
        mb = stats["mamba"]
        lines.append(f"Mamba slots: {mb['used_slots']:,}/{mb['total_slots']:,}")
    out.extend(lines)
    return "\n".join(out)


def main() -> int:
    ap = argparse.ArgumentParser(description="FreeToken sidecar monitor")
    ap.add_argument("--url", default="http://127.0.0.1:1919")
    ap.add_argument("--interval", type=float, default=2.0)
    ap.add_argument("--log", default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "ftmon.jsonl"))
    ap.add_argument("--once", action="store_true")
    ap.add_argument("--no-tui", action="store_true")
    args = ap.parse_args()

    running = {"flag": True}

    def _stop(signum, frame):
        running["flag"] = False

    signal.signal(signal.SIGINT, _stop)
    signal.signal(signal.SIGTERM, _stop)

    polls = errors = 0
    logf = open(args.log, "a", buffering=1) if args.log != "-" else sys.stdout
    try:
        while running["flag"]:
            ok = True
            stats = None
            try:
                stats = fetch_json(args.url, "/v1/stats")
                polls += 1
            except (urllib.error.URLError, OSError, json.JSONDecodeError):
                ok = False
                errors += 1

            if stats is not None:
                s = summarize(stats)
                rec = {"ts": datetime.now(timezone.utc).isoformat(), "stats": stats, **s}
                print(json.dumps(rec), file=logf)
                line = render_line(stats, s)
                if args.once:
                    print(line)
                elif not args.no_tui:
                    print(render_dashboard(stats, s, True, polls, errors))
                else:
                    print(line, flush=True)
            elif args.once:
                print("ftmon: server unreachable at " + args.url, file=sys.stderr)
                return 1
            elif not args.no_tui:
                print(render_dashboard(None, None, False, polls, errors))

            if args.once:
                break
            # Interruptible sleep so Ctrl-C lands promptly.
            end = time.monotonic() + args.interval
            while running["flag"] and time.monotonic() < end:
                time.sleep(min(0.25, max(0.05, end - time.monotonic())))
    finally:
        if logf is not sys.stdout:
            logf.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
