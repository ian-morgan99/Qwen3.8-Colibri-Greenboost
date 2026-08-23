#!/usr/bin/env python3
"""ftsmoke — Phase B acceptance test for FreeToken bring-up (FREETOKEN_REVISED_PLAN.md §3-B).

Against a running server (default 127.0.0.1:1919) it verifies:
  1. GET /v1/models returns the served model
  2. POST /v1/chat/completions non-streaming round-trip
  3. Streaming chat completion yields >= N tokens
  4. GET /v1/stats reports throughput counters

Exit code 0 = all checks pass. Designed to run against the small sibling model
(Qwen3.5-35B-A3B) before committing to the 2.4T download/bring-up.

Usage:
    ftsmoke.py [--url URL] [--model ID] [--max-tokens N]
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import urllib.error
import urllib.request


def post(url: str, payload: dict, timeout: float = 600):
    req = urllib.request.Request(url, data=json.dumps(payload).encode(),
                                 headers={"Content-Type": "application/json"})
    return urllib.request.urlopen(req, timeout=timeout)


def check(name: str, ok: bool, detail: str = "") -> bool:
    mark = "PASS" if ok else "FAIL"
    print(f"[{mark}] {name}" + (f" — {detail}" if detail else ""), flush=True)
    return ok


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="http://127.0.0.1:1919")
    ap.add_argument("--model", default=None, help="defaults to first id from /v1/models")
    ap.add_argument("--max-tokens", type=int, default=64)
    args = ap.parse_args()
    base = args.url.rstrip("/")

    ok = True

    # 1. models endpoint
    try:
        with urllib.request.urlopen(base + "/v1/models", timeout=10) as r:
            doc = json.load(r)
        ids = [m["id"] for m in doc.get("data", [])]
        ok &= check("GET /v1/models", True, f"models={ids}")
        model = args.model or (ids[0] if ids else None)
    except Exception as e:
        ok &= check("GET /v1/models", False, str(e))
        return 1
    if not model:
        print("no model id available", file=sys.stderr)
        return 1

    # The engine may still be loading weights (503 on completions); retry patiently.
    def wait_engine(max_wait_s: float = 7200):
        deadline = time.monotonic() + max_wait_s
        while time.monotonic() < deadline:
            try:
                return post(base + "/v1/chat/completions",
                            {"model": model, "messages": [{"role": "user", "content": "ping"}],
                             "max_tokens": 1}, timeout=120)
            except urllib.error.HTTPError as e:
                if e.code == 503:
                    time.sleep(30)
                    continue
                raise
        return None

    # 2. non-streaming completion
    t0 = time.monotonic()
    resp = wait_engine()
    if resp is None:
        ok &= check("chat completion (non-streaming)", False, "engine never became ready")
        return 1
    r = json.load(resp)
    content = r["choices"][0]["message"]["content"]
    usage = r.get("usage", {})
    dt = time.monotonic() - t0
    ok &= check("chat completion (non-streaming)", bool(content),
                f"{len(content)} chars in {dt:.1f}s, usage={usage}")

    # 3. streaming completion
    t0 = time.monotonic()
    chunks = 0
    ttft = None
    with post(base + "/v1/chat/completions",
              {"model": model, "stream": True,
               "messages": [{"role": "user", "content": "Count from one to ten."}],
               "max_tokens": args.max_tokens}, timeout=600) as sr:
        for raw in sr:
            line = raw.decode().strip()
            if not line.startswith("data:"):
                continue
            data = line[5:].strip()
            if data == "[DONE]":
                break
            if ttft is None:
                ttft = time.monotonic() - t0
            delta = json.loads(data)["choices"][0].get("delta", {}).get("content")
            if delta:
                chunks += 1
    ok &= check("streaming completion", chunks >= 3,
                f"{chunks} content chunks, TTFT={ttft:.2f}s" if ttft else "no tokens")

    # 4. stats endpoint
    try:
        with urllib.request.urlopen(base + "/v1/stats", timeout=10) as r:
            stats = json.load(r)
        dec = stats.get("throughput", {}).get("decode_tps", 0)
        completed = stats.get("requests", {}).get("completed", 0)
        ok &= check("GET /v1/stats", completed >= 2,
                    f"decode_tps={dec} completed={completed} vram_gb={(stats.get('vram_bytes') or 0)/2**30:.2f}")
    except Exception as e:
        ok &= check("GET /v1/stats", False, str(e))

    print("\nRESULT:", "ALL PASS" if ok else "FAILURES PRESENT")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
