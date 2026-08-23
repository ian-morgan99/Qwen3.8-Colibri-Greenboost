#!/usr/bin/env python3
"""ftserve — launch and supervise a FreeToken server for Qwen3.8 bring-up.

Wraps `ft serve` with the flag profiles from FREETOKEN_REVISED_PLAN.md (Phases B/D),
waits for readiness by polling /v1/models, and never touches any other process
(it does NOT kill or restart the user's LM Studio instance).

Usage:
    ftserve.py --model Qwen/Qwen3.5-35B-A3B [--profile smoke|tune24t] [ft serve flags...]

Profiles:
    smoke    Phase B defaults (auto backend selection, moe-cache-auto).
    tune24t  Phase D starting point for the 2.4T checkpoint:
             --moe-backend auto --moe-cache-auto --memory-ratio 0.9
             --kv-reserve-tokens 8192

Readiness: polls GET /v1/models until 200 or timeout; prints "READY" so shell
scripts can wait on it. Extra args after the model are passed verbatim to ft serve.
"""

from __future__ import annotations

import argparse
import json
import os
import signal
import subprocess
import sys
import time
import urllib.request

PROFILES = {
    "smoke": [],
    "tune24t": [
        "--moe-backend", "auto",
        "--moe-cache-auto",
        "--memory-ratio", "0.9",
        "--kv-reserve-tokens", "8192",
    ],
}


def find_ft() -> str:
    env_ft = os.environ.get("FT_BIN")
    if env_ft and os.path.exists(env_ft):
        return env_ft
    here = os.path.dirname(os.path.abspath(__file__))
    candidate = os.path.join(here, "..", "freetoken-env", ".venv", "bin", "ft")
    if os.path.exists(candidate):
        return os.path.abspath(candidate)
    return "ft"


def wait_ready(port: int, timeout_s: float, log_path: str) -> bool:
    """Poll /v1/models. A 503 from /v1/chat means weights still loading; /v1/models
    only answers once the frontend is up, so treat 200 as ready."""
    deadline = time.monotonic() + timeout_s
    url = f"http://127.0.0.1:{port}/v1/models"
    while time.monotonic() < deadline:
        try:
            with urllib.request.urlopen(url, timeout=5) as resp:
                if resp.status == 200:
                    doc = json.load(resp)
                    ids = [m["id"] for m in doc.get("data", [])]
                    print(f"READY models={ids}", flush=True)
                    return True
        except Exception:
            pass
        time.sleep(5)
    print(f"NOT READY after {timeout_s:.0f}s — see {log_path}", file=sys.stderr, flush=True)
    return False


def main() -> int:
    ap = argparse.ArgumentParser(description="FreeToken serve wrapper")
    ap.add_argument("--model", required=True)
    ap.add_argument("--profile", choices=sorted(PROFILES), default="smoke")
    ap.add_argument("--port", type=int, default=1919)
    ap.add_argument("--ready-timeout", type=float, default=3600,
                    help="seconds to wait for weight load before declaring failure "
                         "(large checkpoints can take hours)")
    ap.add_argument("extra", nargs="*", help="additional flags passed to ft serve")
    args = ap.parse_args()

    ft = find_ft()
    cmd = [ft, "serve", "--model", args.model, "--port", str(args.port)] + PROFILES[args.profile] + args.extra
    log_path = os.path.abspath(f"ftserve_{args.port}.log")
    print("launching:", " ".join(cmd), flush=True)
    print("log:", log_path, flush=True)

    with open(log_path, "a") as logf:
        proc = subprocess.Popen(cmd, stdout=logf, stderr=subprocess.STDOUT)

        def _forward(signum, frame):
            # Forward signals rather than killing blindly; child shuts down cleanly.
            proc.send_signal(signum)

        signal.signal(signal.SIGINT, _forward)
        signal.signal(signal.SIGTERM, _forward)

        try:
            ready = wait_ready(args.port, args.ready_timeout, log_path)
            rc = proc.wait()
        except KeyboardInterrupt:
            proc.terminate()
            rc = proc.wait()
    sys.exit(rc if ready or rc == 0 else rc)


if __name__ == "__main__":
    main()
