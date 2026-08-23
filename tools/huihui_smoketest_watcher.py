#!/usr/bin/env python3
"""Watch for huihui NVFP4 download completion + free VRAM, then run the ft serve smoke test.

Does NOT touch LM Studio or any user process. Polls every 60s; when both
conditions hold, starts `ft serve` in the background and runs ftsmoke.py.
"""
import os, subprocess, sys, time

REPO = "/home/ian/Documents/VSCodeProjects/Qwen3.8"
MODEL_DIR = f"{REPO}/models-download/Huihui-Qwen3.8-27B-abliterated-NVFP4"
EXPECTED_SIZE = 19_629_932_544  # from HF API tree; downloader renames .part on success
VENV = f"{REPO}/freetoken-env/.venv/bin/python"
LOG = f"{REPO}/models-download/huihui_smoketest.log"

def download_done():
    final = os.path.join(MODEL_DIR, "model.safetensors")
    if os.path.exists(final):
        return os.path.getsize(final) >= EXPECTED_SIZE * 0.99
    return False

def vram_free_mib():
    out = subprocess.run(["nvidia-smi", "--query-gpu=memory.free", "--format=csv,noheader,nounits"],
                         capture_output=True, text=True).stdout.strip().splitlines()
    return int(out[0]) if out else 0

def main():
    print(f"[watcher] waiting for model download + >=21GB free VRAM...", flush=True)
    while True:
        if download_done():
            free = vram_free_mib()
            print(f"[watcher] download done; VRAM free = {free} MiB", flush=True)
            if free >= 21_000:
                break
        time.sleep(60)

    print("[watcher] launching ft serve...", flush=True)
    serve = subprocess.Popen(
        [VENV, "-m", "freetoken.cli", "serve", "--model", MODEL_DIR],
        stdout=open(LOG, "a"), stderr=subprocess.STDOUT, cwd="/tmp")
    # wait for server to accept connections
    import urllib.request
    for _ in range(180):
        time.sleep(10)
        if serve.poll() is not None:
            print(f"[watcher] ft serve exited rc={serve.returncode}", flush=True); return 1
        try:
            urllib.request.urlopen("http://127.0.0.1:8000/health", timeout=2)
            break
        except Exception:
            pass
    else:
        print("[watcher] server never became healthy", flush=True); return 1

    print("[watcher] running ftsmoke.py...", flush=True)
    rc = subprocess.run([VENV, f"{REPO}/tools/ftsmoke.py"], cwd="/tmp",
                        stdout=open(LOG, "a"), stderr=subprocess.STDOUT).returncode
    print(f"[watcher] ftsmoke rc={rc}; see {LOG}", flush=True)
    return rc

if __name__ == "__main__":
    sys.exit(main())
