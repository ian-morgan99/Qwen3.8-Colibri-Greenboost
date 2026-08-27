import urllib.request, json, time, os
base = "https://huggingface.co/sakamakismile/Huihui-Qwen3.8-27B-abliterated-NVFP4/resolve/main/"
dest_dir = "/home/ian/Documents/VSCodeProjects/Qwen3.8/models-download/Huihui-Qwen3.8-27B-abliterated-NVFP4"
files = ["model.safetensors", "model.safetensors.index.json", "config.json",
         "generation_config.json", "chat_template.jinja", "tokenizer.json",
         "tokenizer_config.json", "merges.txt", "vocab.json"]
CHUNK = 1<<20
RATE = 0  # unthrottled: GGUF downloader keeps its own 3 MB/s cap

os.makedirs(dest_dir, exist_ok=True)
tree = json.load(urllib.request.urlopen(
    "https://huggingface.co/api/models/sakamakismile/Huihui-Qwen3.8-27B-abliterated-NVFP4/tree/main"))
SIZES = {os.path.basename(t["path"]): t["size"] for t in tree}

for f in files:
    dest = os.path.join(dest_dir, f)
    expected = SIZES.get(f)
    if os.path.exists(dest):
        if expected is None or os.path.getsize(dest) == expected:
            print("skip", f, flush=True); continue
        print("revert partial", f, os.path.getsize(dest), flush=True)
        os.replace(dest, dest + ".part")
    part = dest + ".part"
    url = base + f
    for attempt in range(200):
        try:
            offset = os.path.getsize(part) if os.path.exists(part) else 0
            req = urllib.request.Request(url, headers={"User-Agent":"python"})
            if offset: req.add_header("Range", f"bytes={offset}-")
            with urllib.request.urlopen(req, timeout=60) as r:
                mode = "ab" if offset else "wb"
                with open(part, mode) as out:
                    t0 = time.time(); sent = 0
                    while True:
                        chunk = r.read(CHUNK)
                        if not chunk: break
                        out.write(chunk); out.flush(); sent += len(chunk)
                        if RATE:
                            target = sent/RATE; elapsed = time.time()-t0
                            if elapsed < target: time.sleep(target-elapsed)
            if expected and os.path.getsize(part) != expected:
                print("incomplete", f, os.path.getsize(part), "of", expected, "- retrying", flush=True)
                time.sleep(15); continue
            os.replace(part, dest)
            print("done", f, os.path.getsize(dest), flush=True)
            break
        except Exception as e:
            print("retry", f, attempt, repr(e), flush=True)
            time.sleep(15)
print("ALL DONE", flush=True)
