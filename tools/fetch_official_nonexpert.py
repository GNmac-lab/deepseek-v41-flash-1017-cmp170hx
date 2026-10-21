"""Fetch only the official DeepSeek-V4.1-Flash-1017 tensors the hybrid checkpoint needs (everything except the
layer 0-39 routed experts and the Engram tables, ~17.5 GiB of 510 GB) with HTTP range requests, and write them as
new safetensors files (one per source shard) plus a model.safetensors.index.json. Resumable: finished files are
skipped. Set HF_TOKEN if your Hugging Face account needs a token for the repository.
Usage: fetch_official_nonexpert.py OUT_DIR"""
import concurrent.futures as cf
import json
import os
import re
import struct
import sys
import time
import urllib.request

REPO = "https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash-1017/resolve/main/"
OUT = sys.argv[1]
os.makedirs(OUT, exist_ok=True)


def get(url, rng=None, tries=8):
    for a in range(tries):
        try:
            hdrs = {"Range": f"bytes={rng}"} if rng else {}
            if os.environ.get("HF_TOKEN"):
                hdrs["Authorization"] = f"Bearer {os.environ['HF_TOKEN']}"
            req = urllib.request.Request(url, headers=hdrs)
            data = urllib.request.urlopen(req, timeout=60).read()
            if rng:
                lo, hi = (int(x) for x in rng.split("-"))
                if len(data) != hi - lo + 1:
                    raise IOError(f"short read {len(data)} != {hi - lo + 1}")
            return data
        except Exception as e:
            if a == tries - 1:
                raise
            print(f"retry {a + 1} {url.rsplit('/', 1)[-1]} {rng}: {e}", flush=True)
            time.sleep(5 * (a + 1))


def wanted(name):
    return not re.match(r"layers\.\d+\.ffn\.experts\.", name) and ".engram.embed." not in name


src_idx = json.loads(get(REPO + "model.safetensors.index.json"))["weight_map"]
for f in ("config.json", "model.safetensors.index.json"):
    open(os.path.join(OUT, "official-" + f), "wb").write(get(REPO + f))
weight_map = {}
total = 0
for sh in sorted(set(src_idx.values())):
    out_name = "official-" + sh
    out_path = os.path.join(OUT, out_name)
    n = struct.unpack("<Q", get(REPO + sh, "0-7"))[0]
    hdr = json.loads(get(REPO + sh, f"8-{7 + n}"))
    base = 8 + n
    keep = sorted(((k, v) for k, v in hdr.items() if k != "__metadata__" and wanted(k)),
                  key=lambda kv: kv[1]["data_offsets"][0])
    if not keep:
        continue
    for k, _ in keep:
        weight_map[k] = out_name
    size = sum(v["data_offsets"][1] - v["data_offsets"][0] for _, v in keep)
    total += size
    if os.path.exists(out_path + ".done"):
        print(f"skip {out_name} (done)", flush=True)
        continue
    # merge adjacent tensors into ranges of at most 256 MiB
    ranges = []
    for k, v in keep:
        lo, hi = v["data_offsets"]
        if ranges and ranges[-1][1] == lo and hi - ranges[-1][0] <= 256 << 20:
            ranges[-1][1] = hi
            ranges[-1][2].append(k)
        else:
            ranges.append([lo, hi, [k]])
    with cf.ThreadPoolExecutor(8) as ex:
        blobs = list(ex.map(lambda r: get(REPO + sh, f"{base + r[0]}-{base + r[1] - 1}"), ranges))
    new_hdr, chunks, off = {}, [], 0
    for (lo, hi, names), blob in zip(ranges, blobs):
        for k in names:
            a, b = hdr[k]["data_offsets"]
            data = blob[a - lo:b - lo]
            new_hdr[k] = {"dtype": hdr[k]["dtype"], "shape": hdr[k]["shape"], "data_offsets": [off, off + len(data)]}
            chunks.append(data)
            off += len(data)
    hj = json.dumps(new_hdr, separators=(",", ":")).encode()
    hj += b" " * (-len(hj) % 8)
    with open(out_path + ".part", "wb") as fh:
        fh.write(struct.pack("<Q", len(hj)) + hj)
        for c in chunks:
            fh.write(c)
    os.replace(out_path + ".part", out_path)
    open(out_path + ".done", "w").write(str(off))
    print(f"wrote {out_name}: {len(keep)} tensors, {off / 2**30:.2f} GiB, {len(ranges)} ranges", flush=True)
json.dump({"metadata": {"total_size": total}, "weight_map": weight_map},
          open(os.path.join(OUT, "model.safetensors.index.json"), "w"), indent=1)
print(f"ALL DONE: {len(weight_map)} tensors, {total / 2**30:.2f} GiB", flush=True)
