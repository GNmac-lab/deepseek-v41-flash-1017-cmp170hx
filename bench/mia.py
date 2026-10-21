#!/usr/bin/env python3
"""Decode throughput under the MiaAI-Lab protocol (the same one Morrowmake's recipe reports).

Downloads MiaAI-Lab's tests/bench_decode.py at a fixed commit (AGPL-3.0, so it is fetched, not shipped here), checks
its git blob hash, and runs it unmodified against this server, overriding only BASE and MODEL:
  structured / code / prose prompts, temperature 0, thinking off, 400-token cap;
  1 user = median of 5 streaming decode rates (measured after the first token);
  2, 4 and 8 users = median of 3 runs of aggregate throughput (completion tokens / batch wall time).
Usage: bench/mia.py [--port 8000] [--model deepseek-v4.1-flash] [--label NAME] [--users 1,2,4,8]"""
import argparse
import hashlib
import json
import os
import runpy
import statistics
import sys
import urllib.request
from pathlib import Path

MIA_REPO = "MiaAI-Lab/GLM-5.3-Flash-EXL3-2x-DGX-Sparks"
MIA_COMMIT = "6278ecb"
MIA_BLOB = "d9b24cf29a56c6e215144b267bc874e428b9dcb4"  # git blob of tests/bench_decode.py at that commit
RUNS = {1: 5, 2: 3, 4: 3, 8: 3}
KINDS = (("structured", ["--structured"]), ("code", ["--coding"]), ("prose", []))


def fetch_bench(cache: Path) -> Path:
    path = cache / f"bench_decode-{MIA_COMMIT}.py"
    if not path.exists():
        url = f"https://raw.githubusercontent.com/{MIA_REPO}/{MIA_COMMIT}/tests/bench_decode.py"
        data = urllib.request.urlopen(url, timeout=60).read()
        cache.mkdir(parents=True, exist_ok=True)
        path.write_bytes(data)
    data = path.read_bytes()
    blob = hashlib.sha1(b"blob %d\0" % len(data) + data).hexdigest()
    if blob != MIA_BLOB:
        sys.exit(f"{path}: unexpected content (git blob {blob}, want {MIA_BLOB}); delete it and retry")
    return path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", default=os.environ.get("PORT", "8000"))
    ap.add_argument("--model", default="deepseek-v4.1-flash")
    ap.add_argument("--label", default="run")
    ap.add_argument("--users", default="1,2,4,8")
    a = ap.parse_args()
    here = Path(__file__).resolve().parent
    out_dir = here / "out"
    out_dir.mkdir(exist_ok=True)
    g = runpy.run_path(str(fetch_bench(here / ".cache")), run_name="miabench")
    G = g["main"].__globals__
    G["BASE"] = f"http://127.0.0.1:{a.port}"
    G["MODEL"] = a.model
    users = [int(u) for u in a.users.split(",")]
    table = {}
    for name, flag in KINDS:
        for conc in users:
            out = out_dir / f"{a.label}-{name}-c{conc}.json"
            sys.argv = ["bench_decode.py", "--phase", f"{a.label}-{name}-c{conc}", "--out", str(out),
                        "--runs", str(RUNS.get(conc, 3)), "--max-tokens", "400", "--concurrency", str(conc),
                        "--skip-coherence", *flag]
            g["main"]()
            rec = json.load(open(out))
            key = "tok_s" if conc == 1 else "aggregate_tok_s"
            table[(name, conc)] = round(statistics.median(r[key] for r in rec["runs"]), 1)
    print(f"\n| {a.label} | structured | code | prose |\n|---|---:|---:|---:|")
    for conc in users:
        what = "1 user, streaming" if conc == 1 else f"{conc} users, aggregate"
        print(f"| {what} | " + " | ".join(f"{table[(k, conc)]}" for k, _ in KINDS) + " |")
    print("MIA", a.label, json.dumps({f"{k} {c}u": v for (k, c), v in table.items()}))


if __name__ == "__main__":
    main()
