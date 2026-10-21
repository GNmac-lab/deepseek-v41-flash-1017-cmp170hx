#!/usr/bin/env python3
"""Cold prefill throughput: median of 9 rates over three real-text prompts (~24K, ~34K and ~38K tokens), 3 runs each.

The texts are public-domain Project Gutenberg books, downloaded on first use and cut to length with the server's own
tokenizer. Every run starts with a different random line, so the prefix cache cannot hit. One request at a time,
max_tokens 1, thinking off; rate = prompt tokens / time to the first token.
Usage: bench/prefill.py [--port 8000] [--model deepseek-v4.1-flash]"""
import argparse
import json
import os
import random
import statistics
import time
import urllib.request
from pathlib import Path

BOOKS = (("pride-and-prejudice", 1342, 24000), ("moby-dick", 2701, 34000), ("war-and-peace", 2600, 38000))


def post(base, path, body, timeout=3600):
    req = urllib.request.Request(base + path, json.dumps(body).encode(), {"Content-Type": "application/json"})
    return json.load(urllib.request.urlopen(req, timeout=timeout))


def book(cache: Path, name: str, num: int) -> str:
    p = cache / f"{name}.txt"
    if not p.exists():
        cache.mkdir(parents=True, exist_ok=True)
        url = f"https://www.gutenberg.org/cache/epub/{num}/pg{num}.txt"
        p.write_bytes(urllib.request.urlopen(url, timeout=120).read())
    t = p.read_text(encoding="utf-8", errors="replace")
    start = t.find("*** START OF")
    return t[t.find("\n", start) + 1:] if start >= 0 else t


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", default=os.environ.get("PORT", "8000"))
    ap.add_argument("--model", default="deepseek-v4.1-flash")
    a = ap.parse_args()
    base = f"http://127.0.0.1:{a.port}"
    cache = Path(__file__).resolve().parent / ".cache"
    rates = []
    for name, num, target in BOOKS:
        text = book(cache, name, num)
        ids = post(base, "/tokenize", {"model": a.model, "prompt": text})["tokens"][:target]
        excerpt = post(base, "/detokenize", {"model": a.model, "tokens": ids})["prompt"]
        for _ in range(3):
            nonce = f"Reference {random.getrandbits(64):016x}.\n\n"
            body = {"model": a.model, "max_tokens": 1, "temperature": 0,
                    "messages": [{"role": "user", "content": nonce + excerpt + "\n\nSummarise the text above in one sentence."}],
                    "chat_template_kwargs": {"enable_thinking": False, "thinking": False}}
            t0 = time.perf_counter()
            r = post(base, "/v1/chat/completions", body)
            dt = time.perf_counter() - t0
            n = r["usage"]["prompt_tokens"]
            cached = (r["usage"].get("prompt_tokens_details") or {}).get("cached_tokens") or 0
            rates.append(n / dt)
            print(f"{name:20s} {n:6d} tokens  {dt:6.2f} s  {n / dt:7.0f} tok/s  cached={cached}", flush=True)
    print(f"cold prefill median of {len(rates)}: {statistics.median(rates):.0f} tok/s")


if __name__ == "__main__":
    main()
