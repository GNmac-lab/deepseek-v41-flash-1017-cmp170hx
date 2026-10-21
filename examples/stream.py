#!/usr/bin/env python3
"""Stream a reply token by token and report the decode rate. Usage: examples/stream.py "prompt" [--port 8000]"""
import argparse
import json
import time
import urllib.request

ap = argparse.ArgumentParser()
ap.add_argument("prompt")
ap.add_argument("--port", default="8000")
a = ap.parse_args()
body = {"model": "deepseek-v4.1-flash", "messages": [{"role": "user", "content": a.prompt}], "max_tokens": 1000,
        "reasoning_effort": "none", "stream": True, "stream_options": {"include_usage": True}}
req = urllib.request.Request(f"http://127.0.0.1:{a.port}/v1/chat/completions", json.dumps(body).encode(),
                             {"Content-Type": "application/json"})
first = usage = None
with urllib.request.urlopen(req, timeout=3600) as r:
    for line in r:
        line = line.decode().strip()
        if not line.startswith("data: ") or line == "data: [DONE]":
            continue
        chunk = json.loads(line[6:])
        if chunk.get("usage"):
            usage = chunk["usage"]
        for ch in chunk.get("choices", []):
            text = ch.get("delta", {}).get("content")
            if text:
                first = first or time.perf_counter()
                print(text, end="", flush=True)
end = time.perf_counter()
if usage and first:
    print(f"\n\n{usage['completion_tokens']} tokens, {(usage['completion_tokens'] - 1) / (end - first):.1f} tok/s after the first token")
