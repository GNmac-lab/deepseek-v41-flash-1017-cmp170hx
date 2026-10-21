#!/usr/bin/env python3
"""One chat request (non-streaming). Usage: examples/chat.py "your question" [--port 8000] [--effort none|low|high|xhigh|max]"""
import argparse
import json
import urllib.request

ap = argparse.ArgumentParser()
ap.add_argument("prompt")
ap.add_argument("--port", default="8000")
ap.add_argument("--effort", default="high")
ap.add_argument("--max-tokens", type=int, default=4096)
a = ap.parse_args()
body = {"model": "deepseek-v4.1-flash", "messages": [{"role": "user", "content": a.prompt}],
        "max_tokens": a.max_tokens, "reasoning_effort": a.effort}
req = urllib.request.Request(f"http://127.0.0.1:{a.port}/v1/chat/completions", json.dumps(body).encode(),
                             {"Content-Type": "application/json"})
msg = json.load(urllib.request.urlopen(req, timeout=3600))["choices"][0]["message"]
if msg.get("reasoning"):
    print("--- reasoning ---\n" + msg["reasoning"] + "\n--- answer ---")
print(msg.get("content") or "")
