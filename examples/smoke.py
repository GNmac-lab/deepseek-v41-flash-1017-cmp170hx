#!/usr/bin/env python3
"""Smoke test against the running server: one chat, one tool call, one image. Usage: examples/smoke.py [--port 8000]"""
import argparse
import base64
import json
import struct
import urllib.request
import zlib

ap = argparse.ArgumentParser()
ap.add_argument("--port", default="8000")
ap.add_argument("--model", default="deepseek-v4.1-flash")
a = ap.parse_args()
URL = f"http://127.0.0.1:{a.port}/v1/chat/completions"


def chat(messages, **kw):
    body = {"model": a.model, "messages": messages, "temperature": 0, "max_tokens": 512, **kw}
    req = urllib.request.Request(URL, json.dumps(body).encode(), {"Content-Type": "application/json"})
    return json.load(urllib.request.urlopen(req, timeout=600))["choices"][0]["message"]


def png(w, h, rgb):
    raw = b"".join(b"\x00" + bytes(rgb) * w for _ in range(h))
    chunk = lambda t, d: struct.pack(">I", len(d)) + t + d + struct.pack(">I", zlib.crc32(t + d) & 0xFFFFFFFF)
    return (b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", struct.pack(">IIBBBBB", w, h, 8, 2, 0, 0, 0))
            + chunk(b"IDAT", zlib.compress(raw)) + chunk(b"IEND", b""))


ok = True
m = chat([{"role": "user", "content": "Is 221 prime? Answer yes or no with one reason."}], reasoning_effort="none")
print("chat :", m["content"].strip()[:80]); ok &= "13" in m["content"] and "17" in m["content"]

tools = [{"type": "function", "function": {"name": "get_weather", "description": "Current weather for a city",
          "parameters": {"type": "object", "properties": {"city": {"type": "string"}}, "required": ["city"]}}}]
m = chat([{"role": "user", "content": "What's the weather in Lisbon?"}], tools=tools, reasoning_effort="none")
tc = (m.get("tool_calls") or [{}])[0].get("function", {})
print("tool :", tc); ok &= tc.get("name") == "get_weather" and "Lisbon" in tc.get("arguments", "")

img = "data:image/png;base64," + base64.b64encode(png(64, 64, (220, 20, 20))).decode()
m = chat([{"role": "user", "content": [{"type": "text", "text": "What colour is this image? One word."},
                                       {"type": "image_url", "image_url": {"url": img}}]}], reasoning_effort="none")
print("image:", m["content"].strip()[:40]); ok &= "red" in m["content"].lower()
print("SMOKE", "PASS" if ok else "FAIL")
raise SystemExit(0 if ok else 1)
