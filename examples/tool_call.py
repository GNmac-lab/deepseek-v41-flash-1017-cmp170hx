#!/usr/bin/env python3
"""A tool-calling round trip: the model asks for a tool, we answer it, the model replies. Usage: examples/tool_call.py [--port 8000]"""
import argparse
import json
import urllib.request

ap = argparse.ArgumentParser()
ap.add_argument("--port", default="8000")
a = ap.parse_args()
URL = f"http://127.0.0.1:{a.port}/v1/chat/completions"
tools = [{"type": "function", "function": {"name": "get_weather", "description": "Current weather for a city",
          "parameters": {"type": "object", "properties": {"city": {"type": "string"}}, "required": ["city"]}}}]


def chat(messages):
    body = {"model": "deepseek-v4.1-flash", "messages": messages, "tools": tools, "max_tokens": 1024,
            "reasoning_effort": "none"}
    req = urllib.request.Request(URL, json.dumps(body).encode(), {"Content-Type": "application/json"})
    return json.load(urllib.request.urlopen(req, timeout=600))["choices"][0]["message"]


messages = [{"role": "user", "content": "Should I take an umbrella in Lisbon today?"}]
m = chat(messages)
call = m["tool_calls"][0]
print("model calls:", call["function"]["name"], call["function"]["arguments"])
messages += [m, {"role": "tool", "tool_call_id": call["id"],
                 "content": json.dumps({"city": "Lisbon", "forecast": "light rain after 3 pm", "temp_c": 17})}]
print("model says:", chat(messages)["content"])
