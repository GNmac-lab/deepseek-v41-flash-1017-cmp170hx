#!/usr/bin/env python3
"""HumanEval (164) and GSM8K (1,319) through the served model, scored on the answer after the reasoning (`content`).

Requests go out in fixed order, up to 8 at a time, temperature 0, with the server's default reasoning. HumanEval
allows 4,096 reply tokens and GSM8K 3,072; length-capped replies stay in the score. The datasets are read from the
Hugging Face datasets-server API on first use. Generated HumanEval code runs in a throwaway container with no
network (--network none, read-only root, 10 s per problem), never on the host.
Usage: bench/quality.py humaneval|gsm8k|all [--port 8000] [--model deepseek-v4.1-flash] [--image IMAGE]"""
import argparse
import concurrent.futures as cf
import json
import os
import re
import subprocess
import tempfile
import time
import urllib.error
import urllib.request
from pathlib import Path

HERE = Path(__file__).resolve().parent
SETS = {"humaneval": ("openai/openai_humaneval", "openai_humaneval", "test", 4096),
        "gsm8k": ("openai/gsm8k", "main", "test", 3072)}


def fetch_json(url, tries=6):
    """GET a JSON page; the datasets server sometimes answers 5xx or times out, so retry with backoff."""
    for i in range(tries):
        try:
            return json.load(urllib.request.urlopen(url, timeout=120))
        except (urllib.error.HTTPError, urllib.error.URLError, TimeoutError) as e:
            if i == tries - 1 or (isinstance(e, urllib.error.HTTPError) and e.code < 500 and e.code != 429):
                raise
            time.sleep(5 * 2 ** i)


def rows(dataset, config, split):
    cache = HERE / ".cache" / f"{dataset.replace('/', '__')}-{config}-{split}.json"
    if cache.exists():
        return json.loads(cache.read_text())
    out, off = [], 0
    while True:
        url = (f"https://datasets-server.huggingface.co/rows?dataset={dataset}&config={config}&split={split}"
               f"&offset={off}&length=100")
        page = fetch_json(url)
        out += [r["row"] for r in page["rows"]]
        off += len(page["rows"])
        if not page["rows"] or off >= page["num_rows_total"]:
            break
    cache.parent.mkdir(parents=True, exist_ok=True)
    cache.write_text(json.dumps(out))
    return out


def chat(base, model, prompt, max_tokens):
    body = {"model": model, "temperature": 0, "max_tokens": max_tokens,
            "messages": [{"role": "user", "content": prompt}]}
    req = urllib.request.Request(base + "/v1/chat/completions", json.dumps(body).encode(),
                                 {"Content-Type": "application/json"})
    r = json.load(urllib.request.urlopen(req, timeout=3600))
    ch = r["choices"][0]
    return ch["message"].get("content") or "", ch.get("finish_reason")


def run_all(base, model, prompts, max_tokens):
    with cf.ThreadPoolExecutor(8) as ex:
        return list(ex.map(lambda p: chat(base, model, p, max_tokens), prompts))


def humaneval(base, model, image):
    data = rows(*SETS["humaneval"][:3])
    prompts = [("Complete the following Python function. Reply with the complete function, including the "
                "signature, in one ```python code block.\n\n```python\n" + d["prompt"] + "```") for d in data]
    replies = run_all(base, model, prompts, SETS["humaneval"][3])
    progs = []
    for d, (content, _) in zip(data, replies):
        blocks = re.findall(r"```(?:python|py)?\n(.*?)```", content, re.S)
        code = max(blocks, key=len) if blocks else content
        if f"def {d['entry_point']}" not in code:
            code = d["prompt"] + code
        header = "\n".join(l for l in d["prompt"].splitlines() if l.startswith(("import ", "from ")))
        progs.append(header + "\n" + code + "\n\n" + d["test"] + f"\ncheck({d['entry_point']})\n")
    with tempfile.TemporaryDirectory() as td:
        Path(td, "progs.json").write_text(json.dumps(progs))
        Path(td, "run.py").write_text(
            "import json, subprocess, sys\n"
            "res = []\n"
            "for p in json.load(open('/work/progs.json')):\n"
            "    try:\n"
            "        r = subprocess.run([sys.executable, '-c', p], capture_output=True, timeout=10, cwd='/tmp')\n"
            "        res.append(r.returncode == 0)\n"
            "    except subprocess.TimeoutExpired:\n"
            "        res.append(False)\n"
            "print(json.dumps(res))\n")
        out = subprocess.run(["docker", "run", "--rm", "--network", "none", "--read-only", "--tmpfs", "/tmp",
                              "--memory", "4g", "--pids-limit", "256", "-v", f"{td}:/work:ro", "--entrypoint",
                              "python3", image, "/work/run.py"], capture_output=True, text=True, check=True)
    passed = json.loads(out.stdout.strip().splitlines()[-1])
    capped = sum(fr == "length" for _, fr in replies)
    print(f"HumanEval: {sum(passed)}/{len(passed)} (length-capped replies: {capped})")
    return passed


def gsm8k(base, model):
    data = rows(*SETS["gsm8k"][:3])
    prompts = [d["question"] + "\n\nGive the final answer as a number on the last line, as 'Answer: N'." for d in data]
    replies = run_all(base, model, prompts, SETS["gsm8k"][3])
    num = lambda s: s.replace(",", "").replace("$", "").strip().rstrip(".")
    ok = []
    for d, (content, _) in zip(data, replies):
        gold = num(d["answer"].split("####")[-1])
        m = re.findall(r"Answer:\s*\**\s*(-?[\d,]*\.?\d+)", content) or re.findall(r"-?[\d,]*\.?\d+", content)
        try:
            ok.append(bool(m) and float(num(m[-1])) == float(gold))
        except ValueError:
            ok.append(False)
    capped = sum(fr == "length" for _, fr in replies)
    print(f"GSM8K: {sum(ok)}/{len(ok)} (length-capped replies: {capped})")
    return ok


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("suite", choices=("humaneval", "gsm8k", "all"))
    ap.add_argument("--port", default=os.environ.get("PORT", "8000"))
    ap.add_argument("--model", default="deepseek-v4.1-flash")
    ap.add_argument("--image", default=os.environ.get("IMAGE", "ds41-cmp170hx:latest"),
                    help="any local image with python3, used as the no-network sandbox for HumanEval")
    ap.add_argument("--label", default="run")
    a = ap.parse_args()
    base = f"http://127.0.0.1:{a.port}"
    res = {}
    if a.suite in ("humaneval", "all"):
        res["humaneval"] = humaneval(base, a.model, a.image)
    if a.suite in ("gsm8k", "all"):
        res["gsm8k"] = gsm8k(base, a.model)
    out = HERE / "out" / f"{a.label}-quality.json"
    out.parent.mkdir(exist_ok=True)
    out.write_text(json.dumps(res))
    print("saved", out)


if __name__ == "__main__":
    main()
