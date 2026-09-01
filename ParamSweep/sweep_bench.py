#!/usr/bin/env python3
"""
sweep_bench.py — lightweight, dependency-free benchmark client for the vLLM
parameter sweep on CMP 170HX.

Talks to a vLLM OpenAI-compatible server (/v1/completions) and measures the
four numbers used to compare parallelism configurations:

  decode1    : single stream,  128 -> 1024 tokens   (per-stream decode tok/s)
  decode3    : 3 streams,    128 -> 400 tokens each (aggregate tok/s, mirrors
               the production deepseek-v4 "decode3" harness)
  prefill    : single stream, 4096 -> 1 token, streaming, TTFT -> prefill tok/s
  conc16     : 16 streams,   512 -> 256 tokens each (aggregate tok/s)

Usage:
    VLLM_API_KEY=... python3 sweep_bench.py --base-url http://127.0.0.1:PORT
    VLLM_API_KEY=... python3 sweep_bench.py --base-url http://127.0.0.1:PORT --skip conc16

Prints a JSON object with all results to stdout. No external dependencies.
"""

import argparse
import json
import os
import random
import threading
import time
import urllib.request
from dataclasses import dataclass, asdict
from typing import Optional


def _post_stream(base_url: str, key: str, body: dict):
    """POST a streaming request; returns (ttft_s, total_s, output_tokens)."""
    req = urllib.request.Request(
        base_url + "/v1/completions",
        data=json.dumps(body).encode(),
        headers={
            "Authorization": f"Bearer {key}",
            "Content-Type": "application/json",
        },
    )
    t0 = time.time()
    first_tok_time = None
    with urllib.request.urlopen(req, timeout=1800) as resp:
        for raw in resp:
            if first_tok_time is None:
                first_tok_time = time.time() - t0
    total = time.time() - t0
    return first_tok_time or total, total


def _post_nonstream(base_url: str, key: str, body: dict):
    """POST a non-streaming request; returns (total_s, output_tokens)."""
    req = urllib.request.Request(
        base_url + "/v1/completions",
        data=json.dumps(body).encode(),
        headers={
            "Authorization": f"Bearer {key}",
            "Content-Type": "application/json",
        },
    )
    t0 = time.time()
    with urllib.request.urlopen(req, timeout=1800) as resp:
        data = json.load(resp)
    total = time.time() - t0
    return total, data["usage"]["completion_tokens"]


def gen_prompt(n: int, seed: int = 0) -> str:
    rng = random.Random(seed)
    words = ["alpha", "bravo", "charlie", "delta", "echo", "foxtrot",
             "golf", "hotel", "india", "juliet", "kilo", "lima",
             "mike", "november", "oscar", "papa"]
    # ~4 tokens/word is a rough heuristic; the server reports real usage anyway.
    return " ".join(rng.choice(words) for _ in range(max(1, n // 4)))


def measure_decode1(base, key, prompt_len=128, out_len=1024):
    body = {
        "model": "deepseek-v4-flash",
        "prompt": gen_prompt(prompt_len),
        "max_tokens": out_len,
        "temperature": 0.0,
        "stream": True,
    }
    ttft, total = _post_stream(base, key, body)
    n = out_len
    decode_tok_s = (n - 1) / max(1e-6, total - ttft)
    return {
        "output_tokens": n,
        "ttft_s": round(ttft, 3),
        "total_s": round(total, 3),
        "decode_tok_s": round(decode_tok_s, 1),
    }


def measure_decode3(base, key, out_len=400):
    results = [None] * 3

    def run(i):
        body = {
            "model": "deepseek-v4-flash",
            "prompt": gen_prompt(128, seed=i),
            "max_tokens": out_len,
            "temperature": 0.0,
            "stream": True,
        }
        _, total = _post_stream(base, key, body)
        results[i] = total

    threads = [threading.Thread(target=run, args=(i,)) for i in range(3)]
    t0 = time.time()
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    elapsed = time.time() - t0
    return {
        "total_output_tokens": 3 * out_len,
        "elapsed_s": round(elapsed, 3),
        "aggregate_tok_s": round(3 * out_len / elapsed, 1),
        "per_stream_tok_s": round(out_len / (elapsed), 1),
    }


def measure_prefill(base, key, prompt_len=4096):
    body = {
        "model": "deepseek-v4-flash",
        "prompt": gen_prompt(prompt_len),
        "max_tokens": 1,
        "temperature": 0.0,
        "stream": True,
    }
    ttft, total = _post_stream(base, key, body)
    return {
        "prompt_tokens": prompt_len,
        "ttft_s": round(ttft, 3),
        "prefill_tok_s": round(prompt_len / max(1e-6, ttft), 1),
    }


def measure_concurrency(base, key, n=16, out_len=256, prompt_len=512):
    counts = [0] * n

    def run(i):
        body = {
            "model": "deepseek-v4-flash",
            "prompt": gen_prompt(prompt_len, seed=i),
            "max_tokens": out_len,
            "temperature": 0.0,
            "stream": True,
        }
        _, total = _post_stream(base, key, body)
        counts[i] = total

    threads = [threading.Thread(target=run, args=(i,)) for i in range(n)]
    t0 = time.time()
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    elapsed = time.time() - t0
    return {
        "concurrency": n,
        "total_output_tokens": n * out_len,
        "elapsed_s": round(elapsed, 3),
        "aggregate_tok_s": round(n * out_len / elapsed, 1),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-url", required=True)
    ap.add_argument("--api-key", default=None)
    ap.add_argument("--skip", nargs="*", default=[],
                  help="skip tests: decode1 decode3 prefill conc16")
    args = ap.parse_args()

    key = args.api_key or os.environ.get("VLLM_API_KEY")
    if not key:
        print(json.dumps({"error": "VLLM_API_KEY not set"}))
        return 1

    out = {"base_url": args.base_url}
    if "decode1" not in args.skip:
        out["decode1"] = measure_decode1(args.base_url, key)
    if "decode3" not in args.skip:
        out["decode3"] = measure_decode3(args.base_url, key)
    if "prefill" not in args.skip:
        out["prefill"] = measure_prefill(args.base_url, key)
    if "conc16" not in args.skip:
        out["conc16"] = measure_concurrency(args.base_url, key, n=16)

    print(json.dumps(out, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
