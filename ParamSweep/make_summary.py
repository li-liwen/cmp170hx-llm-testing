#!/usr/bin/env python3
"""make_summary.py — assemble ParamSweep/results/*/bench.json into a summary table."""
import glob
import json
import os

ROWS = []
for f in sorted(glob.glob("results/*/bench.json")):
    cfg = f.split("/")[-2]
    try:
        r = json.load(open(f))
        if "error" in r:
            ROWS.append((cfg, "ERROR: " + str(r["error"]), "-", "-", "-", "-"))
            continue
        d1 = r.get("decode1", {}).get("decode_tok_s", "-")
        d3 = r.get("decode3", {}).get("aggregate_tok_s", "-")
        pf = r.get("prefill", {}).get("prefill_tok_s", "-")
        c16 = r.get("conc16", {}).get("aggregate_tok_s", "-")
        ROWS.append((cfg, "ok", d1, d3, pf, c16))
    except Exception as e:
        ROWS.append((cfg, "ERROR " + str(e), "-", "-", "-", "-"))

lines = [
    "# DeepSeek-V4-Flash parallel-config sweep — summary",
    "",
    "| config | status | decode1 (single, tok/s) | decode3 (3x400, agg tok/s) | prefill 4096 (tok/s) | conc16 (agg tok/s) |",
    "|---|---:|---:|---:|---:|---:|",
]
for cfg, st, d1, d3, pf, c16 in ROWS:
    lines.append(f"| {cfg} | {st} | {d1} | {d3} | {pf} | {c16} |")
lines.append("")
lines.append("Method: `sweep_bench.py` (see ParamSweep README). pp4_prod = read-only")
lines.append("benchmark of an existing PP4 server on a separate GPU group.")
open("results/SUMMARY.md", "w").write("\n".join(lines) + "\n")
print("\n".join(lines))
