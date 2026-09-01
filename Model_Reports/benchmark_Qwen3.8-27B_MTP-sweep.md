<!--
Benchmark Report Generated: 2026-08-31
Model: qwen3.8-27b
Hardware: 2xCMP 170HX (GA100, sm_80, unlocked 64GB)
Note: decode-only measurements, single 1,024-token stream, OpenAI /chat/completions (non-streaming).
-->

# Qwen3.8-27B — Parallelism & MTP speculative-decoding sweep (2x CMP 170HX)

**Question:** for the dense 27B model on two unlocked CMP 170HX cards, which
parallelism + speculative-decoding combination is fastest at single-stream decode?

**Answer:** TP=2 + MTP n=3 ≈ **48 tok/s**, ~1.9x the plain PP=2 baseline.

## Configurations tested

All on the same patched vLLM fork, two GPUs, bf16 weights,
fp8 KV cache, `--max-model-len 262144`, `--gpu-memory-utilization 0.93`.

| Config | Compute layout | Spec decode |
|---|---|---|
| PP2 (baseline) | pipeline parallel, 2 ranks (32/32 layers) | none |
| TP2 + MTP1 | tensor parallel, 2 ranks | MTP, 1 draft token |
| TP2 + MTP2 | tensor parallel, 2 ranks | MTP, 2 draft tokens |
| TP2 + MTP3 | tensor parallel, 2 ranks | MTP, 3 draft tokens |

Measured with a single non-streaming request: 67-token prompt,
1,024 generated tokens, `temperature=0`. The first run of each server is
cold (draft/cudagraph warmup); the steady-state figure comes from runs 2-3.

## Results

| Config | Decode tok/s (steady) | Run 2 | Run 3 | vs PP2 baseline |
|---|---:|---:|---:|---:|
| PP=2, no MTP | **25.7** | 25.7 | — | 1.00x |
| TP=2, MTP n=1 | **33.0** | 34.1 | 32.0 | 1.28x |
| TP=2, MTP n=2 | **44.8** | 45.2 | 44.5 | 1.74x |
| TP=2, MTP n=3 | **47.8** | 47.8 | 47.9 | **1.86x** |

## MTP draft acceptance (from /metrics)

The checkpoint ships **one** MTP layer; the engine re-applies it per speculative
step (`spec_step_idx % num_mtp_layers`), with an explicit engine warning that
n>1 lowers acceptance.

| Config | pos0 | pos1 | pos2 | overall |
|---|---:|---:|---:|---:|
| MTP n=1 | 77.2% | — | — | 77.2% |
| MTP n=2 | 74.5% | 47.0% | — | 60.7% |
| MTP n=3 | 75.0% | 48.8% | 29.7% | 51.2% |

## Take-aways

- MTP is the big win: n=3 is 1.86x the plain-PP2 decode rate.
- The 3rd draft token is only ~30% accepted, so n=2 (1.74x) is nearly as
  fast with 50% less draft compute — a good "best value" operating point.
- MTP requires TP (or PP=1); PP+MTP is rejected by this fork because the
  `Qwen3_5MTP` drafter doesn't implement the pipeline-parallel interface.
- These are **decode** numbers only; full TTFT/TPOT/prefill/concurrency
  characteristics of the shipped config (TP2 + MTP n=3) are in
  `benchmark_Qwen3.8-27B.md`.

---

*Captured 2026-08-31.*
