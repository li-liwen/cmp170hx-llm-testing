# cmp170hx-llm-testing

Documentation and benchmark results for running popular LLM inference engines on
**NVIDIA CMP 170HX** GPUs. This repository is maintained publicly for reference;
do not put secrets, API keys, or credentials in any file here.

## What is a CMP 170HX?

The CMP 170HX is an ex-mining card built on the **GA100** die (same silicon as
an A100), sm_80, with 64 GB of HBM2e (~1.6 TB/s). NVIDIA ships it locked
down: 8 GB VRAM, reduced SMs, PCIe Gen1. The machines here run a **patched
kernel driver** (no vBIOS flash) that unlocks **65,536 MiB** per card and Gen2
x16 links. Details: [`CMP170HX Hardware and Deployment.md`](CMP170HX%20Hardware%20and%20Deployment.md).

## Hardware

* 8x NVIDIA CMP 170HX (GA100 sm_80, 64 GB unlocked each, PCIe Gen2 x16)
* 2x Intel Xeon E5-2683 v4 (2 NUMA nodes), 251 GB RAM
* AlmaLinux 10.2 / kernel 6.12, driver 610.43.02
* vLLM patched fork (local image)

## Models & results

| Model | GPUs | Config | Headline numbers | Report |
|---|---|---|---|---|
| **DeepSeek-V4-Flash-0731** (~284B/13B MoE) | 4 | PP=4 + DSpark n=5, 1M ctx | 109.7 tok/s decode aggregate; 5,440 tok/s prefill @50k; needle PASS to 700k | [`Model_Reports/benchmark_DeepSeek-V4-Flash.md`](Model_Reports/benchmark_DeepSeek-V4-Flash.md) |
| **Qwen3.8-27B** (dense 27B, hybrid linear-attn, VLM) | 2 | TP=2 + MTP n=3, 262k ctx | ~48 tok/s single-stream decode; full TTFT/TPOT/concurrency in report | [`Model_Reports/benchmark_Qwen3.8-27B.md`](Model_Reports/benchmark_Qwen3.8-27B.md) |
| Qwen3.8-27B parallelism sweep | 2 | PP2 / TP2+MTP n=1..3 | MTP n=3 = 1.86x PP2 decode | [`Model_Reports/benchmark_Qwen3.8-27B_MTP-sweep.md`](Model_Reports/benchmark_Qwen3.8-27B_MTP-sweep.md) |
| **DeepSeek-V4-Flash parallel sweep** | 4 | PP4 / TP4 / TP4+EP / TP2+PP2 | PP4 best; TP4 ≈ PP4 single-stream but worse aggregate; EP=MegaMoE is SM100-only; DSpark+TP+PP fails | [`Model_Reports/benchmark_DeepSeek-V4-Flash_parallel-sweep.md`](Model_Reports/benchmark_DeepSeek-V4-Flash_parallel-sweep.md) |
| **GLM-5.3-Flash** (~320B/18B MoE, W4A16 AutoRound) | 4 | vLLM fork PP=4, 32k ctx (AutoRound INT4) | 39.5 tok/s single-stream decode; 143.6 tok/s best aggregate @c16; 84.6 tok/s @16k ctx | [`Model_Reports/benchmark_GLM-5.3-Flash-AutoRound.md`](Model_Reports/benchmark_GLM-5.3-Flash-AutoRound.md) |
| **DeepSeek-V4-Flash-0731** on the 2nd GPU group | 4 | PP=4 + DSpark n=5, 1M ctx, fp8 KV | 7.72M-token KV; 9x1M fills stable; 4x~661k concurrent OK; KV-offload push 4.6-4.7 GB/s | [`Model_Reports/deepseek-v4-flash-cmp170hx-gpu47.md`](Model_Reports/deepseek-v4-flash-cmp170hx-gpu47.md) |

Raw structured data: [`Model_Reports/json_data/`](Model_Reports/json_data/).

GLM-5.3-Flash image build + serving details:
[`Model_Reports/GLM-5.3-Flash-deployment.md`](Model_Reports/GLM-5.3-Flash-deployment.md).

Known hardware faults / per-card checks for CMP 170HX units:
[`Model_Reports/CMP170HX-GPU6-MMU-Fault.md`](Model_Reports/CMP170HX-GPU6-MMU-Fault.md).

## Notable findings

* **PP beats TP on this hardware** for the MoE model — and TP is arithmetically
  impossible for it anyway (64 heads / 256 experts). Measured TP4 matches PP4 at
  single-stream decode but loses on multi-stream aggregate; true expert parallel
  (MegaMoE) requires SM100 and won't start on this sm_80 card.
* **MTP is the decode accelerator for dense Qwen3.5-arch models.** A single MTP
  layer is re-applied per draft step; n=3 hit ~48 tok/s (1.86x plain PP2).
* The unlock's driver plus this vLLM fork enables the 1M-token context ceiling
  for DeepSeek-V4-Flash (`DSV4_LOGITS_ROW_CHUNK` fix); needle retrieval passes
  at 100k/300k/500k/700k.
* **Per-card faults happen: see `CMP170HX-GPU6-MMU-Fault.md`** — one GPU on our
  second group faults (Xid 31) only under GLM long-prefill with CUDA graphs
  (eager OK, DSV4 OK, raw big allocs OK); ECC telemetry is `[N/A]` on these
  cards. Check every card, not just one.
* vLLM native CPU KV offload in these forks is **store-only in practice**:
  GPU→CPU push measured 4.6-6.65 GB/s, but CPU→GPU load is never served
  (re-requests recompute). GLM fp8 KV is unsupported on SM80 (sparse-MLA
  backends reject it); DSV4 fp8 KV is fine.
* Qwen3.8 reasoning effort: `xhigh` / `medium` / `low` (and `none` = no
  thinking). `high` is **invalid** and errors — important when wiring into proxies
  (e.g. LiteLLM/OpenWebUI) that send OpenAI-style `high`.
* The GPU groups on our rig perform differently: identical PP4 ran 20-55% slower
  on one group than on another (different PCIe root complexes / NUMA nodes —
  topology, not engine config). Expect position-dependent results.

## Benchmark harness

[`BenchAndReport.py`](BenchAndReport.py) is the self-documenting suite used for
reports in `Model_Reports/` (adapted from the MI100 project; adds `--api-key` /
`$VLLM_BENCH_API_KEY` auth support for servers that enforce a token). It drives
`vllm bench serve` and asks the served model to write its own report.

```bash
VLLM_BENCH_API_KEY=$YOUR_KEY python3 BenchAndReport.py \
  --model qwen3.8-27b --base-url http://localhost:8000 \
  --tokenizer /path/to/weights --hardware "2xCMP 170HX (GA100, sm_80)" \
  --scaffolded-report --save-results -o Model_Reports/benchmark_qwen3.8-27b.md
```

DeepSeek-V4-Flash numbers were captured with the cmp170hx repo's own harnesses
(`bench_decode3.py`, `bench_prefill.py`, `bench_longctx.py`,
`bench_needle.py`, `bench_conc_needle.py`, etc.).

## Related

* The unlock, patched vLLM build, and full reproduction runbook live in the
  companion repo `cmp170hx` (private; this repo only publishes results and
  deployment configs that are safe to share).
