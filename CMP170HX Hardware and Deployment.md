# CMP 170HX — Hardware & vLLM Deployment

## What is the CMP 170HX?

The NVIDIA CMP 170HX is an ex-mining card built on the **GA100 die** (the same
silicon as an A100), compute capability **sm_80**, with 64 GB of HBM2e at
~1.6 TB/s. NVIDIA ships it with firmware/driver restrictions:

| Property | Stock | After the unlock |
|---|---:|---:|
| VRAM | 8,192 MiB | **65,536 MiB** |
| PCIe link | Gen1 | **Gen2 x16** |
| SM throughput | reduced | full |

The unlock used in this project is a **patched kernel driver module** (no vBIOS
flash — nothing is ever written to the card ROM). See the companion repo
(`cmp170hx/` → `docs/02-gpu-unlock.md`) for the driver patch and the cold-reboot
requirement.

## Test rig

| Component | Value |
|---|---|
| GPU(s) | 8x NVIDIA CMP 170HX (used as independent GPU groups) |
| GPU VRAM | 65,536 MiB each (unlocked) |
| GPU link | PCIe Gen2 x16 |
| Compute capability | 8.0 |
| Driver | 610.43.02 (CUDA UMD 13.3) |
| CPU | 2x Intel Xeon E5-2683 v4 (16c/32t each, 2 NUMA nodes) |
| System RAM | 251 GB |
| OS | AlmaLinux 10.2 / kernel 6.12 |
| Engine | vLLM patched fork (local image) |

## Why pipeline parallel (and no tensor parallel)

- **Pipeline parallel** is used throughout. On this hardware PP prefill is **6.6x
  faster** than TP for the MoE model, and TP is also arithmetically impossible for
  DeepSeek-V4-Flash (64 attention heads and 256 experts don't divide evenly).
- For the **dense** Qwen3.8-27B, TP=2 is arithmetic-friendly (24 Q / 4 KV
  heads, no experts) — and TP=2 happens to be the only way to combine with MTP
  speculative decoding in this vLLM fork (the Qwen3.5 MTP drafter does not
  implement the pipeline-parallel interface; PP + MTP is rejected at load).

## DeepSeek-V4-Flash deployment

- 4x CMP 170HX, PP=4 + DSpark speculative decoding (5 draft tokens).
- Weights: DeepSeek-V4-Flash-0731, 155.4 GiB bf16, 48 shards.
- 1M-token context (`DSV4_LOGITS_ROW_CHUNK=64` is the context-ceiling fix).
- See `Model_Reports/benchmark_DeepSeek-V4-Flash.md` for results.

```bash
vllm serve /model \
  --served-model-name=deepseek-v4-flash \
  --pipeline-parallel-size=4 \
  --kv-cache-dtype=fp8 --block-size=256 \
  --max-model-len=1048576 --max-num-batched-tokens=2048 \
  --trust-remote-code --gpu-memory-utilization=0.93 --max-num-seqs=8 \
  --no-enable-flashinfer-autotune \
  --tokenizer-mode=deepseek_v4 --reasoning-parser=deepseek_v4 \
  --tool-call-parser=deepseek_v4 \
  --speculative-config='{"method":"dspark","num_speculative_tokens":5}'
# env: NVIDIA_VISIBLE_DEVICES=<4 GPU IDs>  VLLM_PP_LAYER_PARTITION=12,12,12,7
#      DSV4_LOGITS_ROW_CHUNK=64  VLLM_WORKER_MULTIPROC_METHOD=spawn
```

## Qwen3.8-27B deployment

- 2x CMP 170HX, TP=2 + MTP speculative decoding (3 draft tokens — single MTP
  layer re-applied per step).
- Weights: Qwen3.8-27B, 43 GiB bf16, 18 shards (dense Qwen3.5-arch,
  hybrid linear-attention + 16 full-attention layers, native VLM).
- `max-model-len 262144` (= checkpoint `max_position_embeddings`).
- See `Model_Reports/benchmark_Qwen3.8-27B.md` and
  `benchmark_Qwen3.8-27B_MTP-sweep.md` for results.

```bash
vllm serve /model \
  --served-model-name=qwen3.8-27b \
  --tensor-parallel-size=2 \
  --kv-cache-dtype=fp8 --block-size=256 \
  --max-model-len=262144 --max-num-batched-tokens=2048 \
  --trust-remote-code --gpu-memory-utilization=0.93 --max-num-seqs=8 \
  --no-enable-flashinfer-autotune \
  --enable-auto-tool-choice --tool-call-parser=qwen3_coder \
  --reasoning-parser=qwen3 \
  --speculative-config='{"method":"mtp","num_speculative_tokens":3}'
# env: NVIDIA_VISIBLE_DEVICES=<2 GPU IDs>  VLLM_WORKER_MULTIPROC_METHOD=spawn
```

## Reasoned-effort / thinking notes (Qwen3.8-27B)

- The Qwen3.8 chat template accepts only `xhigh` (default) / `medium` / `low`.
  `high` is **not** valid and fails with `Unexpected reasoning effort high`.
- `reasoning_effort: none` (or `enable_thinking: false`) disables thinking.
- These arrive via top-level `reasoning_effort` or `chat_template_kwargs`.
