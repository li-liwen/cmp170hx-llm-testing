<!--
DeepSeek-V4-Flash family on GPU group 4-7 of an 8x CMP 170HX rig. Sanitized (no
host paths/IPs/users). Recorded 2026-09-05. Full raw logs kept locally beside
this repo's sibling checkout (not public).
-->

# DeepSeek-V4-Flash on the "second" GPU group (indices 4-7, NUMA 1): 1M window, KV offload, concurrency, and the Vision-Exp checkpoint

## 1. DeepSeek-V4-Flash-Vision-Exp — NOT deployable with the available engine

The experimental vision checkpoint (`DeepseekV4ForCausalLM`, fp8 e4m3, 1,048,576
max position, ~150 GiB) cannot be loaded by the DeepSeek-V4 vLLM fork used for the
successful 0731 text deployment. The fork's `DeepseekV4ForCausalLM` implements no
vision/aligner modules, and weight loading is strict, e.g.:

```
ValueError: There is no module or parameter named 'aligner' in DeepseekV4ForCausalLM.
RuntimeError: Engine core initialization failed. ... Failed core proc(s): {}
```

`ignore_unexpected_*` is not CLI-configurable, there is no vision-capable fork/image
on the machine, and the checkpoint's bundled reference runtime requires a TP4 conversion
with `--expert-dtype fp4` (FP4 tensor cores are sm100-only; absent on GA100 sm_80).
=> Vision-Exp needs a vision-bearing fork; not a CMP limitation per se, just the software.

## 2. DeepSeek-V4-Flash-0731 on GPU indices 4-7 (production flags)

Same server flags as the long-running deployment on GPU 0-3: PP=4 (partition
12,12,12,7), fp8 KV, block 256, 1M ctx, 0.93 util, max-num-seqs 8,
max-num-batched-tokens 2048, DSpark n=5, tokenizer/parsers deepseek_v4.

| Metric | value |
|---|---|
| GPU KV cache | **7,724,275 tokens** (7.37x concurrency at 1M) |
| Max single-stream context | ~7.72M tokens (checkpoint caps usable at 1.05M) |
| 1M-ctx fill (sequential, ~1.02M tok, x9) | 510-515 s each (~2.0k tok/s prefill), 0 errors |
| 4x ~661k-token concurrent | all OK; wall 1834 s; ~1.53k tok/s aggregate; 0 errors |

Note: GPU6 of this group is the card documented in
[`CMP170HX-GPU6-MMU-Fault.md`](CMP170HX-GPU6-MMU-Fault.md) (GLM-only fault;
DSV4 on GPU6 is unaffected — verified: 783k-token prefill, plus the above runs).

## 3. KV-cache CPU offload (vLLM native `--kv-offloading-size`, backport lineage)

Same code in both forks (DSV4 and GLM-5.3 builds). Measured on these GPUs
(12 GiB CPU region; larger regions need a `/dev/shm` that has room):

| Transfer | DSV4 (fp8 KV) | GLM (bf16 KV) |
|---|---:|---:|
| GPU -> CPU (push/store) | **4.6-4.7 GB/s** (92 GB / 9x1M fills) | **6.65 GB/s** (34 GB / 11 GB / 10.7 GB runs) |
| CPU -> GPU (load/pull) | **0 (never observed)** | **0 (never observed)** |
| Re-hit of an evicted prompt | re-PREFILLS (TTFT unchanged) | re-PREFILLS (TTFT unchanged) |

Conclusion: in this fork generation the connector **stores to CPU but does not serve
reads back** — evicted blocks are recomputed; no TTFT/throughput benefit. Flag for
upstream. `--kv-offloading-size` also demands a `cpu_bytes_to_use` shared region in
`/dev/shm`; a 50 GiB region can time out when `/dev/shm` is heavily occupied, and
NCCL `psm_*` segments plus other tens of GiB can exhaust a default 50%-of-RAM
tmpfs. Growing `/dev/shm` (`mount -o remount,size=...`) is non-destructive.
(No disk offload tested — CPU RAM only, as requested.)

## 4. Aux cascade of findings relevant to CMP buyers

- fp8 KV: works for DSV4 (sparse-MLA accepts fp8); is **unsupported for
  Glm5Next on SM80** in the GLM fork (backend selection rejects fp8 KV).
- Long-context stability on group 4-7 (DSV4): ~9 MiB-token (fp8) KV; 1M-ctx
  workloads stable; GPU6 is the lone fault-prone unit (see the GPU6 report).
- Sequential 1M-window needle (GLM, bf16 KV, eager): PASS at 0/25/50/75/95%.

## 5. Recommended production setting (GLM-5.3-Flash on CMP 170HX)

`--kv-cache-dtype auto` (fp8 unavailable on SM80 for Glm5Next), `--enforce-eager`
for long prompts on GPU4-7 until the marginal GPU6 unit is replaced; no CPU KV
offload benefit with the current fork (store-only). Vision-Exp: wait for a
vision-capable DeepSeek fork.
