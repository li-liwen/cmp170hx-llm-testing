<!--
Benchmark Report Generated: 2026-08-31
Model: deepseek-v4-flash
Hardware: cards 4-7 = 4x CMP 170HX (GA100 sm_80, unlocked 64GB), NUMA node 1
Harness: ParamSweep/sweep_bench.py (see ParamSweep/README.md)
-->

# DeepSeek-V4-Flash — parallelism parameter sweep (4x CMP 170HX)

**Model:** DeepSeek-V4-Flash-0731 (148 GiB bf16, MoE 256 experts, DSpark n=5)
**Hardware:** 4x CMP 170HX on **cards 4-7** (NUMA node 1), production on 0-3
kept untouched. Driver 610.43.02, `dsv4-a100:devel` vLLM fork,
`--max-model-len 1048576`, `DSV4_LOGITS_ROW_CHUNK=64`, fp8 KV,
`--gpu-memory-utilization 0.9`.

A read-only reference pass benchmarks the **production PP4 server on cards 0-3**
with the same client for direct comparison.

## Configurations

| config | engine args | result |
|---|---|---|
| `pp4` | `--pipeline-parallel-size=4` | runs |
| `pp4_prod` | same, but on cards 0-3 (production, read-only bench) | runs |
| `tp4` | `--tensor-parallel-size=4` | runs |
| `tp4_ep` | `--tensor-parallel-size=4 --enable-expert-parallel` | runs |
| `tp4_ep_megamoe` | `+ --moe-backend deep_gemm_mega_moe` | **fails: DeepGEMM MegaMoE requires SM100** |
| `tp2_pp2` | `--tensor-parallel-size=2 --pipeline-parallel-size=2` | **fails: DSpark draft load, TP+PP** |
| `tp2_pp2_ep` | `+ --enable-expert-parallel` | **fails: same DSpark draft load** |
| `tp2_pp2_nospec` | TP2+PP2, DSpark **off** | runs |

## Results (steady-state; same client for every row)

| config | GPUs | decode1 single (tok/s) | decode3 3x400 agg (tok/s) | prefill 4096 (tok/s) | conc16 agg (tok/s) |
|---|---:|---:|---:|---:|---:|
| **pp4_prod** | 0-3 | **175.9** | **311.2** | **6,385** | **553** |
| tp4 | 4-7 | 171.0 | 145.8 | 5,830 | 496 |
| tp4_ep | 4-7 | 169.2 | 136.7 | 5,920 | 473 |
| pp4 | 4-7 | 142.3 | 133.0 | 4,857 | 459 |
| tp2_pp2_nospec | 4-7 | 51.9 | 118.2 | 6,136 | 186 |

(All rows are cards 4-7 except `pp4_prod`.)

## Findings

1. **PP4 remains the best overall.** On the same client and the same 4-card group,
   nothing beats PP4. TP4 matches PP4 for single-stream decode (~170 vs 176) and
   is close on prefill, but pays a large multi-stream decode penalty (decode3/con-16
   ~135-150 vs 311) — the per-layer all-reduce over PCIe Gen2 eats the aggregate.

2. **`--enable-expert-parallel` alone changes nothing on sm_80.** TP4 vs TP4+EP are
   within noise (145.8 vs 136.7 decode3; 5830 vs 5920 prefill). True expert
   parallel (all2all / MegaMoE) is gated behind `deep_gemm_mega_moe`, which
   refuses to start on this hardware:
   `NotImplementedError: DeepGEMM MegaMoE requires SM100 GPUs.`

3. **DSpark drafter cannot load with TP+PP** (TP2+PP2 and TP2+PP2+EP):
   `RuntimeError: The size of tensor a (64640) must match the size of tensor b
   (129280)` in `dspark/utils.py:_load_embed_from_checkpoint`. Without DSpark,
   TP2+PP2 boots but is by far the slowest config (52 single / 186 conc16) — do
   not use.

4. **Card group 4-7 is slower than 0-3 for the identical PP4 config**
   (decode3 133 vs 311, conc16 459 vs 553, prefill 4857 vs 6385). The two
   groups sit on different PCIe root complexes and different NUMA nodes (0-3 → node 0,
   4-7 → node 1; inter-group hops = SYS). The gap is topology, not engine config.

## Engine facts (cards 4-7)

| config | weights/GPU | KV pool (tokens) |
|---|---:|---:|
| tp4 / tp4_ep | 40.6 GiB | 2,632,313 |
| pp4 | 37-38 GiB | (see engine logs) |
| tp2_pp2_nospec | 38.1 GiB | 6,281,851 |

Raw data: `results/<config>/bench.json`; engine lines `engine.log`; failures
`server_error.log`.

---

*Sweep run 2026-08-31, cards 4-7, driver 610.43.02, AlmaLinux 10.2.*
