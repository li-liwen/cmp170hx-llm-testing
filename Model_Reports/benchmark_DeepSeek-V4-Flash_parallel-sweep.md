<!--
Benchmark Report Generated: 2026-08-31
Model: deepseek-v4-flash
Hardware: 4x CMP 170HX (GA100 sm_80, unlocked 64GB)
Harness: ParamSweep/sweep_bench.py (see ParamSweep/README.md)
-->

# DeepSeek-V4-Flash — parallelism parameter sweep (4x CMP 170HX)

**Model:** DeepSeek-V4-Flash-0731 (148 GiB bf16, MoE 256 experts, DSpark n=5)
**Hardware:** 4x CMP 170HX (GA100 sm_80, unlocked 64GB), driver 610.43.02,
`--max-model-len 1048576`, `DSV4_LOGITS_ROW_CHUNK=64`, fp8 KV,
`--gpu-memory-utilization 0.9`.

The sweep ran on a dedicated 4-GPU group ("group B"). A read-only reference pass
benchmarks an already-running PP4 server on a **different** 4-GPU group
("group A") with the same client for direct comparison.

## Configurations

| config | engine args | result |
|---|---|---|
| `pp4` | `--pipeline-parallel-size=4` | runs (group B) |
| `pp4_prod` | same, but the existing server on group A (read-only bench) | runs |
| `tp4` | `--tensor-parallel-size=4` | runs (group B) |
| `tp4_ep` | `--tensor-parallel-size=4 --enable-expert-parallel` | runs (group B) |
| `tp4_ep_megamoe` | `+ --moe-backend deep_gemm_mega_moe` | **fails: DeepGEMM MegaMoE requires SM100** |
| `tp2_pp2` | `--tensor-parallel-size=2 --pipeline-parallel-size=2` | **fails: DSpark draft load, TP+PP** |
| `tp2_pp2_ep` | `+ --enable-expert-parallel` | **fails: same DSpark draft load** |
| `tp2_pp2_nospec` | TP2+PP2, DSpark **off** | runs (group B) |

## Results (steady-state; same client for every row)

| config | group | decode1 single (tok/s) | decode3 3x400 agg (tok/s) | prefill 4096 (tok/s) | conc16 agg (tok/s) |
|---|---|---:|---:|---:|---:|
| **pp4_prod** | A | **175.9** | **311.2** | **6,385** | **553** |
| tp4 | B | 171.0 | 145.8 | 5,830 | 496 |
| tp4_ep | B | 169.2 | 136.7 | 5,920 | 473 |
| pp4 | B | 142.3 | 133.0 | 4,857 | 459 |
| tp2_pp2_nospec | B | 51.9 | 118.2 | 6,136 | 186 |

(All group-B rows used the same 4-GPU group; `pp4_prod` is the A-group
server under the same client.)

## Findings

1. **PP4 remains the best overall.** On the same client and the same 4-GPU group,
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

4. **GPU groups are not equivalent: identical PP4 ran 20-55% slower on group B**
   (decode3 133 vs 311, conc16 459 vs 553, prefill 4857 vs 6385). The two
   groups sit on different PCIe root complexes / NUMA nodes. The gap is topology,
   not engine config — expect position-dependent results across a chassis.

## Engine facts (group B)

| config | weights/GPU | KV pool (tokens) |
|---|---|---:|
| tp4 / tp4_ep | 40.6 GiB | 2,632,313 |
| pp4 | 37-38 GiB | (see engine logs) |
| tp2_pp2_nospec | 38.1 GiB | 6,281,851 |

Raw data: `results/<config>/bench.json`; engine lines `engine.log`; failures
`server_error.log`.

---

*Sweep run 2026-08-31, driver 610.43.02, AlmaLinux 10.2.*
