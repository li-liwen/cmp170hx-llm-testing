<!--
GPU6 (one of eight) MMU-fault field report — CMP 170HX. Sanitized: no host names,
IPs, users, or local paths. All technical detail retained.
Recorded 2026-09-05.
-->

# Field report: recurrent Xid-31 MMU fault on one CMP 170HX under GLM long-prefill with CUDA graphs

## Summary

On an 8x **NVIDIA CMP 170HX** rig (GA100, sm_80, VRAM-unlocked 65,536 MiB/card,
PCIe Gen2 x16, patched kernel driver, NO vBIOS flash), one specific card
(physical GPU index **6**, PCI `0000:8B:00`, unique UUID) reproducibly throws
**`Xid 31` (MMU fault)** while GLM-5.3-Flash-W4A16-AutoRound runs a **long
prefill (>~150-300k prompt tokens) under default CUDA-graph capture** in the
community vLLM fork (`0.8.1+glm53a800`). The problem is **per-card, not
per-software**: the same GPU6 passes raw large allocations, works for hours at <=32k
context, and runs a full 783k-token DeepSeek-V4-Flash-0731 prefill with zero
faults. `--enforce-eager` avoids the fault for GLM.

## Symptom

At every GLM long-prefill with CUDA graphs (reproduced 3/3 across fresh
restarts, including after a clean reboot), the engine dies mid-prefill:

```
NVRM: Xid (PCI:0000:8B:00): 31, pid=NNNNN, name=python3, channel 0x00000002,
      intr 00000000. MMU Fault: ENGINE GRAPHICS GPC3 GPCCLIENT_T1_0 faulted
      @ 0x7f..bce00000. Fault is of type FAULT_PDE ACCESS_TYPE_VIRT_READ

RuntimeError: RPC call to sample_tokens aborted: a worker process died before
RuntimeError: Engine core initialization failed. See root cause above.
HTTP Error 500: Internal Server Error
```

The fault address always lands near `0x...bce00000` with `FAULT_PDE ... VIRT_READ`
on `ENGINE GRAPHICS GPC3` — a page-directory-entry read, consistent with the
documented GA100-CMP VMM/page-table fragility on large allocation/captured-memory
patterns.

## Scope / attribution evidence (same GPU group 4-7, same driver/day)

| Probe | GLM W4A16 AutoRound | DeepSeek-V4-Flash-0731 |
|---|---|---|
| 32k-context workloads (hours of benchmarks) | OK | (n/a) |
| Raw 42 GB alloc + fill (bfloat16 tensor) on GPU6 alone | OK | (n/a) |
| Long prefill (~300k) with CUDA graphs | **Xid 31** (3x, GPU6 only) | n/a |
| Full 1M-window: 5-depth needle @~960k, ~9x ~1M fills | OK with `--enforce-eager` | OK with CUDA graphs (783k prefill) |
| KV capacity (fp8) on 4 cards | (bf16 KV: 1,190,050 tok) | 7,724,275 tok (7.37x @1M) |
| Related NVRM history on GPU6 | scrubber/SEC2 timeouts + VASPACE/PTE alloc assertion failures (`_scrubWaitAndSave`, `pool_alloc.c:605`, `vaspace_api.c:771`) logged earlier the same week | none observed |

No ECC telemetry is available on these cards (`ecc.errors.* = [N/A]`), so a
memory-ECC explanation can be neither confirmed nor excluded.

## Reproduction recipe (GLM)

1. Serve on GPUs 4-7:
   `vllm serve <model> --pipeline-parallel-size 4 --tensor-parallel-size 1
   --kv-cache-dtype auto --max-model-len 1048576 --enforce-eager` (**must** remove
   `--enforce-eager` to reproduce the fault).
2. Send one ~300k-token prompt (any filler text) via `/v1/completions` or
   `/v1/chat/completions`. Under CUDA graphs the worker on GPU6 faults within the
   first prefill; under eager it completes (measured 262k-token prefill = 71.8 s
   on GPU6 at the time).

Expected: `Xid 31 @ PCI 8B:00` in `dmesg`, worker death, HTTP 500.

## Bottom line for buyers / users

- **Check every physical card** (`nvidia-smi -L` + a long-prefill probe per card,
  per model, in eager AND graph mode) before relying on a given CMP 170HX.
  Cheap pre-checks (small-context benchmarks) do not catch this.
- The fault is reproducible per-card (GPU6) across model forks sharing the same
  vLLM lineage; it is NOT a general "GLM can't do long context on CMP" issue
  (verifiable: same card handles DSV4 at 1M).
- Workaround available: `--enforce-eager` for the affected card; otherwise plan a
  doa replacement for the marginal unit.

## Associated operational notes (this project's findings)

- Driver/driver-model interactions on these cards also include: teardown-triggered
  NVRM `_scrubWaitAndSave` timeouts and VASPACE allocation assertion failures
  (GPU6 most often); containers whose init processes wedge in-kernel after such faults
  (unkillable threads, `docker rm` cannot reap them; cleared only by full reboot).
- `--kv-cache-dtype fp8` is **unsupported for GLM's sparse-MLA path on SM80** in
  the tested fork (all sparse-MLA backends reject fp8 KV; no `--calculate-kv-scales`;
  GLM checkpoint carries no k/v scales). DSV4 fp8 KV works on the same GPUs.
- Native CPU KV offload in this fork: GPU->CPU store measured 6.65 GB/s (GLM) and
  4.6-4.7 GB/s (DSV4); CPU->GPU load never observed (see the DSV4-on-4-7 report).
