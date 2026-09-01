# DeepSeek-V4-Flash parameter sweep on CMP 170HX

Compares DeepSeek-V4-Flash-0731 parallelism configurations on a free GPU group,
optionally with a read-only reference run against an existing deployment on a separate
GPU group.

## Configurations

| config | engine args | notes |
|---|---|---|
| `pp4` | `--pipeline-parallel-size=4` | mirrors production |
| `tp4` | `--tensor-parallel-size=4` | TP-only on 4 cards |
| `tp4_ep` | `--tensor-parallel-size=4 --enable-expert-parallel` | TP + EP flag |
| `tp4_ep_megamoe` | `--tensor-parallel-size=4 --enable-expert-parallel --moe-backend deep_gemm_mega_moe` | EP requires MegaMoE backend (Hopper-oriented; expected to fail on sm_80 — recorded if so) |
| `tp2_pp2` | `--tensor-parallel-size=2 --pipeline-parallel-size=2` | hybrid |
| `tp2_pp2_ep` | `--tensor-parallel-size=2 --pipeline-parallel-size=2 --enable-expert-parallel` | hybrid + EP flag |

Common: `--max-model-len 1048576`, `DSV4_LOGITS_ROW_CHUNK=64`, fp8 KV cache,
DSpark spec decode n=5, `--gpu-memory-utilization 0.9`.

A read-only reference pass (`results/pp4_prod/`) benchmarks an existing PP4
server on a different GPU group with the *same* client so all numbers are directly
comparable.

## How to run

```bash
# reference (read-only) against an existing server:
VLLM_API_KEY=<key> python3 sweep_bench.py --base-url http://127.0.0.1:PORT

# full sweep on your free GPUs:
sudo env VLLM_API_KEY=<key> SWEEP_GPUS=<N GPU IDs> ./sweep_all.sh
# results land in results/<config>/
```

Each config: launch container -> wait for `/v1/models` -> run `sweep_bench.py`
(decode1 / decode3 / prefill / conc16) -> capture engine KV log -> teardown.

## Files

- `sweep_bench.py`   — dependency-free OpenAI-compatible bench client
- `run_dsv4_sweep.sh`— run one config on a GPU set
- `sweep_all.sh`      — run the whole campaign + summary table
- `results/`          — raw JSON + engine logs per config

## Results

See `results/SUMMARY.md` (assembled by `make_summary.py`) and the per-config
JSON files.

*Sweep run: 2026-08-31, driver 610.43.02, AlmaLinux 10.2.*
