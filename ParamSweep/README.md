# DeepSeek-V4-Flash parameter sweep on CMP 170HX

Compares DeepSeek-V4-Flash-0731 parallelism configurations on **4 free cards**
(GPU IDs 4-7) using the same `dsv4-a100:devel` vLLM fork that runs the
production deployment on cards 0-3.

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

A read-only reference pass (`results/pp4_prod/`) benchmarks the production PP4
server on cards 0-3 with the *same* client so all numbers are directly comparable.

## How to run

```bash
# reference (read-only) on the production server:
VLLM_API_KEY=<prod key> python3 sweep_bench.py --base-url http://127.0.0.1:8098

# full sweep on cards 4-7:
sudo env VLLM_API_KEY=<key> ./sweep_all.sh
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

*Sweep run: 2026-08-31, cards 4-7, driver 610.43.02, AlmaLinux 10.2.*
