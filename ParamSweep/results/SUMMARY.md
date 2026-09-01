# DeepSeek-V4-Flash parallel-config sweep — summary

| config | status | decode1 (single, tok/s) | decode3 (3x400, agg tok/s) | prefill 4096 (tok/s) | conc16 (agg tok/s) |
|---|---:|---:|---:|---:|---:|
| pp4 | ok | 142.3 | 133.0 | 4857.4 | 458.8 |
| pp4_prod | ok | 175.9 | 311.2 | 6385.2 | 553.2 |
| tp2_pp2_nospec | ok | 51.9 | 118.2 | 6135.9 | 185.7 |
| tp4 | ok | 171.0 | 145.8 | 5830.3 | 495.8 |
| tp4_ep | ok | 169.2 | 136.7 | 5920.3 | 473.4 |

Method: `sweep_bench.py` (see ParamSweep README). pp4_prod = read-only
benchmark of the production PP4 server on cards 0-3.
