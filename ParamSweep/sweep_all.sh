#!/usr/bin/env bash
# =============================================================================
# sweep_all.sh — run the full DeepSeek-V4-Flash parallelism sweep on N free cards.
#
# The sweep GPUs come from SWEEP_GPUS (or pass a GPU list); keep them off any
# busy deployment. Optionally also do a read-only reference measurement of an
# already-running server via PROD_VLLM_API_KEY + PROD_PORT.
#
# Usage:
#   sudo env VLLM_API_KEY=<sweep key> SWEEP_GPUS=<N GPU IDs> ./sweep_all.sh
#   PROD_VLLM_API_KEY=<key> PROD_PORT=<port> ./sweep_all.sh  # + read-only ref
# =============================================================================
set -uo pipefail
cd "$(dirname "$0")"

# Optional: read-only reference measurement of an existing server on a separate GPU group.
if [ -n "${PROD_VLLM_API_KEY:-}" ]; then
  PROD_PORT="${PROD_PORT:-8000}"
  echo "=== [reference] existing PP4 server — read-only bench ==="
  mkdir -p results/pp4_prod
  VLLM_API_KEY="${PROD_VLLM_API_KEY}" python3 sweep_bench.py --base-url "http://127.0.0.1:${PROD_PORT}" \
    > results/pp4_prod/bench.json 2> results/pp4_prod/bench.err && echo "prod ref done"
fi

: "${SWEEP_GPUS:?set SWEEP_GPUS to the GPU IDs for the sweep}"

for cfg in tp4 tp4_ep tp4_ep_megamoe tp2_pp2 tp2_pp2_ep pp4; do
  sudo env VLLM_API_KEY="${VLLM_API_KEY}" MAX_MODEL_LEN="${MAX_MODEL_LEN:-1048576}" \
    ./run_dsv4_sweep.sh "${cfg}" "${SWEEP_GPUS}"
done

echo "=== summary ==="
python3 - <<'PYEOF'
import json, glob, os
for f in sorted(glob.glob("results/*/bench.json")):
    cfg = f.split("/")[-2]
    try:
        r = json.load(open(f))
        d3 = r.get("decode3", {}); pf = r.get("prefill", {}); c16 = r.get("conc16", {})
        e = os.path.join(os.path.dirname(f), "engine.log")
        eng = open(e).read().splitlines()[0] if os.path.exists(e) and open(e).read() else ""
        print(f"{cfg:12s} decode3={d3.get('aggregate_tok_s','x')} tok/s  "
              f"prefill={pf.get('prefill_tok_s','x')} tok/s  "
              f"conc16={c16.get('aggregate_tok_s','x')} tok/s  "
              f"single={r.get('decode1',{}).get('decode_tok_s','x')} tok/s")
    except Exception as ex:
        print(f"{cfg:12s} ERROR {ex}")
PYEOF
