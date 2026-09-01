#!/usr/bin/env bash
# =============================================================================
# sweep_all.sh — run the full DeepSeek-V4-Flash parallelism sweep on 4 free cards.
#
# Cards 0-3 are the production deployment and are NEVER used by this script unless
# you pass PROD_REF=1 (read-only benchmarking against the already-running server).
#
# Usage:
#   sudo env VLLM_API_KEY=<sweep key> ./sweep_all.sh
#   PROD_VLLM_API_KEY=<prod key> ./sweep_all.sh   # also bench prod as pp4 ref
# =============================================================================
set -uo pipefail
cd "$(dirname "$0")"

# Optional: read-only reference measurement of the running production server (cards 0-3).
if [ -n "${PROD_VLLM_API_KEY:-}" ]; then
  echo "=== [reference] production PP4 (cards 0-3) — read-only bench ==="
  mkdir -p results/pp4_prod
  VLLM_API_KEY="${PROD_VLLM_API_KEY}" python3 sweep_bench.py --base-url http://127.0.0.1:8098 \
    > results/pp4_prod/bench.json 2> results/pp4_prod/bench.err && echo "prod ref done"
fi

for cfg in tp4 tp4_ep tp4_ep_megamoe tp2_pp2 tp2_pp2_ep pp4; do
  sudo env VLLM_API_KEY="${VLLM_API_KEY}" MAX_MODEL_LEN="${MAX_MODEL_LEN:-1048576}" \
    ./run_dsv4_sweep.sh "${cfg}" "${SWEEP_GPUS:-4,5,6,7}"
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
