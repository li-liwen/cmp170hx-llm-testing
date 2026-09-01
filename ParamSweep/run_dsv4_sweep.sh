#!/usr/bin/env bash
# =============================================================================
# run_dsv4_sweep.sh — DeepSeek-V4-Flash parallelism sweep on CMP 170HX.
#
# Launches ONE config of the DeepSeek-V4-Flash-0731 checkpoint on a specified
# GPU set (default: $SWEEP_GPUS; keep it off any busy deployment's GPUs),
# waits for the API to come up, runs sweep_bench.py, and tears the server down.
#
# Writes raw JSON + a short markdown summary per config.
#
# Usage:
#   sudo env VLLM_API_KEY=<key> SWEEP_GPUS=<N GPU IDs> ./run_dsv4_sweep.sh <config> [gpus]
#
# Configs: pp4 | tp4 | tp4_ep | tp4_ep_megamoe | tp2_pp2 | tp2_pp2_ep | tp2_pp2_nospec
# Other env: MODEL_DIR (weights path, default /path/to/DeepSeek-V4-Flash-0731),
#            IMG (vLLM image, default vllm-cmp170hx:latest), SWEEP_PORT.
# =============================================================================
set -uo pipefail

CONFIG="${1:?usage: run_dsv4_sweep.sh <config> [gpus]}"
GPUS="${2:-${SWEEP_GPUS:?set SWEEP_GPUS or pass GPU IDs as arg 2}}"
PORT="${SWEEP_PORT:-8000}"
MODEL_DIR="${MODEL_DIR:-/path/to/DeepSeek-V4-Flash-0731}"
IMG="${IMG:-vllm-cmp170hx:latest}"
CONTAINER="vllm-sweep"

KEY="${VLLM_API_KEY:?VLLM_API_KEY must be set}"
BASE="http://127.0.0.1:${PORT}"
RESULTS="results/${CONFIG}"

echo "=== config=${CONFIG} gpus=${GPUS} port=${PORT} ==="
mkdir -p "${RESULTS}"

EXTRA=()
NOSPEC=""
case "${CONFIG}" in
  pp4)
    EXTRA=(--pipeline-parallel-size=4)
    ROW_CHUNK="64"
    ;;
  tp4)
    EXTRA=(--tensor-parallel-size=4)
    ROW_CHUNK="64"
    ;;
  tp4_ep)
    EXTRA=(--tensor-parallel-size=4 --enable-expert-parallel)
    ROW_CHUNK="64"
    ;;
  tp4_ep_megamoe)
    EXTRA=(--tensor-parallel-size=4 --enable-expert-parallel --moe-backend deep_gemm_mega_moe)
    ROW_CHUNK="64"
    ;;
  tp2_pp2)
    EXTRA=(--tensor-parallel-size=2 --pipeline-parallel-size=2)
    ROW_CHUNK="64"
    ;;
  tp2_pp2_nospec)
    EXTRA=(--tensor-parallel-size=2 --pipeline-parallel-size=2)
    ROW_CHUNK="64"
    NOSPEC="1"
    ;;
  tp2_pp2_ep)
    EXTRA=(--tensor-parallel-size=2 --pipeline-parallel-size=2 --enable-expert-parallel)
    ROW_CHUNK="64"
    ;;
  *)
    echo "unknown config ${CONFIG}"; exit 2 ;;
esac

if [ -n "${NOSPEC}" ]; then
  SPEC_ARGS=()
else
  SPEC_ARGS=(--speculative-config='{"method":"dspark","num_speculative_tokens":5}')
fi

# Tear down any stale sweep container first.
sudo docker rm -f "${CONTAINER}" >/dev/null 2>&1 || true

echo "--- launching server ---"
sudo docker run -d --name "${CONTAINER}" \
  --runtime nvidia \
  --shm-size 16g \
  -p "${PORT}:8000" \
  -e NVIDIA_VISIBLE_DEVICES="${GPUS}" \
  -e HF_HUB_OFFLINE=1 \
  -e VLLM_WORKER_MULTIPROC_METHOD=spawn \
  -e DSV4_LOGITS_ROW_CHUNK="${ROW_CHUNK}" \
  -e VLLM_API_KEY="${KEY}" \
  -v "${MODEL_DIR}:/model" \
  -v vllm_compile_cache:/root/.cache/vllm \
  "${IMG}" \
  vllm serve /model \
  --served-model-name=deepseek-v4-flash \
  "${EXTRA[@]}" \
  --kv-cache-dtype=fp8 \
  --block-size=256 \
  --max-model-len="${MAX_MODEL_LEN:-1048576}" \
  --max-num-batched-tokens=2048 \
  --trust-remote-code \
  --gpu-memory-utilization="${GPU_MEM_UTIL:-0.9}" \
  --max-num-seqs="${MAX_NUM_SEQS:-8}" \
  --no-enable-flashinfer-autotune \
  --tokenizer-mode=deepseek_v4 \
  --enable-auto-tool-choice \
  --tool-call-parser=deepseek_v4 \
  --reasoning-parser=deepseek_v4 \
  "${SPEC_ARGS[@]}" > /dev/null
# Wait for health (with auth). Cap at ~25 minutes (load + compile + cudagraph).
echo "--- waiting for health ---"
ok=0
for i in $(seq 1 100); do
  code=$(curl -s -o /dev/null -w "%{http_code}" -m 5 \
        -H "Authorization: Bearer ${KEY}" "http://127.0.0.1:${PORT}/v1/models" 2>/dev/null || true)
  if [ "${code}" = "200" ]; then ok=1; echo "server up after ~$((i*15))s"; break; fi
  # If the container died, stop waiting and report the error.
  if ! sudo docker ps --format '{{.Names}}' | grep -q "^${CONTAINER}$"; then
    echo "SERVER EXITED — dumping last logs:"; sudo docker logs "${CONTAINER}" 2>&1 | tail -25
    sudo docker logs "${CONTAINER}" 2>&1 | tail -40 > "${RESULTS}/server_error.log"
    break
  fi
  sleep 15
done
if [ "${ok}" != "1" ]; then
  echo "=== FAILED to start config ${CONFIG} ==="
  sudo docker logs "${CONTAINER}" 2>&1 > "${RESULTS}/server_error.log"
  sudo docker rm -f "${CONTAINER}" >/dev/null 2>&1 || true
  exit 1
fi

sleep 5
echo "--- running sweep_bench ---"
env VLLM_API_KEY="${KEY}" python3 sweep_bench.py --base-url "${BASE}" \
  > "${RESULTS}/bench.json" 2> "${RESULTS}/bench.err" || echo "bench failed"
python3 - <<PYEOF
import json,sys
try:
    r=json.load(open("${RESULTS}/bench.json"))
    print(json.dumps(r.get("decode3",{}),indent=2))
    print(json.dumps(r.get("prefill",{}),indent=2))
    print(json.dumps(r.get("conc16",{}),indent=2))
except Exception as e:
    print("json parse failed:",e)
    sys.exit(0)
PYEOF

# Capture engine-side KV pool / startup facts
sudo docker logs "${CONTAINER}" 2>&1 | grep -iE "GPU KV cache size|Maximum concurrency|Model loading took|init engine \(profile" \
  | sed 's/^/    /' | head -8 > "${RESULTS}/engine.log" || true

echo "--- tearing down ---"
sudo docker rm -f "${CONTAINER}" >/dev/null 2>&1 || true
echo "=== done: ${CONFIG} ==="
