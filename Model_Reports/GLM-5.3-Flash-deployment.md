<!--
GLM-5.3-Flash deployment notes — vLLM fork + W4A16-AutoRound on 4x CMP 170HX.
De-localized: no host paths, GPU IDs, container names, or credentials.
-->

# GLM-5.3-Flash deployment (vLLM fork, W4A16-AutoRound)

This page documents how GLM-5.3-Flash is deployed on 4x CMP 170HX with the
community vLLM fork (SM80), how that image is built, and how the W4A16-AutoRound
checkpoint is served. Benchmarks and the feasibility of other engines/quantizations are
in the accompanying report:

[`benchmark_GLM-5.3-Flash-AutoRound.md`](benchmark_GLM-5.3-Flash-AutoRound.md)

## Stack

| Component | Choice |
|---|---|
| Model | **GLM-5.3-Flash**, `Glm5NextForConditionalGeneration` (45 layers: 3 dense + 42 MoE, hidden 4096, 64 heads, ~320B total / ~18B active, native VLM) |
| Weights | **W4A16-AutoRound** INT4 checkpoint (162 GiB, 36 shards; vLLM `quantization=inc`, auto-detected from `quantization_config`) |
| Engine | Community vLLM fork `0.8.1+glm53a800` ([`Mrzhiyao/glm53-a800-vllm`](https://github.com/Mrzhiyao/glm53-a800-vllm)), built as a local docker image (SM80-only CUDA extension + Triton overrides for the GLM linear-attention/KPool kernels) |
| GPU group | 4x NVIDIA CMP 170HX (GA100 sm_80, VRAM-unlocked 65536 MiB, PCIe Gen2 x16), one NUMA/PCIe group |
| Parallelism | **PP=4 + TP=1** (`VLLM_PP_LAYER_PARTITION=12,11,11,11`) |
| Context | 32,768 tokens (fork supports the full 1,048,576 the checkpoint allows; 32k chosen to fit 4x 64 GiB with a KV margin of ~12 GiB) |

## Why PP=4 and never TP

Same arithmetic barrier as the DeepSeek-V4-Flash note in the top-level
deployment doc: 64 attention heads do not divide across tensor-parallel ranks on 4
GPUs, and this card group has no NVLink, so PP is the only practical sharding.

## How the engine image is built

The fork is assembled from two upstream sources plus an SM80 build:

1. **Baseline vLLM backport** — `wtdcode/vllm-backport`
   (pinned commit `2674c8bb6d8799b32158c94bee33356d84772a2a`).
2. **GLM integration** — `ZJY0516/vllm:glm-release`
   (pinned commit `6f0369074d9f755917ee2d29c15809ea73bcbfba`), which adds the
   `glm5next` model family, the Triton sparse/linear-attention MLA overrides, and
   bounded KPool tail-cache kernels that are safe under CUDA Graph capture.
3. **SM80 base container** — `lazymio/vllm-backport:latest-sm80` with the
   prebuilt Rust frontend (`vllm-rs`) reused as-is; only the CUDA/C++ extension
   is recompiled, with `TORCH_CUDA_ARCH_LIST=8.0` and version string
   `0.8.1+glm53a800`.

Build flow (mirrors the upstream [`glm53-a800-vllm`](https://github.com/Mrzhiyao/glm53-a800-vllm) repo's `scripts/`):

```bash
# 1) fetch the baseline + integrate
git clone --filter=blob:none --no-checkout <vllm-backport> build/vllm-src
git -C build/vllm-src fetch --depth 1 origin 2674c8bb6d8799b32158c94bee33356d84772a2a
git -C build/vllm-src checkout --detach 2674c8bb6d8799b32158c94bee33356d84772a2a
git -C build/vllm-src apply   patches/glm53-sm80.patch
cp -a overrides/. build/vllm-src/          # final Python/Triton overrides win

# 2) build the image
DOCKER_BUILDKIT=1 docker build --build-arg BASE_IMAGE=lazymio/vllm-backport:latest-sm80 \
  -f docker/Dockerfile.sm80 -t glm53-a800:sm80-v9-cudagraph build/vllm-src
```

The `Dockerfile.sm80` builds the wheel in a builder stage (`pip wheel --no-deps`,
BuildKit cache mounts for pip/`.deps`), then re-uses the base image's `vllm-rs`
Rust frontend and installs only the rebuilt wheel into the final image (~17 GB).

> **China-network build customization (optional).** The build steps were adapted for
> networks where GitHub/PyPI/Ubuntu are slow or blocked: apt/PyPI/uv indexes
> point at the Tsinghua TUNA mirrors, the wheel step uses the SJTU PyPI mirror, and
> in-image GitHub clones are routed through a GitHub acceleration proxy
> (`git config url."https://gh-proxy.org/https://github.com/".insteadOf ...`).
> These are build-time conveniences only; the produced image runs anywhere the driver
> and model are present.

## Serving (on the 4x CMP 170HX group)

```bash
docker run -d --name <glm-container> \
  --restart unless-stopped --network host --ipc host \
  --gpus "\"device=<4 GPU IDs>\"" \
  -e HF_HUB_OFFLINE=1 -e TRANSFORMERS_OFFLINE=1 \
  -e VLLM_USE_DEEP_GEMM=0 \
  -e VLLM_PP_LAYER_PARTITION=12,11,11,11 \
  -e NCCL_ALGO=Ring -e NCCL_PROTO=Simple \
  -v <model_dir>:<model_dir>:ro \
  <image> <model_dir> \
  --served-model-name GLM-5.3-Flash glm-5.3-flash \
  --tensor-parallel-size 1 --pipeline-parallel-size 4 --trust-remote-code \
  --max-model-len 32768 --gpu-memory-utilization 0.90 \
  --max-num-seqs 8 --max-num-batched-tokens 8192 \
  --disable-custom-all-reduce \
  --limit-mm-per-prompt '{"image":1,"video":0}' \
  --enable-auto-tool-choice --tool-call-parser glm47 \
  --reasoning-parser glm45 \
  --host 0.0.0.0 --port <port>
```

Key settings and why:

| Setting | Value | Why |
|---|---|---|
| `VLLM_PP_LAYER_PARTITION` | `12,11,11,11` | 45 layers across 4 PP ranks |
| `--disable-custom-all-reduce` | on | default NCCL all-reduce on this rig's PCIe/NUMA topology |
| `NCCL_ALGO/PROTO` | `Ring/Simple` | most reliable on the Gen2 PCIe fabric |
| `--gpu-memory-utilization` | 0.90 | ~44.5 GiB weights + ~5.4 GiB peak activation per rank; leaves ~12.1 GiB KV |
| `--max-num-seqs` / `--max-num-batched-tokens` | 8 / 8192 | bounded batch; chunked prefill enabled by default |
| `--reasoning-parser glm45` / `--tool-call-parser glm47` | on | GLM-5.3 thinking + tool-calling schema |

The ~162 GiB checkpoint loads in a few minutes; CUDA graph capture and TileLang
kernel JIT add a warmup window before the health endpoint answers.

## Results

See [`benchmark_GLM-5.3-Flash-AutoRound.md`](benchmark_GLM-5.3-Flash-AutoRound.md):
39.5 tok/s single-stream decode (TPOT ~23 ms), 143.6 tok/s best aggregate
(short context, c=16), 84.6 tok/s at 16k context, c=2 interactive sweet spot.
Cross-checked with the fork's own end-to-end stress tests (all pass).

Other engines/quantizations on sm_80 (SGLang TP4/EP4 + native FP8, NVFP4,
FP8-native) are not runnable on this hardware; details in the report's appendix.
