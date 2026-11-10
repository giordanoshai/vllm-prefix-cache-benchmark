#!/usr/bin/env bash
set -euo pipefail

export PATH="$HOME/vllm_env/bin:$PATH:/usr/lib/wsl/lib:/usr/local/nvidia/bin"
export HF_ENDPOINT="https://hf-mirror.com"
export VLLM_USE_FLASHINFER_SAMPLER=0

VENV_BIN="$HOME/vllm_env/bin"
BENCH_DIR="$HOME/research-vllm-benchmark"
MODEL="Qwen/Qwen2.5-1.5B-Instruct"

cd "$BENCH_DIR"

# Remove flashinfer-python if present without cubin/nvcc so vLLM uses native PyTorch/Triton ops
~/.local/bin/uv pip uninstall --python "$VENV_BIN/python" flashinfer-python 2>/dev/null || true

echo "=== [1/4] Starting vLLM (FP16 KV Cache + Prefix Caching) on RTX 4090 ==="
pkill -f "vllm serve" || true
sleep 2

"$VENV_BIN/vllm" serve "$MODEL" \
  --host 127.0.0.1 --port 8000 \
  --gpu-memory-utilization 0.80 \
  --max-model-len 4096 \
  --enforce-eager \
  --enable-prefix-caching \
  > "$BENCH_DIR/vllm_4090_fp16.log" 2>&1 &
VLLM_PID=$!

echo "Waiting for vLLM server (PID $VLLM_PID) to become healthy..."
READY=0
for i in $(seq 1 90); do
  if curl -sf http://127.0.0.1:8000/health >/dev/null 2>&1; then
    echo "vLLM FP16 is READY!"
    READY=1
    break
  fi
  if ! kill -0 "$VLLM_PID" 2>/dev/null; then
    echo "vLLM exited unexpectedly! Last 40 log lines:"
    tail -n 40 "$BENCH_DIR/vllm_4090_fp16.log"
    exit 1
  fi
  sleep 2
done

if [ "$READY" -ne 1 ]; then
  echo "Timeout waiting for vLLM FP16! Log tail:"
  tail -n 40 "$BENCH_DIR/vllm_4090_fp16.log"
  exit 1
fi

grep -E "Available KV cache memory|GPU KV cache size" "$BENCH_DIR/vllm_4090_fp16.log" || true

echo "=== [2/4] Running 0% Cache Hit Baseline (unique_prefix) on RTX 4090 ==="
"$VENV_BIN/python" benchmark_runner.py \
  --model "$MODEL" \
  --sweep 1,4,8,16,32,64 --requests 16 --max-tokens 96 \
  --no-shared-prefix \
  --save-json results_4090_unique.json

echo "=== [3/4] Running 100% Cache Hit Wave (shared_prefix) on RTX 4090 ==="
"$VENV_BIN/python" benchmark_runner.py \
  --model "$MODEL" \
  --sweep 1,4,8,16,32,64 --requests 16 --max-tokens 96 \
  --save-json results_4090_shared.json

kill "$VLLM_PID" || true
wait "$VLLM_PID" 2>/dev/null || true
sleep 3

echo "=== [4/4] Starting vLLM with Ada Lovelace FP8 KV Cache (--kv-cache-dtype fp8) ==="
"$VENV_BIN/vllm" serve "$MODEL" \
  --host 127.0.0.1 --port 8000 \
  --gpu-memory-utilization 0.80 \
  --max-model-len 4096 \
  --enforce-eager \
  --kv-cache-dtype fp8 \
  --enable-prefix-caching \
  > "$BENCH_DIR/vllm_4090_fp8.log" 2>&1 &
VLLM_FP8_PID=$!

READY_FP8=0
for i in $(seq 1 90); do
  if curl -sf http://127.0.0.1:8000/health >/dev/null 2>&1; then
    echo "vLLM FP8 KV Cache is READY!"
    READY_FP8=1
    break
  fi
  if ! kill -0 "$VLLM_FP8_PID" 2>/dev/null; then
    echo "vLLM FP8 exited! Last 30 log lines:"
    tail -n 30 "$BENCH_DIR/vllm_4090_fp8.log"
    exit 1
  fi
  sleep 2
done

if [ "$READY_FP8" -eq 1 ]; then
  grep -E "Available KV cache memory|GPU KV cache size" "$BENCH_DIR/vllm_4090_fp8.log" || true
  "$VENV_BIN/python" benchmark_runner.py \
    --model "$MODEL" \
    --sweep 1,4,8,16,32,64 --requests 16 --max-tokens 96 \
    --save-json results_4090_fp8_shared.json
fi

kill "$VLLM_FP8_PID" || true
wait "$VLLM_FP8_PID" 2>/dev/null || true
echo "=== ALL RTX 4090 BENCHMARKS COMPLETED SUCCESSFULLY ==="
