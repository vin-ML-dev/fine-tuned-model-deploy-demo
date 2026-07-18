#!/usr/bin/env bash
# =============================================================
# Container entrypoint: start vLLM engine, wait for ready,
# then start the FastAPI wrapper in the foreground.
#
# Configurable via environment variables (set them in the
# RunPod template or docker run -e):
#   MODEL_NAME        default vinmlops/technova-1.5b-instruct
#   MAX_MODEL_LEN     default 2048
#   GPU_MEM_UTIL      default 0.90
#   HF_TOKEN          required if the HF repo is private
#   RATE_LIMIT_RPM    default 60
# =============================================================
set -e

MODEL_NAME="${MODEL_NAME:-vinmlops/technova-1.5b-instruct}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-2048}"
GPU_MEM_UTIL="${GPU_MEM_UTIL:-0.90}"

if [ -n "$HF_TOKEN" ]; then
  export HUGGING_FACE_HUB_TOKEN="$HF_TOKEN"
fi

echo ">> Starting vLLM engine (:8001) for model: $MODEL_NAME"
vllm serve "$MODEL_NAME" \
  --max-model-len "$MAX_MODEL_LEN" \
  --gpu-memory-utilization "$GPU_MEM_UTIL" \
  --port 8001 &
ENGINE_PID=$!

echo ">> Waiting for engine to load the model..."
for i in $(seq 1 120); do
  if curl -sf http://localhost:8001/v1/models > /dev/null 2>&1; then
    echo ">> Engine is ready."
    break
  fi
  # fail fast if the engine process died (bad model name, OOM, etc.)
  if ! kill -0 "$ENGINE_PID" 2>/dev/null; then
    echo "ERROR: vLLM engine process exited during startup."
    exit 1
  fi
  sleep 5
  if [ "$i" == "120" ]; then
    echo "ERROR: engine did not become ready in 10 minutes."
    exit 1
  fi
done

echo ">> Starting FastAPI wrapper (:8000)"
export ENGINE_BASE_URL="http://localhost:8001/v1"
export MODEL_NAME
# exec = uvicorn becomes PID-1-adjacent foreground process;
# if it dies, the container dies -> RunPod/K8s can restart it.
exec uvicorn app.main:app --host 0.0.0.0 --port 8000
