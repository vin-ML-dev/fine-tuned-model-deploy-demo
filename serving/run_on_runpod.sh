#!/usr/bin/env bash
# =============================================================
# Phase 4 on RunPod: run the serving stack WITHOUT Docker.
#
# RunPod pods are containers themselves - you cannot run a Docker
# daemon inside them. So on RunPod we run both processes directly:
#   process 1: vLLM OpenAI server  (:8001)  - the engine
#   process 2: FastAPI wrapper     (:8000)  - the API layer
#
# Usage (on a RunPod GPU pod, from the repo root):
#   export HF_TOKEN=hf_xxx          # if HF repo is private
#   bash serving/run_on_runpod.sh
#
# Then expose port 8000 in the RunPod console (HTTP port) and test:
#   curl -s localhost:8000/readyz
#   python serving/test_api.py
#
# Logs: /workspace/logs/engine.log and /workspace/logs/api.log
# Stop: bash serving/run_on_runpod.sh stop
# =============================================================
set -e

MODEL="vinmlops/technova-1.5b-instruct"
LOG_DIR="/workspace/logs"
mkdir -p "$LOG_DIR"

if [ "$1" == "stop" ]; then
  pkill -f "vllm serve" || true
  pkill -f "uvicorn serving.app.main" || true
  echo "Stopped engine and API."
  exit 0
fi

# ---- 1. Install dependencies ----
echo ">> Installing vLLM + API deps (first run takes a few minutes)..."
pip install -q vllm
pip install -q -r serving/requirements-api.txt

# HF auth for private repos
if [ -n "$HF_TOKEN" ]; then
  export HUGGING_FACE_HUB_TOKEN="$HF_TOKEN"
fi

# ---- 2. Start the vLLM engine (background) ----
echo ">> Starting vLLM engine on :8001 (model load takes 1-3 min)..."
nohup vllm serve "$MODEL" \
  --max-model-len 2048 \
  --gpu-memory-utilization 0.90 \
  --port 8001 \
  > "$LOG_DIR/engine.log" 2>&1 &

# ---- 3. Wait for the engine to be ready ----
echo ">> Waiting for engine..."
for i in $(seq 1 60); do
  if curl -sf http://localhost:8001/v1/models > /dev/null 2>&1; then
    echo ">> Engine is up."
    break
  fi
  sleep 5
  if [ "$i" == "60" ]; then
    echo "ERROR: engine did not start. Check $LOG_DIR/engine.log"
    exit 1
  fi
done

# ---- 4. Start the FastAPI wrapper (background) ----
echo ">> Starting FastAPI wrapper on :8000..."
export ENGINE_BASE_URL="http://localhost:8001/v1"
export MODEL_NAME="$MODEL"
nohup uvicorn serving.app.main:app --host 0.0.0.0 --port 8000 \
  > "$LOG_DIR/api.log" 2>&1 &

sleep 3
echo ""
echo ">> Stack is running:"
echo "   engine: http://localhost:8001  (log: $LOG_DIR/engine.log)"
echo "   api:    http://localhost:8000  (log: $LOG_DIR/api.log)"
echo ""
echo ">> Smoke test:"
curl -s localhost:8000/readyz && echo ""
echo ""
echo ">> Try: python serving/test_api.py"
echo ">> To reach it from your laptop: add port 8000 as an HTTP port in the"
echo "   RunPod console, then use the proxy URL https://<POD_ID>-8000.proxy.runpod.net"
