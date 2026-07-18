#!/usr/bin/env bash
# =============================================================
# Phase 3 - Step 11b/12: Convert merged HF model -> GGUF + quantize
# Hardened version: system update, latest deps, sanity checks.
#
# Usage:  bash scripts/convert_to_gguf.sh
# Outputs:
#   $OUT_DIR/technova-1.5b-bf16.gguf     (~3 GB, full precision)
#   $OUT_DIR/technova-1.5b-Q4_K_M.gguf   (~1 GB, 4-bit quantized)
# =============================================================
set -e

# ---- Paths (edit to match your environment; RunPod layout below) ----
MERGED_MODEL="${MERGED_MODEL:-/workspace/outputs/merged-technova-1.5b}"
OUT_DIR="${OUT_DIR:-/workspace/outputs/gguf}"
LLAMA_DIR="${LLAMA_DIR:-/workspace/llama.cpp}"

mkdir -p "$OUT_DIR"

# ---- 0. System update + build tools ----
echo ">> Updating system packages..."
apt-get update -qq
apt-get install -y -qq --no-install-recommends \
  git cmake build-essential curl ca-certificates

# ---- 1. Get / update llama.cpp ----
if [ ! -d "$LLAMA_DIR" ]; then
  echo ">> Cloning llama.cpp (latest)..."
  git clone --depth 1 https://github.com/ggml-org/llama.cpp "$LLAMA_DIR"
else
  echo ">> Updating existing llama.cpp to latest..."
  git -C "$LLAMA_DIR" pull --ff-only || echo "   (pull failed on shallow clone - delete $LLAMA_DIR to re-clone fresh)"
fi

# ---- 2. Latest Python deps for the converter ----
echo ">> Installing/upgrading converter dependencies..."
pip install -q --upgrade pip
pip install -q --upgrade -r "$LLAMA_DIR/requirements/requirements-convert_hf_to_gguf.txt"
pip install -q --upgrade transformers sentencepiece gguf

# ---- Sanity checks before converting ----
test -f "$MERGED_MODEL/tokenizer.json"    || { echo "ERROR: $MERGED_MODEL/tokenizer.json missing"; exit 1; }
test -f "$MERGED_MODEL/model.safetensors" || ls "$MERGED_MODEL"/model-*.safetensors >/dev/null 2>&1 \
  || { echo "ERROR: model weights missing in $MERGED_MODEL"; exit 1; }

# ---- 3. Convert HF -> GGUF (bf16, matches training/merge dtype) ----
echo ">> Converting to GGUF bf16..."
python "$LLAMA_DIR/convert_hf_to_gguf.py" "$MERGED_MODEL" \
  --outfile "$OUT_DIR/technova-1.5b-bf16.gguf" \
  --outtype bf16

# ---- 4. Build the quantizer (CPU build is enough) ----
if [ ! -f "$LLAMA_DIR/build/bin/llama-quantize" ]; then
  echo ">> Building llama.cpp tools..."
  cmake -S "$LLAMA_DIR" -B "$LLAMA_DIR/build" -DGGML_CUDA=OFF > /dev/null
  cmake --build "$LLAMA_DIR/build" --target llama-quantize llama-cli -j "$(nproc)" > /dev/null
fi

# ---- 5. Quantize bf16 -> Q4_K_M (best quality/size tradeoff) ----
echo ">> Quantizing to Q4_K_M..."
"$LLAMA_DIR/build/bin/llama-quantize" \
  "$OUT_DIR/technova-1.5b-bf16.gguf" \
  "$OUT_DIR/technova-1.5b-Q4_K_M.gguf" Q4_K_M

echo ""
echo ">> Done. Files:"
ls -lh "$OUT_DIR"
echo ""
echo ">> Quick smoke test (CPU):"
echo "$LLAMA_DIR/build/bin/llama-cli -m $OUT_DIR/technova-1.5b-Q4_K_M.gguf -p \"What is TechNova's refund policy?\" -n 100"
