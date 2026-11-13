#!/usr/bin/env bash
# Build llama.cpp CPU-only (no CUDA/Vulkan/BLAS), Release, native AVX2/FMA.
# Debug symbols kept (RelWithDebInfo-like flags) so perf can attribute time to ggml ops.
set -euo pipefail
source "$(dirname "$0")/env.sh"
mkdir -p "$WORK"
if [ ! -d "$LLAMA/.git" ]; then
  git clone https://github.com/ggml-org/llama.cpp "$LLAMA"
fi
cd "$LLAMA"
if [ -s "$LLAMA_COMMIT_FILE" ]; then
  git fetch -q origin && git checkout -q "$(cat "$LLAMA_COMMIT_FILE")"
else
  git rev-parse HEAD > "$LLAMA_COMMIT_FILE"
fi
echo "llama.cpp @ $(git rev-parse --short HEAD)"
cmake -S . -B "$BUILD" -G Ninja \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_C_FLAGS="-g -fno-omit-frame-pointer" \
  -DCMAKE_CXX_FLAGS="-g -fno-omit-frame-pointer" \
  -DGGML_CUDA=OFF -DGGML_VULKAN=OFF -DGGML_METAL=OFF -DGGML_BLAS=OFF \
  -DGGML_NATIVE=ON -DGGML_OPENMP=ON \
  -DLLAMA_CURL=OFF -DLLAMA_BUILD_TESTS=ON
cmake --build "$BUILD" -j"$(nproc)" --target \
  llama-cli llama-bench llama-perplexity llama-quantize llama-completion test-backend-ops 2>&1 | tail -n 5 || \
cmake --build "$BUILD" -j"$(nproc)" 2>&1 | tail -n 5
ls "$BIN"
"$BIN/llama-bench" --help >/dev/null && echo BUILD_OK
