#!/usr/bin/env bash
# Second build of the SAME commit with the opt-in per-op profiler compiled in (-DGGML_OP_PROFILE).
# Benchmarks never use this build; it only produces operator breakdowns (cross-checked with perf).
set -euo pipefail
source "$(dirname "$0")/env.sh"
"$VENV/bin/python" "$REPO/scripts/tools/make_op_profile_patch.py" "$LLAMA"
( cd "$LLAMA" && git diff -- ggml/src/ggml-cpu/ggml-cpu.c > "$REPO/docs/ggml-op-profile.patch" )
cmake -S "$LLAMA" -B "$LLAMA/build-prof" -G Ninja -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_C_FLAGS="-g -fno-omit-frame-pointer -DGGML_OP_PROFILE" \
  -DCMAKE_CXX_FLAGS="-g -fno-omit-frame-pointer -DGGML_OP_PROFILE" \
  -DGGML_CUDA=OFF -DGGML_VULKAN=OFF -DGGML_BLAS=OFF -DGGML_NATIVE=ON -DGGML_OPENMP=ON -DLLAMA_CURL=OFF \
  -DLLAMA_BUILD_TESTS=OFF > /dev/null
cmake --build "$LLAMA/build-prof" -j"$(nproc)" --target llama-bench 2>&1 | tail -n 2
echo PROF_BUILD_OK
