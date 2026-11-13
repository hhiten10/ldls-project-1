#!/usr/bin/env bash
# Build the ggml driver against the pinned llama.cpp's ggml (CPU backend), same flags as the kernel harness.
set -euo pipefail
source "$(dirname "$0")/../../scripts/env.sh"
cd "$(dirname "$0")"
g++ -O3 -march=native -fopenmp -std=c++17 -I"$LLAMA/ggml/include" \
    ggml_driver.cpp ../kernel/m3_kernels.cpp -L"$BIN" -lggml -lggml-base -lggml-cpu \
    -Wl,-rpath,"$BIN" -o ggml_driver
echo built ggml_driver
