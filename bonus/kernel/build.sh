#!/usr/bin/env bash
# Build the bonus kernel harness. -march=native on the Ryzen 4800H = AVX2 + FMA (no AVX-512).
set -euo pipefail
cd "$(dirname "$0")"
g++ -O3 -march=native -mavx2 -mfma -fopenmp -g -fno-omit-frame-pointer -std=c++17 -Wall -Wno-unused-result \
    main.cpp m3_kernels.cpp -o m3bench
echo built m3bench
