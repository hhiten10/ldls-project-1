#!/usr/bin/env bash
# PARAM Rudra, LOGIN NODE ONLY (manual: login nodes are for compiling, transferring data, submitting jobs).
# Everything lives under /home/bapic_iiitd/2_group/aditya/p1_rudra - nothing is written elsewhere
# (pip / HF caches are redirected inside this folder too).
set -euo pipefail
D=/home/bapic_iiitd/2_group/aditya/p1_rudra
COMMIT="$1"                                   # same llama.cpp commit as the laptop runs
mkdir -p "$D" && cd "$D"
export PIP_CACHE_DIR="$D/.pip-cache" XDG_CACHE_HOME="$D/.cache" HF_HOME="$D/.cache/hf"
log() { echo "[$(date -Iseconds)] $*" >> /home/bapic_iiitd/2_group/aditya/claude_commands.log; }

# build tools in a small PERSONAL venv (not the shared bhaskera-venv)
if [ ! -x venv/bin/cmake ]; then
  log "WRITE: python3 -m venv $D/venv; pip install cmake ninja huggingface_hub (personal venv)"
  python3 -m venv venv && venv/bin/pip install -q --upgrade pip && venv/bin/pip install -q cmake ninja huggingface_hub
fi
export PATH="$D/venv/bin:$PATH"
module load compiler/gcc/12.3 2>/dev/null || true
gcc --version | head -1

if [ ! -d llama.cpp/.git ]; then
  log "WRITE: shallow clone llama.cpp @ $COMMIT into $D/llama.cpp"
  git init -q llama.cpp && git -C llama.cpp remote add origin https://github.com/ggml-org/llama.cpp
  git -C llama.cpp fetch -q --depth 1 origin "$COMMIT" && git -C llama.cpp checkout -q FETCH_HEAD
fi
log "CMD: build llama.cpp CPU-only (GGML_NATIVE on login node = same Xeon 6240R as compute nodes), -j8"
rm -rf llama.cpp/build
# explicit CC/CXX: otherwise CMake picks /usr/bin/c++ (system gcc 8.5) and the link fails on std::filesystem
CC=$(command -v gcc) CXX=$(command -v g++) \
cmake -S llama.cpp -B llama.cpp/build -G Ninja -DCMAKE_BUILD_TYPE=Release -DGGML_NATIVE=ON -DGGML_OPENMP=ON \
  -DGGML_CUDA=OFF -DLLAMA_CURL=OFF -DLLAMA_BUILD_TESTS=OFF > /dev/null
cmake --build llama.cpp/build -j8 --target llama-bench 2>&1 | tail -n 1

if [ ! -s models/Llama-3.2-1B-Instruct-Q4_K_M.gguf ]; then
  log "WRITE: download Llama-3.2-1B-Instruct-Q4_K_M.gguf (0.8 GB) to $D/models (login node has network)"
  mkdir -p models
  curl -sSfL -o models/Llama-3.2-1B-Instruct-Q4_K_M.gguf \
    https://huggingface.co/bartowski/Llama-3.2-1B-Instruct-GGUF/resolve/main/Llama-3.2-1B-Instruct-Q4_K_M.gguf
fi
sha256sum models/Llama-3.2-1B-Instruct-Q4_K_M.gguf
lscpu | grep -E "Model name|^CPU\(s\)|Flags" | cut -c1-200
echo SETUP_OK
