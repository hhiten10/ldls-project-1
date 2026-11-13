#!/usr/bin/env bash
# Python venv for GGUF conversion, plotting and the Mamba-3 PyTorch reference (CPU-only torch).
set -euo pipefail
source "$(dirname "$0")/env.sh"
mkdir -p "$WORK"
[ -x "$VENV/bin/python" ] || python3 -m venv "$VENV"
"$VENV/bin/python" -m pip install -q --upgrade pip
"$VENV/bin/python" -m pip install -q torch --index-url https://download.pytorch.org/whl/cpu
"$VENV/bin/python" -m pip install -q "huggingface_hub[cli]" numpy sentencepiece safetensors \
  transformers einops matplotlib pandas gguf protobuf
"$VENV/bin/python" -c "import torch,transformers;print('torch',torch.__version__,'transformers',transformers.__version__)"
