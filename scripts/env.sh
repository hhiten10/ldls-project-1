# Shared paths. Source this from every script:  source "$(dirname "$0")/env.sh"
# Heavy files (llama.cpp build, models, venv) live on the WSL ext4 disk, not /mnt/c,
# because /mnt/c (9p) is ~10x slower for builds and mmap'd model loads.
export WORK="${WORK:-$HOME/p1work}"
export LLAMA="$WORK/llama.cpp"
export BUILD="$LLAMA/build-cpu"
export BIN="$BUILD/bin"
export MODELS="$WORK/models"
export VENV="$WORK/venv"
export REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export RESULTS="$REPO/results"
# Pinned llama.cpp commit (filled in by 01_build.sh on first build; keep fixed for all runs)
export LLAMA_COMMIT_FILE="$REPO/docs/llama_cpp_commit.txt"
