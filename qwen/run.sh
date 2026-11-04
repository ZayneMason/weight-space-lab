#!/usr/bin/env bash
# Paired evaluation of the frozen published edit on Linux or WSL.
set -euo pipefail
TOOLS=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
LAB=${WEIGHT_SPACE_LAB_HOME:-"$HOME/.local/share/weight-space-lab"}
PYTHON=${WEIGHT_SPACE_LAB_PYTHON:-"$LAB/.venv/bin/python"}
MODE=${1:-pilot}
case "$MODE" in
  smoke) OPTIONS=(--pilot --limit 2 --max-tokens 256) ;;
  pilot) OPTIONS=(--pilot --limit 2 --max-tokens 32768) ;;
  full) OPTIONS=() ;;
  *) printf 'Usage: run.sh smoke|pilot|full\n' >&2; exit 2 ;;
esac
if [[ ! -x "$PYTHON" ]]; then
  printf 'Python environment missing; follow qwen/README.md.\n' >&2
  exit 2
fi
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}
export HF_HUB_OFFLINE=1
export VLLM_USE_FLASHINFER_SAMPLER=${VLLM_USE_FLASHINFER_SAMPLER:-1}
export PATH="$(dirname -- "$PYTHON"):$PATH"
SITE=$("$PYTHON" -c 'import sysconfig; print(sysconfig.get_path("purelib"))')
if [[ -z ${CUDA_HOME:-} && -d "$SITE/nvidia/cu13" ]]; then
  export CUDA_HOME="$SITE/nvidia/cu13"
fi
if [[ -n ${CUDA_HOME:-} && -f "$CUDA_HOME/lib/libcudart.so.13" ]]; then
  mkdir -p "$LAB/cuda-link"
  if [[ ! -e "$LAB/cuda-link/libcudart.so" ]]; then
    ln -s "$CUDA_HOME/lib/libcudart.so.13" "$LAB/cuda-link/libcudart.so"
  fi
  export LIBRARY_PATH="$LAB/cuda-link:${LIBRARY_PATH:-}"
  export LD_LIBRARY_PATH="$CUDA_HOME/lib:${LD_LIBRARY_PATH:-}"
fi
if [[ -d /usr/lib/wsl/lib ]]; then
  export LIBRARY_PATH="/usr/lib/wsl/lib:${LIBRARY_PATH:-}"
  export LD_LIBRARY_PATH="/usr/lib/wsl/lib:${LD_LIBRARY_PATH:-}"
fi
OUT="$LAB/$MODE-paired"
mkdir -p "$OUT"
for ARM in baseline p50c; do
  "$PYTHON" "$TOOLS/reproduce_qwen.py" run "$LAB/experiment" "$ARM" "$OUT/$ARM" \
    --gpu-memory "${WEIGHT_SPACE_GPU_MEMORY:-0.65}" "${OPTIONS[@]}" \
    > "$OUT/$ARM.log" 2>&1
done
"$PYTHON" "$TOOLS/reproduce_qwen.py" compare "$OUT/baseline" "$OUT/p50c" "$OUT/comparison.json"
