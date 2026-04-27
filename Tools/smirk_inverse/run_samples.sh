#!/usr/bin/env bash
# run_8gpus_nolog.sh
set -euo pipefail

GPU_LIST="${GPU_LIST:-0 1 2 3}" 
ENV_NAME="${ENV_NAME:-smirk}"    
SCRIPT="${SCRIPT:-datawheel_sample.py}" 
EXTRA_ARGS="${EXTRA_ARGS:-}"    

#
read -r -a GPUS <<< "$GPU_LIST"
NGPUS=${#GPUS[@]}

if (( $# < NGPUS )); then
  echo "Use: $0 DIR0 DIR1 ... DIR7   (${NGPUS}）"
  exit 1
fi
TRACK_DIRS=("$@")

if command -v conda >/dev/null 2>&1; then
  # shellcheck disable=SC1090
  source "$(conda info --base)/etc/profile.d/conda.sh"
elif [[ -f "$HOME/miniconda3/etc/profile.d/conda.sh" ]]; then
  source "$HOME/miniconda3/etc/profile.d/conda.sh"
elif [[ -f "$HOME/anaconda3/etc/profile.d/conda.sh" ]]; then
  source "$HOME/anaconda3/etc/profile.d/conda.sh"
else
  echo "ERROR: conda error"
  exit 1
fi
conda activate "$ENV_NAME"

echo "[INFO] ENV=$ENV_NAME | GPUS=(${GPUS[*]}) | SCRIPT=$SCRIPT"
trap 'echo "[INFO] stop"; jobs -pr | xargs -r kill || true' INT TERM


for i in $(seq 0 $((NGPUS-1))); do
  gpu="${GPUS[$i]}"
  dir="${TRACK_DIRS[$i]}"

  if [[ ! -d "$dir" ]]; then
    echo "WARN: skip: $dir"
    continue
  fi

  echo "[RUN ] GPU=$gpu  DIR=$dir"
  (
    export CUDA_VISIBLE_DEVICES="$gpu"
    python "$SCRIPT" --checkpoint pretrained_models/SMIRK_em1.pt --crop --render_orig --Track_Dirt "$dir" $EXTRA_ARGS
  ) &
done

wait
echo "[DONE]"
