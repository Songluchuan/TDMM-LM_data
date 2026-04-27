#!/usr/bin/env bash
# run_8gpus_nolog.sh
set -euo pipefail

GPU_LIST="${GPU_LIST:-0 1 2 3 4 5 6 7}"   # default GPU set: 0-7
ENV_NAME="${ENV_NAME:-diffpose}"               # conda env
SCRIPT="${SCRIPT:-demo.py}"        # Python script
EXTRA_ARGS="${EXTRA_ARGS:-}"              # extra pass-through args

#
read -r -a GPUS <<< "$GPU_LIST"
NGPUS=${#GPUS[@]}

if (( $# < NGPUS )); then
  echo "Usage: $0 DIR0 DIR1 ... DIR7   (provide at least ${NGPUS} directories)"
  exit 1
fi
TRACK_DIRS=("$@")

# ---- init & activate conda ----
if command -v conda >/dev/null 2>&1; then
  # shellcheck disable=SC1090
  source "$(conda info --base)/etc/profile.d/conda.sh"
elif [[ -f "$HOME/miniconda3/etc/profile.d/conda.sh" ]]; then
  source "$HOME/miniconda3/etc/profile.d/conda.sh"
elif [[ -f "$HOME/anaconda3/etc/profile.d/conda.sh" ]]; then
  source "$HOME/anaconda3/etc/profile.d/conda.sh"
else
  echo "ERROR: conda not found."
  exit 1
fi

echo "[INFO] ENV=$ENV_NAME | GPUS=(${GPUS[*]}) | SCRIPT=$SCRIPT"
trap 'echo "[INFO] interrupt signal, terminating subprocesses"; jobs -pr | xargs -r kill || true' INT TERM

# ---- launch on all GPUs concurrently (no log files; output goes straight to terminal) ----
for i in $(seq 0 $((NGPUS-1))); do
  gpu="${GPUS[$i]}"
  dir="${TRACK_DIRS[$i]}"

  if [[ ! -d "$dir" ]]; then
    echo "WARN: directory does not exist, skipping: $dir"
    continue
  fi

  echo "[RUN ] GPU=$gpu  DIR=$dir"
  (
    export CUDA_VISIBLE_DEVICES="$gpu"
    # To prefix each line with [GPU x] for clearer output, uncomment the next line:
    # stdbuf -oL -eL python "$SCRIPT" --Track_Dirt "$dir" $EXTRA_ARGS | sed -u "s/^/[GPU ${gpu}] /"
    python ./sixdrepnet/demo.py --snapshot 6DRepNet_300W_LP_AFLW2000.pth --input_dirt "$dir" $EXTRA_ARGS
  ) &
done

# wait for all concurrent tasks to finish
wait
echo "[DONE] all tasks finished."
