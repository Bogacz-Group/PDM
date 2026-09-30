#!/usr/bin/env bash
# Sequential learning-rate sweep for the backprop baselines:
#   BP_Adam.py, BP_SGD.py, FixedInputWeightsBP_Adam.py, FixedInputWeightsBP_SGD.py
#
# One process at a time. Does not request a GPU, load CUDA, or set a TPU device.
# get_device() uses CUDA when it is available, otherwise the CPU.
#
# The grids are the ones stored in Results/ (ReLU, 5 seeds, 15 epochs):
#   MNIST:   3e-2 1e-2 5e-3 3e-3 1e-3 5e-4 3e-4 1e-4
#   CIFAR-10: 1e-2 5e-3 3e-3 1e-3 5e-4 3e-4 1e-4 5e-5
#
# Usage (from anywhere):
#   bash contrastive_pdm/Simulations/run_bp_sweep.sh
#
# Skip a learning rate that already has a finished 5-seed result (default).
#   SKIP_COMPLETED=0 bash contrastive_pdm/Simulations/run_bp_sweep.sh
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

if [ -f "$HERE/venv/bin/activate" ]; then
  # shellcheck disable=SC1091
  source "$HERE/venv/bin/activate"
elif [ -f "$HERE/.venv/bin/activate" ]; then
  # shellcheck disable=SC1091
  source "$HERE/.venv/bin/activate"
fi

PYTHON_EXEC="${PYTHON_EXEC:-python}"
if ! "$PYTHON_EXEC" -c "import numpy, torch" >/dev/null 2>&1; then
  echo "ERROR: $PYTHON_EXEC cannot import numpy and torch."
  echo "From contrastive_pdm/Simulations: python -m venv .venv && source .venv/bin/activate && pip install -r requirements.txt"
  exit 1
fi

SKIP_COMPLETED="${SKIP_COMPLETED:-1}"

SCRIPTS=(
  BP_Adam.py
  BP_SGD.py
  FixedInputWeightsBP_Adam.py
  FixedInputWeightsBP_SGD.py
)
# dataset|learning rates
GRIDS=(
  "MNIST|3e-2 1e-2 5e-3 3e-3 1e-3 5e-4 3e-4 1e-4"
  "CIFAR10|1e-2 5e-3 3e-3 1e-3 5e-4 3e-4 1e-4 5e-5"
)

algo_for() {
  case "$1" in
    BP_Adam.py) echo BP_Adam ;;
    BP_SGD.py) echo BP_SGD ;;
    FixedInputWeightsBP_Adam.py) echo FixedInputWeightsBP_Adam ;;
    FixedInputWeightsBP_SGD.py) echo FixedInputWeightsBP_SGD ;;
    *)
      echo "ERROR: unknown script $1" >&2
      return 1
      ;;
  esac
}

# Exit 0 when a finished 5-seed ReLU run already exists at this learning rate.
already_done() {
  "$PYTHON_EXEC" - "$1" "$2" <<'PY'
import json
import sys
from pathlib import Path

import numpy as np

results_dir, lr = sys.argv[1:3]
want_lr = float(lr)
test_keys = ["bp_test_acc", "fixed_input_weights_bp_test_acc", "test_acc"]

for npz in Path(results_dir).glob("*_accuracy_arrays.npz"):
    hp_path = Path(str(npz).replace("_accuracy_arrays.npz", "_hyperparams.json"))
    if not hp_path.exists():
        continue
    hp = json.loads(hp_path.read_text())
    if hp.get("hidden_activation", hp.get("activation")) != "relu":
        continue
    try:
        if abs(float(hp["lr"]) - want_lr) > 1e-12:
            continue
    except (KeyError, TypeError, ValueError):
        continue
    with np.load(npz) as z:
        arr = next((z[k] for k in test_keys if k in z.files), None)
        if arr is None or getattr(arr, "shape", (0,))[0] < 5:
            continue
    sys.exit(0)
sys.exit(1)
PY
}

echo "Backprop learning-rate sweep"
echo "  python: $PYTHON_EXEC"
echo "  device: CUDA if available, otherwise CPU"
echo "  skip:   SKIP_COMPLETED=${SKIP_COMPLETED}"

ran=0
skipped=0
for grid in "${GRIDS[@]}"; do
  dataset="${grid%%|*}"
  lrs="${grid#*|}"
  exp_dir="$HERE/$dataset"
  for script in "${SCRIPTS[@]}"; do
    if [ ! -f "$exp_dir/$script" ]; then
      echo "ERROR: missing $exp_dir/$script"
      exit 1
    fi
    algo="$(algo_for "$script")"
    results_dir="$HERE/Results/$dataset/$algo"
    for lr in $lrs; do
      extra="--lr ${lr} --activation relu"
      if [ "$SKIP_COMPLETED" = "1" ] && already_done "$results_dir" "$lr"; then
        echo "skip (complete): $dataset $script $extra"
        skipped=$((skipped + 1))
        continue
      fi
      log_dir="$exp_dir/logs/${script%.py}"
      mkdir -p "$log_dir"
      lr_tag="$("$PYTHON_EXEC" -c 'import sys; print(f"{float(sys.argv[1]):g}")' "$lr")"
      log_file="$log_dir/lr${lr_tag}.log"
      echo "run: $dataset $script $extra"
      (
        cd "$exp_dir"
        # shellcheck disable=SC2086
        "$PYTHON_EXEC" "$script" $extra
      ) >"$log_file" 2>&1 || {
        echo "FAILED: $dataset $script $extra"
        echo "log: $log_file"
        exit 1
      }
      ran=$((ran + 1))
    done
  done
done

echo "Sweep finished. ran=${ran} skipped=${skipped}"
echo "Results: $HERE/Results/<MNIST|CIFAR10>/{BP_Adam,BP_SGD,FixedInputWeightsBP_Adam,FixedInputWeightsBP_SGD}/"
