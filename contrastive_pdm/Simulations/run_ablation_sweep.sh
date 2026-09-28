#!/usr/bin/env bash
# Sequential Fig. 4a / Supplementary Fig. 6 runs.
#
# Retrains only the learning-rate setting plot_results.py selects for each
# curve (highest final validation accuracy):
#   predictive-error propagation: ReLU, no gating, fixed positive beta
#   MNIST beta 0.05, CIFAR-10 beta 0.0025
#   lr-config-idx: 0=low, 1=base, 2=high
#   backprop and fixed-input-weight baselines: ReLU, one --lr each
#
# Usage (from anywhere):
#   bash contrastive_pdm/Simulations/run_ablation_sweep.sh
#
# Skip configs that already have a finished 5-seed result (default).
#   SKIP_COMPLETED=0 bash contrastive_pdm/Simulations/run_ablation_sweep.sh
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$HERE/.." && pwd)"

if [ -f "$ROOT/.venv/bin/activate" ]; then
  # shellcheck disable=SC1091
  source "$ROOT/.venv/bin/activate"
fi

PYTHON_EXEC="${PYTHON_EXEC:-python}"
if ! "$PYTHON_EXEC" -c "import numpy, torch" >/dev/null 2>&1; then
  echo "ERROR: $PYTHON_EXEC cannot import numpy and torch."
  echo "From the contrastive_pdm directory: python -m venv .venv && source .venv/bin/activate && pip install -r requirements.txt"
  exit 1
fi

SKIP_COMPLETED="${SKIP_COMPLETED:-1}"

# pep|dataset|script|lr-config-idx|neural-lr-start
# bp|dataset|script|lr
RUNS=(
  "pep|MNIST|ContrastivePredictiveErrorPropagationWeakClamp_experiment.py|1|1.6"
  "pep|MNIST|ContrastivePredictiveErrorPropagationFullClamp_experiment.py|0|1.0"
  "pep|MNIST|PredictiveErrorPropagationWeakClampNoFreePhase_experiment.py|1|0.6"
  "pep|MNIST|PredictiveErrorPropagationFullClampNoFreePhase_experiment.py|1|0.6"
  "bp|MNIST|BP_Adam.py|0.0005"
  "bp|MNIST|BP_SGD.py|0.03"
  "bp|MNIST|FixedInputWeightsBP_Adam.py|0.003"
  "bp|MNIST|FixedInputWeightsBP_SGD.py|0.03"
  "pep|CIFAR10|ContrastivePredictiveErrorPropagationWeakClamp_experiment.py|2|1.6"
  "pep|CIFAR10|ContrastivePredictiveErrorPropagationFullClamp_experiment.py|2|0.6"
  "pep|CIFAR10|PredictiveErrorPropagationWeakClampNoFreePhase_experiment.py|1|1.6"
  "pep|CIFAR10|PredictiveErrorPropagationFullClampNoFreePhase_experiment.py|0|1.0"
  "bp|CIFAR10|BP_Adam.py|5e-05"
  "bp|CIFAR10|BP_SGD.py|0.001"
  "bp|CIFAR10|FixedInputWeightsBP_Adam.py|0.001"
  "bp|CIFAR10|FixedInputWeightsBP_SGD.py|0.01"
)

algo_for() {
  case "$1" in
    ContrastivePredictiveErrorPropagationWeakClamp_experiment.py) echo CPEP_WeakClamp ;;
    ContrastivePredictiveErrorPropagationFullClamp_experiment.py) echo CPEP_FullClamp ;;
    PredictiveErrorPropagationWeakClampNoFreePhase_experiment.py) echo PEP_WeakClamp_NoFreePhase ;;
    PredictiveErrorPropagationFullClampNoFreePhase_experiment.py) echo PEP_FullClamp_NoFreePhase ;;
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

beta_for() {
  case "$1" in
    MNIST) echo 0.05 ;;
    CIFAR10) echo 0.0025 ;;
    *)
      echo "ERROR: unknown dataset $1" >&2
      return 1
      ;;
  esac
}

# "-" means the script does not take --beta (full-clamp models).
beta_arg_for() {
  case "$1" in
    *WeakClamp*) beta_for "$2" ;;
    *) echo "-" ;;
  esac
}

extra_args_for() {
  local script="$1"
  local dataset="$2"
  local lr_idx="$3"
  local nlr="$4"
  local beta
  local common="--lr-config-idx ${lr_idx} --neural-lr-start ${nlr} --activation relu"
  case "$script" in
    ContrastivePredictiveErrorPropagationWeakClamp_experiment.py)
      beta="$(beta_for "$dataset")"
      echo "${common} --beta ${beta} --no-random-sign-beta"
      ;;
    PredictiveErrorPropagationWeakClampNoFreePhase_experiment.py)
      beta="$(beta_for "$dataset")"
      echo "${common} --beta ${beta}"
      ;;
    *)
      echo "${common}"
      ;;
  esac
}

# Exit 0 when Results already contain a finished 5-seed npz for this cell.
task_already_done() {
  "$PYTHON_EXEC" - "$1" "$2" "$3" "$4" <<'PY'
import json
import sys
from pathlib import Path

import numpy as np

results_dir, lr_idx, nlr, beta = sys.argv[1:5]
lr_idx = int(lr_idx)
nlr = float(nlr)
want_beta = None if beta == "-" else float(beta)
test_keys = ["bp_test_acc", "fixed_input_weights_bp_test_acc", "test_acc"]

for npz in Path(results_dir).glob("*_accuracy_arrays.npz"):
    hp_path = Path(str(npz).replace("_accuracy_arrays.npz", "_hyperparams.json"))
    if not hp_path.exists():
        continue
    hp = json.loads(hp_path.read_text())
    if hp.get("activation") != "relu" or hp.get("use_gating"):
        continue
    if hp.get("lr_config_idx") != lr_idx:
        continue
    try:
        if abs(float(hp["neural_lr_start"]) - nlr) > 1e-9:
            continue
    except (KeyError, TypeError, ValueError):
        continue
    if want_beta is not None:
        try:
            if abs(float(hp["beta"]) - want_beta) > 1e-9:
                continue
        except (KeyError, TypeError, ValueError):
            continue
        if hp.get("use_random_sign_beta", False):
            continue
    with np.load(npz) as z:
        arr = next((z[k] for k in test_keys if k in z.files), None)
        if arr is None or getattr(arr, "shape", (0,))[0] < 5:
            continue
    sys.exit(0)
sys.exit(1)
PY
}

# Exit 0 when a finished 5-seed backprop or fixed-input-weight run exists at this lr.
baseline_already_done() {
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
    activation = hp.get("hidden_activation", hp.get("activation"))
    if activation != "relu":
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

echo "Notebook ablation runs"
echo "  python: $PYTHON_EXEC"
echo "  cells:  ${#RUNS[@]} selected learning-rate settings"
echo "  beta:   MNIST 0.05, CIFAR-10 0.0025, ReLU, ungated"
echo "  skip:   SKIP_COMPLETED=${SKIP_COMPLETED}"

ran=0
skipped=0
for run in "${RUNS[@]}"; do
  kind="${run%%|*}"
  rest="${run#*|}"
  dataset="${rest%%|*}"
  rest="${rest#*|}"
  script="${rest%%|*}"
  rest="${rest#*|}"
  exp_dir="$HERE/$dataset"
  if [ ! -f "$exp_dir/$script" ]; then
    echo "ERROR: missing $exp_dir/$script"
    exit 1
  fi
  algo="$(algo_for "$script")"
  results_dir="$HERE/Results/$dataset/$algo"
  if [ "$kind" = "bp" ]; then
    lr="$rest"
    extra="--lr ${lr} --activation relu"
    log_tag="lr$("$PYTHON_EXEC" -c 'import sys; print(f"{float(sys.argv[1]):g}")' "$lr")"
    already_done() { baseline_already_done "$results_dir" "$lr"; }
  else
    lr_idx="${rest%%|*}"
    nlr="${rest#*|}"
    extra="$(extra_args_for "$script" "$dataset" "$lr_idx" "$nlr")"
    beta_arg="$(beta_arg_for "$script" "$dataset")"
    nlr_tag="$("$PYTHON_EXEC" -c 'import sys; print(f"{float(sys.argv[1]):g}")' "$nlr")"
    log_tag="lr${lr_idx}_nlr${nlr_tag}"
    already_done() { task_already_done "$results_dir" "$lr_idx" "$nlr" "$beta_arg"; }
  fi
  if [ "$SKIP_COMPLETED" = "1" ] && already_done; then
    echo "skip (complete): $dataset $script $extra"
    skipped=$((skipped + 1))
    continue
  fi
  log_dir="$exp_dir/logs/${script%.py}"
  mkdir -p "$log_dir"
  log_file="$log_dir/${log_tag}.log"
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

echo "Sweep finished. ran=${ran} skipped=${skipped}"
echo "Results: $HERE/Results/<MNIST|CIFAR10>/<ALGO>/"
