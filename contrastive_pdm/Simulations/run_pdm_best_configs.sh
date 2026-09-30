#!/usr/bin/env bash
# Retrain the predictive-dendrite settings plot_results.py selects.
#
# One already chosen setting per model, not a learning-rate search.
# ReLU, no gating, fixed positive beta:
#   MNIST beta 0.05, CIFAR-10 beta 0.0025
#   lr-config-idx: 0=low, 1=base, 2=high
#
# Backprop and fixed-input-weight baselines are run_bp_sweep.sh.
#
# Usage (from anywhere):
#   bash contrastive_pdm/Simulations/run_pdm_best_configs.sh
#
# Skip a config that already has a finished 5-seed result (default).
#   SKIP_COMPLETED=0 bash contrastive_pdm/Simulations/run_pdm_best_configs.sh
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

# dataset|script|lr-config-idx|neural-lr-start
RUNS=(
  "MNIST|ContrastivePredictiveErrorPropagationWeakClamp_experiment.py|1|1.6"
  "MNIST|ContrastivePredictiveErrorPropagationFullClamp_experiment.py|0|1.0"
  "MNIST|PredictiveErrorPropagationWeakClampNoFreePhase_experiment.py|1|0.6"
  "MNIST|PredictiveErrorPropagationFullClampNoFreePhase_experiment.py|1|0.6"
  "CIFAR10|ContrastivePredictiveErrorPropagationWeakClamp_experiment.py|2|1.6"
  "CIFAR10|ContrastivePredictiveErrorPropagationFullClamp_experiment.py|2|0.6"
  "CIFAR10|PredictiveErrorPropagationWeakClampNoFreePhase_experiment.py|1|1.6"
  "CIFAR10|PredictiveErrorPropagationFullClampNoFreePhase_experiment.py|0|1.0"
)

algo_for() {
  case "$1" in
    ContrastivePredictiveErrorPropagationWeakClamp_experiment.py) echo CPEP_WeakClamp ;;
    ContrastivePredictiveErrorPropagationFullClamp_experiment.py) echo CPEP_FullClamp ;;
    PredictiveErrorPropagationWeakClampNoFreePhase_experiment.py) echo PEP_WeakClamp_NoFreePhase ;;
    PredictiveErrorPropagationFullClampNoFreePhase_experiment.py) echo PEP_FullClamp_NoFreePhase ;;
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
test_keys = ["test_acc"]

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

echo "Predictive-dendrite best configs"
echo "  python: $PYTHON_EXEC"
echo "  cells:  ${#RUNS[@]} selected settings"
echo "  beta:   MNIST 0.05, CIFAR-10 0.0025, ReLU, ungated"
echo "  skip:   SKIP_COMPLETED=${SKIP_COMPLETED}"

ran=0
skipped=0
for run in "${RUNS[@]}"; do
  dataset="${run%%|*}"
  rest="${run#*|}"
  script="${rest%%|*}"
  rest="${rest#*|}"
  lr_idx="${rest%%|*}"
  nlr="${rest#*|}"
  exp_dir="$HERE/$dataset"
  if [ ! -f "$exp_dir/$script" ]; then
    echo "ERROR: missing $exp_dir/$script"
    exit 1
  fi
  algo="$(algo_for "$script")"
  results_dir="$HERE/Results/$dataset/$algo"
  extra="$(extra_args_for "$script" "$dataset" "$lr_idx" "$nlr")"
  beta_arg="$(beta_arg_for "$script" "$dataset")"
  if [ "$SKIP_COMPLETED" = "1" ] && task_already_done "$results_dir" "$lr_idx" "$nlr" "$beta_arg"; then
    echo "skip (complete): $dataset $script $extra"
    skipped=$((skipped + 1))
    continue
  fi
  log_dir="$exp_dir/logs/${script%.py}"
  mkdir -p "$log_dir"
  nlr_tag="$("$PYTHON_EXEC" -c 'import sys; print(f"{float(sys.argv[1]):g}")' "$nlr")"
  log_file="$log_dir/lr${lr_idx}_nlr${nlr_tag}.log"
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

echo "Finished. ran=${ran} skipped=${skipped}"
echo "Results: $HERE/Results/<MNIST|CIFAR10>/{CPEP_WeakClamp,CPEP_FullClamp,PEP_WeakClamp_NoFreePhase,PEP_FullClamp_NoFreePhase}/"
