#!/usr/bin/env bash
# =============================================================================
# MNIST / CIFAR-10 TPU sweep for the predictive-dendrite models.
# This is the sweep that produced the grids in Results/.
# Default cell: ReLU, no gating, fixed positive beta (MNIST: 0.05; CIFAR-10: 0.0025).
# Experiment scripts use mixing weights gamma_forward=gamma_backward=0.5.
#
# Models (3x3 grid: --lr-config-idx x --neural-lr-start):
#   ContrastivePredictiveErrorPropagationWeakClamp_experiment.py
#   ContrastivePredictiveErrorPropagationFullClamp_experiment.py
#   PredictiveErrorPropagationWeakClampNoFreePhase_experiment.py
#   PredictiveErrorPropagationFullClampNoFreePhase_experiment.py
#
# Skipped: BP_SGD / BP_Adam / FixedInputWeightsBP_* (see run_bp_sweep.sh)
#
# Parallelism: up to NUM_CHIPS jobs at once (default 4 for ct6e-standard-4t),
# each pinned to one TPU chip via TPU_VISIBLE_CHIPS.
#
# Usage (on the TPU VM, after setup):
#
#     bash contrastive_pdm/Simulations/run_pdm_sweep_tpu.sh --dataset CIFAR10
#     bash contrastive_pdm/Simulations/run_pdm_sweep_tpu.sh --dataset MNIST
#
# Equivalents: DATASET=MNIST bash ...   or   bash ... MNIST
#
# Options (CLI or env):
#   --activation relu|hard_sigmoid|both     (env ACTIVATION, default relu)
#   --gating on|off|both                    (env GATING, default off)
#   --beta-mode positive|random-sign|positive-sweep
#                                           (env BETA_MODE, default positive)
#
#   positive (default): CPEP-WeakClamp and PEP-WeakClamp-NoFreePhase use
#                a single fixed positive beta. Defaults: MNIST 0.05, CIFAR-10
#                0.0025. Override with POSITIVE_BETA. Contrastive models pass
#                --no-random-sign-beta.
#   random-sign: contrastive models use |beta|=0.05 with a random sign per
#                batch. Weak-clamp no-free-phase still uses positive beta
#                (dataset default).
#   positive-sweep: CPEP-WeakClamp and PEP-WeakClamp-NoFreePhase each run a
#                fixed-positive-beta grid (default: 5e-5, 5e-4, 0.0025, 0.005,
#                0.01, 0.025, 0.05). Override with POSITIVE_BETAS="...".
#                Full-clamp models ignore beta (they clamp the output) and are
#                still run once per cell.
#
# Logs: Simulations/<DATASET>/logs_tpu/<script>/...
# Results: Simulations/Results/<DATASET>/<ALGO>/
#
# Resume: re-run this script after recreating the VM. By default it skips any
# config that already has a complete 5-seed *_accuracy_arrays.npz under
# Simulations/Results/<DATASET>/<ALGO>/ matching activation, gating, and beta.
# Force a full re-run with SKIP_COMPLETED=0.
# =============================================================================
set -euo pipefail

NUM_CHIPS="${NUM_CHIPS:-4}"

print_usage() {
  echo "Usage: bash $0 [--dataset MNIST|CIFAR10] [--activation relu|hard_sigmoid|both]"
  echo "         [--gating on|off|both] [--beta-mode positive|random-sign|positive-sweep]"
  echo "  Defaults: --activation relu --gating off --beta-mode positive"
  echo "            POSITIVE_BETA=0.05 (MNIST) or 0.0025 (CIFAR10)"
  echo "  Env: DATASET ACTIVATION GATING BETA_MODE POSITIVE_BETA POSITIVE_BETAS SKIP_COMPLETED NUM_CHIPS"
}

# Dataset / activation / gating / beta-mode: CLI flags, env, or positional dataset.
DATASET="${DATASET:-CIFAR10}"
ACTIVATION="${ACTIVATION:-relu}"
GATING="${GATING:-off}"
BETA_MODE="${BETA_MODE:-positive}"
# If unset, filled after DATASET is resolved (MNIST 0.05, CIFAR-10 0.0025).
_POSITIVE_BETA_OVERRIDE="${POSITIVE_BETA:-}"
while [[ $# -gt 0 ]]; do
  case "$1" in
    --dataset)
      DATASET="$2"
      shift 2
      ;;
    --dataset=*)
      DATASET="${1#*=}"
      shift
      ;;
    --activation)
      ACTIVATION="$2"
      shift 2
      ;;
    --activation=*)
      ACTIVATION="${1#*=}"
      shift
      ;;
    --gating)
      GATING="$2"
      shift 2
      ;;
    --gating=*)
      GATING="${1#*=}"
      shift
      ;;
    --beta-mode)
      BETA_MODE="$2"
      shift 2
      ;;
    --beta-mode=*)
      BETA_MODE="${1#*=}"
      shift
      ;;
    MNIST|mnist|CIFAR10|cifar10|CIFAR-10|cifar-10)
      DATASET="$1"
      shift
      ;;
    -h|--help)
      print_usage
      exit 0
      ;;
    *)
      echo "ERROR: unknown argument '$1'"
      print_usage
      exit 1
      ;;
  esac
done
case "$(printf '%s' "$DATASET" | tr '[:lower:]' '[:upper:]')" in
  MNIST) DATASET=MNIST ;;
  CIFAR10|CIFAR-10) DATASET=CIFAR10 ;;
  *)
    echo "ERROR: dataset must be MNIST or CIFAR10, got '$DATASET'"
    exit 1
    ;;
esac

ACTIVATION_LIST=()
case "$(printf '%s' "$ACTIVATION" | tr '[:upper:]' '[:lower:]')" in
  relu) ACTIVATION_LIST=(relu) ;;
  hard_sigmoid|hardsigmoid|hard-sigmoid) ACTIVATION_LIST=(hard_sigmoid) ;;
  both) ACTIVATION_LIST=(relu hard_sigmoid) ;;
  *)
    echo "ERROR: --activation must be relu, hard_sigmoid, or both, got '$ACTIVATION'"
    exit 1
    ;;
esac

GATING_LIST=()
case "$(printf '%s' "$GATING" | tr '[:upper:]' '[:lower:]')" in
  on|true|1|gated|yes) GATING_LIST=(1) ;;
  off|false|0|ungated|no|none) GATING_LIST=(0) ;;
  both) GATING_LIST=(0 1) ;;
  *)
    echo "ERROR: --gating must be on, off, or both, got '$GATING'"
    exit 1
    ;;
esac

case "$(printf '%s' "$BETA_MODE" | tr '[:upper:]' '[:lower:]')" in
  positive|fixed) BETA_MODE=positive ;;
  random-sign|randsign|random_sign|random) BETA_MODE=random-sign ;;
  positive-sweep|positive_sweep|sweep) BETA_MODE=positive-sweep ;;
  *)
    echo "ERROR: --beta-mode must be positive, random-sign, or positive-sweep, got '$BETA_MODE'"
    exit 1
    ;;
esac
# shellcheck disable=SC2206
POSITIVE_BETAS=(${POSITIVE_BETAS:-5e-5 5e-4 0.0025 0.005 0.01 0.025 0.05})

# Resolve repo + run scripts from Simulations/$DATASET (default CIFAR10)
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
if [ -d "$HERE/$DATASET" ]; then
  # Invoked as Simulations/run_ablation_sweep_tpu.sh.template
  REPO_ROOT="$(cd "$HERE/.." && pwd)"
  EXP_DIR="$HERE/$DATASET"
elif [ -f "$HERE/ContrastivePredictiveErrorPropagationWeakClamp_experiment.py" ]; then
  # Invoked from a copy inside Simulations/MNIST or Simulations/CIFAR10
  EXP_DIR="$HERE"
  DATASET="$(basename "$HERE")"
  REPO_ROOT="$(cd "$HERE/../.." && pwd)"
else
  echo "ERROR: cannot find Simulations/$DATASET relative to $HERE"
  exit 1
fi

if [ -n "$_POSITIVE_BETA_OVERRIDE" ]; then
  POSITIVE_BETA="$_POSITIVE_BETA_OVERRIDE"
elif [ "$DATASET" = "CIFAR10" ]; then
  POSITIVE_BETA=0.0025
else
  POSITIVE_BETA=0.05
fi

if [ -n "${VENV_DIR:-}" ]; then
  :
elif [ -f "$HERE/venv/bin/activate" ]; then
  VENV_DIR="$HERE/venv"
elif [ -f "$HERE/.venv/bin/activate" ]; then
  VENV_DIR="$HERE/.venv"
else
  VENV_DIR="$REPO_ROOT/.venv"
fi
if [ -f "$VENV_DIR/bin/activate" ]; then
  # shellcheck disable=SC1091
  source "$VENV_DIR/bin/activate"
fi
export PJRT_DEVICE="${PJRT_DEVICE:-TPU}"

# uv-managed CPython ships libpython outside the default linker path; torch_xla needs it.
if [ -z "${LD_LIBRARY_PATH:-}" ] || [[ "${LD_LIBRARY_PATH}" != *"/uv/python/"* ]]; then
  _UV_LIBPY="$(find "${HOME}/.local/share/uv/python" -name 'libpython3.12.so.1.0' 2>/dev/null | head -1 || true)"
  if [ -n "${_UV_LIBPY}" ]; then
    export LD_LIBRARY_PATH="$(dirname "${_UV_LIBPY}"):${LD_LIBRARY_PATH:-}"
  fi
  unset _UV_LIBPY
fi

# Newer TPU VM metadata uses TPU_ACCELERATOR_TYPE; torch_xla 2.9 still reads
# ACCELERATOR_TYPE from the MDS dict and KeyErrors. Skip MDS and set topology
# from the ct6e-standard-4t / v6e-4 tpu-env (override via env if needed).
export TPU_SKIP_MDS_QUERY="${TPU_SKIP_MDS_QUERY:-1}"
export ACCELERATOR_TYPE="${ACCELERATOR_TYPE:-v6e-4}"
export TPU_ACCELERATOR_TYPE="${TPU_ACCELERATOR_TYPE:-v6e-4}"
export TPU_HOST_BOUNDS="${TPU_HOST_BOUNDS:-1,1,1}"
export TPU_CHIPS_PER_HOST_BOUNDS="${TPU_CHIPS_PER_HOST_BOUNDS:-2,2,1}"
export TPU_WORKER_HOSTNAMES="${TPU_WORKER_HOSTNAMES:-localhost}"
export TPU_WORKER_ID="${TPU_WORKER_ID:-0}"
export WORKER_ID="${WORKER_ID:-0}"

PYTHON_EXEC="${PYTHON_EXEC:-$(command -v python)}"

cd "$EXP_DIR"

# --- Models (non-BP only) ----------------------------------------------------
ABLATION_SCRIPTS=(
  ContrastivePredictiveErrorPropagationWeakClamp_experiment.py
  ContrastivePredictiveErrorPropagationFullClamp_experiment.py
  PredictiveErrorPropagationWeakClampNoFreePhase_experiment.py
  PredictiveErrorPropagationFullClampNoFreePhase_experiment.py
)

LR_CONFIG_IDXS=(0 1 2)
NEURAL_LR_LIST=(0.6 1.0 1.6)

# Resume-friendly: skip cells that already have a complete 5-seed npz in Results
# matching this activation / gating / beta. Set SKIP_COMPLETED=0 to force a re-run.
SKIP_COMPLETED="${SKIP_COMPLETED:-1}"
RESULTS_ROOT="${RESULTS_ROOT:-$EXP_DIR/../Results/$DATASET}"

# script -> Results/<ALGO> folder name
declare -A RESULTS_ALGO=(
  [ContrastivePredictiveErrorPropagationWeakClamp_experiment.py]=CPEP_WeakClamp
  [ContrastivePredictiveErrorPropagationFullClamp_experiment.py]=CPEP_FullClamp
  [PredictiveErrorPropagationWeakClampNoFreePhase_experiment.py]=PEP_WeakClamp_NoFreePhase
  [PredictiveErrorPropagationFullClampNoFreePhase_experiment.py]=PEP_FullClamp_NoFreePhase
)
LR_LABELS=(low base high)

# Contrastive weak-clamp + PEM: random-sign or fixed-positive beta.
# Weak-clamp no-free-phase: positive beta only.
# Full-clamp models clamp the output and do not take --beta.
script_uses_beta() {
  case "$1" in
    ContrastivePredictiveErrorPropagationWeakClamp_experiment.py|\
    PredictiveErrorPropagationWeakClampNoFreePhase_experiment.py|\
    ContrastivePredictiveEntropy_experiment.py)
      return 0
      ;;
    *) return 1 ;;
  esac
}

script_uses_random_sign_beta() {
  case "$1" in
    ContrastivePredictiveErrorPropagationWeakClamp_experiment.py|\
    ContrastivePredictiveEntropy_experiment.py)
      return 0
      ;;
    *) return 1 ;;
  esac
}

gating_args_for() {
  if [ "$1" = "1" ]; then
    echo "--gating"
  else
    echo ""
  fi
}

gate_tag_for() {
  if [ "$1" = "1" ]; then
    echo "gated"
  else
    echo "ungated"
  fi
}

# True if Results already contain a finished 5-seed accuracy npz for this cell
# with matching activation, gating, and beta / random-sign.
task_already_done() {
  local script="$1"
  local extra_args="$2"
  local algo="${RESULTS_ALGO[$script]:-}"
  local results_dir="$RESULTS_ROOT/$algo"
  [ -n "$algo" ] || return 1
  [ -d "$results_dir" ] || return 1

  local pattern
  if [[ "$extra_args" == *"--lr-config-idx"* ]]; then
    local lr_idx nlr label nlr_str
    lr_idx=$(echo "$extra_args" | sed -n 's/.*--lr-config-idx \([0-9]\).*/\1/p')
    nlr=$(echo "$extra_args" | sed -n 's/.*--neural-lr-start \([0-9.]*\).*/\1/p')
    label="${LR_LABELS[$lr_idx]}"
    # Match experiment scripts: f"{NEURAL_LR_START:g}" → 0.6, 1, 1.6
    nlr_str=$("$PYTHON_EXEC" -c "print(f'{float('$nlr'):g}')")
    pattern="*_lr${label}_nlr${nlr_str}_*_accuracy_arrays.npz"
  else
    pattern="PEM_${DATASET}*_accuracy_arrays.npz"
  fi

  local want_act want_gated want_beta want_randsign
  want_act=$(echo "$extra_args" | sed -n 's/.*--activation \([^ ]*\).*/\1/p')
  want_gated=0
  [[ "$extra_args" == *"--gating"* ]] && want_gated=1

  want_beta=""
  if echo " $extra_args " | grep -q -- " --beta "; then
    want_beta=$(echo "$extra_args" | sed -n 's/.*--beta \([0-9.eE+-]*\).*/\1/p')
  elif script_uses_beta "$script"; then
    want_beta="0.05"
  fi

  want_randsign=""
  if script_uses_random_sign_beta "$script"; then
    if [[ "$extra_args" == *"--no-random-sign-beta"* ]]; then
      want_randsign=0
    else
      want_randsign=1
    fi
  fi

  # MNIST CPEP at a fixed beta ≠ 1 uses synaptic LRs scaled by that beta.
  # Do not skip unscaled 5-seed runs.
  want_lrscaled=0
  if [ "$DATASET" = "MNIST" ] && [ "$want_randsign" = "0" ] && [ -n "$want_beta" ]; then
    if ! "$PYTHON_EXEC" -c "import sys; sys.exit(0 if abs(float('$want_beta')-1.0)<1e-9 else 1)"; then
      want_lrscaled=1
    fi
  fi

  "$PYTHON_EXEC" - "$results_dir" "$pattern" "$want_act" "$want_gated" "$want_beta" "$want_randsign" "$want_lrscaled" <<'PY'
import json
import re
import sys
from pathlib import Path
import numpy as np

results_dir, pattern, want_act, want_gated, want_beta, want_randsign, want_lrscaled = sys.argv[1:8]
want_gated = want_gated == "1"
want_lrscaled = want_lrscaled == "1"
files = sorted(Path(results_dir).glob(pattern))
if not files:
    sys.exit(1)

def meta_of(npz_path: Path):
    hp_path = Path(str(npz_path).replace("_accuracy_arrays.npz", "_hyperparams.json"))
    name = npz_path.name
    if hp_path.exists():
        hp = json.loads(hp_path.read_text())
        act = str(hp.get("activation", hp.get("hidden_activation", "")))
        gated = bool(hp.get("use_gating", False))
        beta = hp.get("beta", None)
        randsign = hp.get("use_random_sign_beta", None)
        lr_scale = hp.get("synaptic_lr_scale", None)
        kappa_absorbed = hp.get("kappa_absorbed", None)
        absorbed = False
        try:
            absorbed = kappa_absorbed is not None and abs(float(kappa_absorbed) - 20) < 1e-9
        except (TypeError, ValueError):
            absorbed = False
        return act, gated, beta, randsign, lr_scale, absorbed
    if "_relu" in name:
        act = "relu"
    elif "_hard_sigmoid" in name:
        act = "hard_sigmoid"
    else:
        act = "hard_sigmoid"  # default when act_tag is omitted
    m = re.search(r"_beta([0-9.eE+-]+)", name)
    beta = float(m.group(1)) if m else 1.0
    randsign = ("_randsign" in name) if m else True
    lr_scale = beta if "_lrscaled" in name else 1.0
    return act, ("_gated" in name), beta, randsign, lr_scale, False

def beta_close(got, want):
    if got is None:
        return False
    try:
        return abs(float(got) - float(want)) < 1e-9
    except (TypeError, ValueError):
        return False

def is_lrscaled(lr_scale, name, want_beta):
    if lr_scale is not None:
        try:
            target = float(want_beta) if want_beta else 0.05
            return abs(float(lr_scale) - target) < 1e-9
        except (TypeError, ValueError):
            pass
    return "_lrscaled" in name

matched = []
for p in files:
    act, gated, beta, randsign, lr_scale, absorbed = meta_of(p)
    if not absorbed:
        continue
    if act != want_act or gated != want_gated:
        continue
    if want_beta and not beta_close(beta, want_beta):
        continue
    if want_randsign != "":
        flag = bool(randsign) if randsign is not None else True
        if flag != (want_randsign == "1"):
            continue
    if want_lrscaled and not is_lrscaled(lr_scale, p.name, want_beta):
        continue
    matched.append(p)
if not matched:
    sys.exit(1)
path = max(matched, key=lambda p: p.stat().st_mtime)
try:
    data = np.load(path)
    key = "trn_acc" if "trn_acc" in data.files else data.files[0]
    arr = data[key]
    sys.exit(0 if getattr(arr, "shape", ()) and arr.shape[0] >= 5 else 1)
except Exception:
    sys.exit(1)
PY
}

maybe_add_task() {
  local script="$1"
  local extra="$2"
  local tag="$3"
  extra=$(printf '%s' "$extra" | tr -s ' ')
  extra="${extra# }"
  extra="${extra% }"
  if [ "$SKIP_COMPLETED" = "1" ] && task_already_done "$script" "$extra"; then
    echo "skip (complete): $script $extra"
    SKIPPED=$((SKIPPED + 1))
    return
  fi
  TASKS+=("${script}|${extra}|${tag}")
}

# Expand one (script, base extra args, tag prefix) over activation x gating x beta.
add_variant_tasks() {
  local script="$1"
  local base_extra="$2"
  local tag_prefix="$3"
  local act gated gating_args gate_tag b extra tag

  for act in "${ACTIVATION_LIST[@]}"; do
    for gated in "${GATING_LIST[@]}"; do
      gating_args=$(gating_args_for "$gated")
      gate_tag=$(gate_tag_for "$gated")
      if script_uses_beta "$script" && [ "$BETA_MODE" = "positive-sweep" ]; then
        for b in "${POSITIVE_BETAS[@]}"; do
          extra="${base_extra} --activation ${act} ${gating_args} --beta ${b}"
          if script_uses_random_sign_beta "$script"; then
            extra="${extra} --no-random-sign-beta"
          fi
          tag="${tag_prefix}_${act}_${gate_tag}_beta${b}"
          maybe_add_task "$script" "$extra" "$tag"
        done
      elif script_uses_beta "$script" && [ "$BETA_MODE" = "positive" ]; then
        extra="${base_extra} --activation ${act} ${gating_args} --beta ${POSITIVE_BETA}"
        if script_uses_random_sign_beta "$script"; then
          extra="${extra} --no-random-sign-beta"
        fi
        tag="${tag_prefix}_${act}_${gate_tag}_beta${POSITIVE_BETA}"
        maybe_add_task "$script" "$extra" "$tag"
      else
        extra="${base_extra} --activation ${act} ${gating_args}"
        if script_uses_random_sign_beta "$script"; then
          extra="${extra} --beta 0.05"
          tag="${tag_prefix}_${act}_${gate_tag}_randsign"
        else
          tag="${tag_prefix}_${act}_${gate_tag}"
        fi
        maybe_add_task "$script" "$extra" "$tag"
      fi
    done
  done
}

# Build task list: each entry is "script|extra_args|log_tag"
TASKS=()
SKIPPED=0
for script in "${ABLATION_SCRIPTS[@]}"; do
  if [ ! -f "$script" ]; then
    echo "ERROR: missing $EXP_DIR/$script"
    exit 1
  fi
  for lr_idx in "${LR_CONFIG_IDXS[@]}"; do
    for nlr in "${NEURAL_LR_LIST[@]}"; do
      add_variant_tasks "$script" "--lr-config-idx ${lr_idx} --neural-lr-start ${nlr}" "lr${lr_idx}_nlr${nlr}"
    done
  done
done

NUM_TASKS=${#TASKS[@]}
LOG_ROOT="${LOG_ROOT:-$EXP_DIR/logs_tpu}"
mkdir -p "$LOG_ROOT"

if [ "$NUM_TASKS" -eq 0 ]; then
  echo "Nothing to run — all $SKIPPED configs already complete under $RESULTS_ROOT"
  exit 0
fi

gating_banner=""
for gated in "${GATING_LIST[@]}"; do
  gating_banner="${gating_banner} $(gate_tag_for "$gated")"
done
beta_banner="$BETA_MODE"
if [ "$BETA_MODE" = "positive" ]; then
  beta_banner="${BETA_MODE} (${POSITIVE_BETA})"
elif [ "$BETA_MODE" = "positive-sweep" ]; then
  beta_banner="${BETA_MODE} (${POSITIVE_BETAS[*]})"
fi

echo "=================================================================="
echo "$DATASET predictive-dendrite TPU sweep"
echo "  exp dir:    $EXP_DIR"
echo "  results:    $RESULTS_ROOT"
echo "  device:     PJRT_DEVICE=$PJRT_DEVICE  NUM_CHIPS=$NUM_CHIPS"
echo "  python:     $PYTHON_EXEC"
echo "  activation: ${ACTIVATION_LIST[*]}"
echo "  gating:    ${gating_banner}"
echo "  beta:       $beta_banner"
echo "  PE mix:     gamma=0.5/0.5"
echo "  models:     ${ABLATION_SCRIPTS[*]}"
echo "  grid:       LR_CONFIG_IDXS=(${LR_CONFIG_IDXS[*]})  NEURAL_LR_LIST=(${NEURAL_LR_LIST[*]})"
echo "  skipped:    $SKIPPED already-complete configs (SKIP_COMPLETED=$SKIP_COMPLETED)"
echo "  remaining:  $NUM_TASKS  (up to $NUM_CHIPS in parallel)"
echo "  logs:       $LOG_ROOT"
echo "=================================================================="

run_one_task() {
  local task_id="$1"
  local chip_id="$2"
  local script="$3"
  local extra_args="$4"
  local log_file="$5"

  export TPU_VISIBLE_CHIPS="$chip_id"
  export TPU_CHIPS_PER_PROCESS_BOUNDS=1,1,1
  export TPU_PROCESS_BOUNDS=1,1,1

  {
    echo "--------------------------------------------------------------------"
    echo "task=$task_id chip=$chip_id host=$(hostname) date=$(date -Is)"
    echo "$PYTHON_EXEC $script $extra_args"
    echo "--------------------------------------------------------------------"
    # shellcheck disable=SC2086
    "$PYTHON_EXEC" "$script" $extra_args
    local rc=$?
    echo "DONE task=$task_id exit=$rc date=$(date -Is)"
    return "$rc"
  } >"$log_file" 2>&1
}

FREE_CHIPS=()
for (( c=0; c<NUM_CHIPS; c++ )); do
  FREE_CHIPS+=("$c")
done

declare -A JOB_META=()
FAIL=0
RUNNING=0

reap_finished() {
  local pid task_chip task_id chip_id
  for pid in "${!JOB_META[@]}"; do
    if ! kill -0 "$pid" 2>/dev/null; then
      task_chip="${JOB_META[$pid]}"
      task_id="${task_chip%%:*}"
      chip_id="${task_chip##*:}"
      if wait "$pid"; then
        echo "finished task=$task_id chip=$chip_id"
      else
        echo "FAILED task=$task_id chip=$chip_id (see $LOG_ROOT)"
        FAIL=1
      fi
      FREE_CHIPS+=("$chip_id")
      unset "JOB_META[$pid]"
      RUNNING=$(( RUNNING - 1 ))
    fi
  done
}

for (( task_id=0; task_id<NUM_TASKS; task_id++ )); do
  while [ "$RUNNING" -ge "$NUM_CHIPS" ] || [ "${#FREE_CHIPS[@]}" -eq 0 ]; do
    reap_finished
    if [ "$RUNNING" -ge "$NUM_CHIPS" ] || [ "${#FREE_CHIPS[@]}" -eq 0 ]; then
      sleep 5
    fi
  done

  entry="${TASKS[$task_id]}"
  script="${entry%%|*}"
  rest="${entry#*|}"
  extra_args="${rest%%|*}"
  tag="${rest##*|}"

  script_base="${script%.py}"
  mkdir -p "$LOG_ROOT/$script_base"
  log_file=$(printf "%s/%s/task_%02d_%s.log" "$LOG_ROOT" "$script_base" "$task_id" "$tag")

  chip_id="${FREE_CHIPS[0]}"
  FREE_CHIPS=("${FREE_CHIPS[@]:1}")

  echo "launch task=$task_id chip=$chip_id $script $extra_args -> $log_file"
  run_one_task "$task_id" "$chip_id" "$script" "$extra_args" "$log_file" &
  pid=$!
  JOB_META[$pid]="$task_id:$chip_id"
  RUNNING=$(( RUNNING + 1 ))
done

while [ "$RUNNING" -gt 0 ]; do
  reap_finished
  if [ "$RUNNING" -gt 0 ]; then
    sleep 5
  fi
done

echo "=================================================================="
if [ "$FAIL" -ne 0 ]; then
  echo "Sweep finished WITH FAILURES. Inspect $LOG_ROOT"
  exit 1
fi
echo "Sweep finished OK. Logs in $LOG_ROOT"
echo "Results under ../Results/$DATASET/<ALGO>/"
