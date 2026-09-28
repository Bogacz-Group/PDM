#!/usr/bin/env python
# coding: utf-8

import sys
sys.path.append("../../src")

from datetime import datetime
import os
import json
import argparse

import torch
import torch.nn.functional as F
import torchvision
import numpy as np
from tqdm import tqdm

from torch_utils import set_all_seeds, get_device
from activations import get_activation, ACTIVATIONS
from evaluate import evaluateContrastivePredictiveErrorPropagationFullClamp
from ContrastivePredictiveErrorPropagationFullClamp import ContrastivePredictiveErrorPropagationFullClamp


# ---------------------------------------------------------
# Command-line arguments (learning-rate / neural-lr ablation)
# ---------------------------------------------------------
# The synaptic learning rate is a per-layer dict of arrays (the first feedback
# entry is NaN by design and is not used in the fully-clamped variant), so we
# select it by index from a small set of hand-tuned configurations. These are
# kept CLOSE to the CIFAR-10 values known to train well -- moving far from them
# tends to break this algorithm.
LR_START_CONFIGS = [
    {"label": "low",  "ff": [0.02, 0.01],   "fb": [float("nan"), 0.006]},
    {"label": "base", "ff": [0.03, 0.015],  "fb": [float("nan"), 0.01]},   # known-good defaults
    {"label": "high", "ff": [0.045, 0.022], "fb": [float("nan"), 0.015]},
]

parser = argparse.ArgumentParser(
    description="Fully-Clamped Predictive Error Propagation (CIFAR-10): "
                "synaptic-LR / neural-LR ablation."
)
parser.add_argument(
    "--lr-config-idx", "--lr_config_idx", type=int, default=1,
    choices=range(len(LR_START_CONFIGS)),
    help="Index into LR_START_CONFIGS (0=low, 1=base, 2=high).",
)
parser.add_argument(
    "--neural-lr-start", "--neural_lr_start", type=float, default=1.0,
    help="Initial neural-dynamics learning rate (sweep around 1.0).",
)
parser.add_argument(
    "--activation", type=str, default="hard_sigmoid",
    choices=sorted(ACTIVATIONS),
    help="Neuron activation used in the neural dynamics (e.g. hard_sigmoid, relu).",
)
parser.add_argument(
    "--gating", action="store_true",
    help="Gate each phase's prediction error by g = f'(x) at that phase's post-synaptic state.",
)
args = parser.parse_args()

LR_CONFIG = LR_START_CONFIGS[args.lr_config_idx]
NEURAL_LR_START = args.neural_lr_start
ACTIVATION_NAME = args.activation
USE_GATING = args.gating
print(f"LR config [{args.lr_config_idx}] '{LR_CONFIG['label']}': "
      f"ff={LR_CONFIG['ff']}, fb={LR_CONFIG['fb']} | neural_lr_start={NEURAL_LR_START}")


# ---------------------------------------------------------
# Device
# ---------------------------------------------------------
device = get_device()
print("Device:", device)


# ---------------------------------------------------------
# Data
# ---------------------------------------------------------
normalization_mean = [0.4914, 0.4822, 0.4465]
normalization_std = [3 * 0.2023, 3 * 0.1994, 3 * 0.2010]

transform = torchvision.transforms.Compose([
    torchvision.transforms.ToTensor(),
    torchvision.transforms.Normalize(
        mean=tuple(normalization_mean),
        std=tuple(normalization_std),
    ),
])

cifar_dset_train = torchvision.datasets.CIFAR10(
    "../../data",
    train=True,
    transform=transform,
    target_transform=None,
    download=True,
)

cifar_dset_test = torchvision.datasets.CIFAR10(
    "../../data",
    train=False,
    transform=transform,
    target_transform=None,
    download=True,
)

# Train/validation/test split: 50k train -> 45k train / 5k validation, plus the
# official 10k test set. The split seed is FIXED (not a trial seed) and matches the
# BP scripts, so every algorithm/LR/trial sees the exact same three sets. Validation
# is used for model selection (LR picking); the test set is never used for tuning.
n_validation = 5000
validation_split_seed = 0
n_train = len(cifar_dset_train) - n_validation
split_generator = torch.Generator().manual_seed(validation_split_seed)
train_subset, val_subset = torch.utils.data.random_split(
    cifar_dset_train, [n_train, n_validation], generator=split_generator
)

train_loader = torch.utils.data.DataLoader(
    train_subset,
    batch_size=20,
    shuffle=True,
    num_workers=0,
)

val_loader = torch.utils.data.DataLoader(
    val_subset,
    batch_size=20,
    shuffle=False,
    num_workers=0,
)

test_loader = torch.utils.data.DataLoader(
    cifar_dset_test,
    batch_size=20,
    shuffle=False,
    num_workers=0,
)


# ---------------------------------------------------------
# Save paths
# ---------------------------------------------------------
results_dir = "../Results/CIFAR10/CPEP_FullClamp"
os.makedirs(results_dir, exist_ok=True)

run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
nlr_str = f"{NEURAL_LR_START:g}"
act_tag = "" if ACTIVATION_NAME == "hard_sigmoid" else f"_{ACTIVATION_NAME}"
gate_tag = "_gated" if USE_GATING else ""
experiment_name = (
    f"CPEP_FullClamp_CIFAR10_lr{LR_CONFIG['label']}_nlr{nlr_str}_bs20_ep15_arch3072x1000x10{act_tag}{gate_tag}"
)

save_path = f"{results_dir}/{experiment_name}_{run_id}_accuracy_arrays.npz"
hyperparam_path = f"{results_dir}/{experiment_name}_{run_id}_hyperparams.json"

print("Saving results to:", save_path)
print("Saving hyperparameters to:", hyperparam_path)


# ---------------------------------------------------------
# Experiment settings
# ---------------------------------------------------------
n_trials = 5
seed_list = [10 * j for j in range(n_trials)]

trn_acc_list_of_list = []
val_acc_list_of_list = []
tst_acc_list_of_list = []


# ---------------------------------------------------------
# Trials
# ---------------------------------------------------------
for trial_ in range(n_trials):

    seed = seed_list[trial_]
    set_all_seeds(seed)

    activation = get_activation(ACTIVATION_NAME)
    architecture = [32 * 32 * 3, 1000, 10]

    lambda_ = 1.0
    epsilon = 0.01
    gamma_forward = 0.5
    gamma_backward = 0.5

    lr_start = {
        "ff": np.array(LR_CONFIG["ff"], dtype=float),
        "fb": np.array(LR_CONFIG["fb"], dtype=float),
    }

    neural_lr_start = NEURAL_LR_START
    neural_lr_stop = 0.02
    neural_lr_rule = "constant"
    neural_lr_decay_multiplier = 0.01
    neural_dynamic_iterations_nudged = 10
    neural_dynamic_iterations_free = 30

    weight_decay = False
    n_epochs = 15

    model = ContrastivePredictiveErrorPropagationFullClamp(
        architecture=architecture,
        lambda_=lambda_,
        epsilon=epsilon,
        gamma_forward=gamma_forward,
        gamma_backward=gamma_backward,
        activation=activation,
        use_gating=USE_GATING,
        device=device,
    )

    train_acc0 = evaluateContrastivePredictiveErrorPropagationFullClamp(
        model,
        train_loader,
        neural_lr_start,
        neural_lr_stop,
        neural_lr_rule,
        neural_lr_decay_multiplier,
        neural_dynamic_iterations_free,
        device,
    )

    val_acc0 = evaluateContrastivePredictiveErrorPropagationFullClamp(
        model,
        val_loader,
        neural_lr_start,
        neural_lr_stop,
        neural_lr_rule,
        neural_lr_decay_multiplier,
        neural_dynamic_iterations_free,
        device,
    )

    test_acc0 = evaluateContrastivePredictiveErrorPropagationFullClamp(
        model,
        test_loader,
        neural_lr_start,
        neural_lr_stop,
        neural_lr_rule,
        neural_lr_decay_multiplier,
        neural_dynamic_iterations_free,
        device,
    )

    trn_acc_list = [train_acc0]
    val_acc_list = [val_acc0]
    tst_acc_list = [test_acc0]

    if trial_ == 0:
        hyperparams = {
            "experiment_name": experiment_name,
            "run_id": run_id,
            "method": "ContrastivePredictiveErrorPropagationFullClamp",
            "dataset": "CIFAR10",
            "lr_config_idx": args.lr_config_idx,
            "lr_config_label": LR_CONFIG["label"],
            "n_validation": n_validation,
            "n_train_after_split": n_train,
            "validation_split_seed": validation_split_seed,
            "n_trials": n_trials,
            "seed_list": seed_list,
            "activation": ACTIVATION_NAME,
            "use_gating": USE_GATING,
            "architecture": architecture,
            "lambda_": lambda_,
            "epsilon": epsilon,
            "gamma_forward": gamma_forward,
            "gamma_backward": gamma_backward,
            "lr_start_ff": lr_start["ff"].tolist(),
            "lr_start_fb": [
                None if np.isnan(x) else float(x)
                for x in lr_start["fb"]
            ],
            "neural_lr_start": neural_lr_start,
            "neural_lr_stop": neural_lr_stop,
            "neural_lr_rule": neural_lr_rule,
            "neural_lr_decay_multiplier": neural_lr_decay_multiplier,
            "neural_dynamic_iterations_nudged": neural_dynamic_iterations_nudged,
            "neural_dynamic_iterations_free": neural_dynamic_iterations_free,
            "weight_decay": weight_decay,
            "n_epochs": n_epochs,
            "batch_size": train_loader.batch_size,
            "normalization_mean": normalization_mean,
            "normalization_std": normalization_std,
            "device": str(device),
            "torch_version": torch.__version__,
            "torchvision_version": torchvision.__version__,
            "note": "Fully clamped nudged phase: output layer is clamped to the one-hot label instead of weak beta nudging.",
        }

        with open(hyperparam_path, "w") as f:
            json.dump(hyperparams, f, indent=2)

    for epoch_ in range(n_epochs):

        if epoch_ < 20:
            lr = {
                "ff": lr_start["ff"] * (0.95) ** epoch_,
                "fb": lr_start["fb"] * (0.95) ** epoch_,
            }
        else:
            lr = {
                "ff": lr_start["ff"] * (0.9) ** epoch_,
                "fb": lr_start["fb"] * (0.9) ** epoch_,
            }

        for idx, (x, y) in tqdm(enumerate(train_loader), total=len(train_loader)):

            x, y = x.to(device), y.to(device)
            x = x.view(x.size(0), -1).T
            y_one_hot = F.one_hot(y, 10).to(device).T.float()

            take_debug_logs_ = (idx % 500 == 0)

            neurons = model.batch_step_hopfield(
                x,
                y_one_hot,
                lr,
                neural_lr_start,
                neural_lr_stop,
                neural_lr_rule,
                neural_lr_decay_multiplier,
                neural_dynamic_iterations_free,
                neural_dynamic_iterations_nudged,
                take_debug_logs_,
                weight_decay,
            )

        trn_acc = evaluateContrastivePredictiveErrorPropagationFullClamp(
            model,
            train_loader,
            neural_lr_start,
            neural_lr_stop,
            neural_lr_rule,
            neural_lr_decay_multiplier,
            neural_dynamic_iterations_free,
            device,
            printing=False,
        )

        val_acc = evaluateContrastivePredictiveErrorPropagationFullClamp(
            model,
            val_loader,
            neural_lr_start,
            neural_lr_stop,
            neural_lr_rule,
            neural_lr_decay_multiplier,
            neural_dynamic_iterations_free,
            device,
            printing=False,
        )

        tst_acc = evaluateContrastivePredictiveErrorPropagationFullClamp(
            model,
            test_loader,
            neural_lr_start,
            neural_lr_stop,
            neural_lr_rule,
            neural_lr_decay_multiplier,
            neural_dynamic_iterations_free,
            device,
            printing=False,
        )

        trn_acc_list.append(trn_acc)
        val_acc_list.append(val_acc)
        tst_acc_list.append(tst_acc)

        print(
            "Trial : {}, Epoch : {}, Train Accuracy : {}, Val Accuracy : {}, Test Accuracy : {}".format(
                trial_ + 1,
                epoch_ + 1,
                trn_acc,
                val_acc,
                tst_acc,
            )
        )

    trn_acc_list_of_list.append(trn_acc_list)
    val_acc_list_of_list.append(val_acc_list)
    tst_acc_list_of_list.append(tst_acc_list)

    np.savez_compressed(
        save_path,
        seed_list=np.asarray(seed_list[:len(trn_acc_list_of_list)]),
        trn_acc=np.asarray(trn_acc_list_of_list, dtype=float),
        val_acc=np.asarray(val_acc_list_of_list, dtype=float),
        test_acc=np.asarray(tst_acc_list_of_list, dtype=float),
    )

print("Saved results to:", save_path)
print("Saved hyperparameters to:", hyperparam_path)
print("Train accuracy array shape:", np.asarray(trn_acc_list_of_list, dtype=float).shape)
print("Val accuracy array shape:", np.asarray(val_acc_list_of_list, dtype=float).shape)
print("Test accuracy array shape:", np.asarray(tst_acc_list_of_list, dtype=float).shape)