import sys
sys.path.append("../../src")

import os
import json
import argparse
from datetime import datetime

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import numpy as np
from tqdm import tqdm

from torch_utils import set_all_seeds, get_device, optimizer_step


class FixedInputWeightsBPCIFAR10MLP(nn.Module):
    """
    Fixed-input-weights backpropagation baseline for CIFAR-10 with architecture [3072, 1000, 10].

    The first layer is fixed after initialization. Only the output layer is trained.
    This tests whether the model can do better than a classifier trained on fixed
    random hidden features.
    """
    def __init__(self, architecture=(3072, 1000, 10), hidden_activation="relu"):
        super().__init__()

        self.architecture = list(architecture)
        self.fc1 = nn.Linear(self.architecture[0], self.architecture[1])
        self.fc2 = nn.Linear(self.architecture[1], self.architecture[2])
        self.hidden_activation = hidden_activation

        self.freeze_hidden_layer()

    def freeze_hidden_layer(self):
        self.fc1.weight.requires_grad_(False)
        if self.fc1.bias is not None:
            self.fc1.bias.requires_grad_(False)

    def hidden_nonlinearity(self, z):
        if self.hidden_activation == "relu":
            return F.relu(z)
        elif self.hidden_activation == "hard_sigmoid":
            return 0.5 * (1.0 + F.hardtanh(2.0 * z - 1.0))
        elif self.hidden_activation == "hardtanh":
            return torch.clamp(z, -1.0, 1.0)
        else:
            raise ValueError(f"Unknown hidden_activation={self.hidden_activation}")

    def forward(self, x):
        if x.dim() == 4:
            x = x.view(x.size(0), -1)
        h = self.hidden_nonlinearity(self.fc1(x))
        logits = self.fc2(h)
        return logits


@torch.no_grad()
def evaluate_fixed_input_weights_bp(model, loader, device):
    model.eval()

    correct = 0
    total = 0
    loss_sum = 0.0

    for x, y in loader:
        x = x.to(device)
        y = y.to(device)

        logits = model(x)
        loss = F.cross_entropy(logits, y, reduction="sum")

        pred = torch.argmax(logits, dim=1)
        correct += (pred == y).sum().item()
        total += y.numel()
        loss_sum += loss.item()

    return {
        "acc": correct / total,
        "loss": loss_sum / total,
    }


def train_fixed_input_weights_bp_one_seed(
    seed,
    n_epochs=15,
    architecture=(3072, 1000, 10),
    batch_size=20,
    hidden_activation="relu",
    optimizer_name="adam",
    lr=1e-3,
    weight_decay=0.0,
    data_root="data",
    n_validation=5000,
    validation_split_seed=0,
    device=None,
):
    set_all_seeds(seed)

    if device is None:
        device = get_device()

    transform = torchvision.transforms.Compose([
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize(
            mean=(0.4914, 0.4822, 0.4465),
            std=(3 * 0.2023, 3 * 0.1994, 3 * 0.2010),
        ),
    ])

    dset_train = torchvision.datasets.CIFAR10(
        data_root,
        train=True,
        transform=transform,
        target_transform=None,
        download=True,
    )

    dset_test = torchvision.datasets.CIFAR10(
        data_root,
        train=False,
        transform=transform,
        target_transform=None,
        download=True,
    )

    # Train/validation split with a FIXED seed, independent of the trial seed,
    # so every algorithm, learning rate, and trial sees the exact same three
    # sets (train / validation / test). Validation is used for model selection
    # (learning-rate picking); the official test set is never used for tuning.
    n_train = len(dset_train) - n_validation
    split_generator = torch.Generator().manual_seed(validation_split_seed)
    train_subset, val_subset = torch.utils.data.random_split(
        dset_train, [n_train, n_validation], generator=split_generator
    )

    # Generator makes DataLoader shuffling seed-specific and reproducible.
    loader_generator = torch.Generator()
    loader_generator.manual_seed(seed)

    train_loader = torch.utils.data.DataLoader(
        train_subset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
        generator=loader_generator,
    )

    train_eval_loader = torch.utils.data.DataLoader(
        train_subset,
        batch_size=256,
        shuffle=False,
        num_workers=0,
    )

    val_loader = torch.utils.data.DataLoader(
        val_subset,
        batch_size=256,
        shuffle=False,
        num_workers=0,
    )

    test_loader = torch.utils.data.DataLoader(
        dset_test,
        batch_size=256,
        shuffle=False,
        num_workers=0,
    )

    model = FixedInputWeightsBPCIFAR10MLP(
        architecture=architecture,
        hidden_activation=hidden_activation,
    ).to(device)

    # Only train parameters with requires_grad=True, i.e. output layer only.
    trainable_params = [p for p in model.parameters() if p.requires_grad]

    if optimizer_name.lower() == "adam":
        optimizer = torch.optim.Adam(trainable_params, lr=lr, weight_decay=weight_decay)
    elif optimizer_name.lower() == "sgd":
        optimizer = torch.optim.SGD(trainable_params, lr=lr, momentum=0.9, weight_decay=weight_decay)
    else:
        raise ValueError(f"Unknown optimizer_name={optimizer_name}")

    trn_acc_list = []
    val_acc_list = []
    tst_acc_list = []
    trn_loss_list = []
    val_loss_list = []
    tst_loss_list = []

    trn0 = evaluate_fixed_input_weights_bp(model, train_eval_loader, device)
    val0 = evaluate_fixed_input_weights_bp(model, val_loader, device)
    tst0 = evaluate_fixed_input_weights_bp(model, test_loader, device)

    trn_acc_list.append(trn0["acc"])
    val_acc_list.append(val0["acc"])
    tst_acc_list.append(tst0["acc"])
    trn_loss_list.append(trn0["loss"])
    val_loss_list.append(val0["loss"])
    tst_loss_list.append(tst0["loss"])

    print(
        f"Fixed-Input-Weights BP CIFAR10 seed {seed}, epoch 0: "
        f"train acc={trn0['acc']:.4f}, val acc={val0['acc']:.4f}, "
        f"test acc={tst0['acc']:.4f}"
    )

    for epoch in range(n_epochs):
        model.train()

        pbar = tqdm(
            train_loader,
            desc=f"Fixed-Input-Weights BP CIFAR10, seed={seed}, epoch {epoch+1}/{n_epochs}"
        )

        for x, y in pbar:
            x = x.to(device)
            y = y.to(device)

            optimizer.zero_grad(set_to_none=True)

            logits = model(x)
            loss = F.cross_entropy(logits, y)

            loss.backward()
            optimizer_step(optimizer)

            pbar.set_postfix({"batch_loss": loss.item()})

        trn = evaluate_fixed_input_weights_bp(model, train_eval_loader, device)
        val = evaluate_fixed_input_weights_bp(model, val_loader, device)
        tst = evaluate_fixed_input_weights_bp(model, test_loader, device)

        trn_acc_list.append(trn["acc"])
        val_acc_list.append(val["acc"])
        tst_acc_list.append(tst["acc"])
        trn_loss_list.append(trn["loss"])
        val_loss_list.append(val["loss"])
        tst_loss_list.append(tst["loss"])

        print(
            f"Fixed-Input-Weights BP CIFAR10 seed {seed}, epoch {epoch+1}: "
            f"train acc={trn['acc']:.4f}, val acc={val['acc']:.4f}, "
            f"test acc={tst['acc']:.4f}, "
            f"train loss={trn['loss']:.4f}, val loss={val['loss']:.4f}, "
            f"test loss={tst['loss']:.4f}"
        )

    return {
        "seed": seed,
        "model": model,
        "trn_acc": trn_acc_list,
        "val_acc": val_acc_list,
        "test_acc": tst_acc_list,
        "trn_loss": trn_loss_list,
        "val_loss": val_loss_list,
        "test_loss": tst_loss_list,
        "n_trainable_params": sum(p.numel() for p in model.parameters() if p.requires_grad),
        "n_frozen_params": sum(p.numel() for p in model.parameters() if not p.requires_grad),
    }


# ------------------------------------------------------------------
# Command-line arguments
# ------------------------------------------------------------------
parser = argparse.ArgumentParser(description="Fixed-input-weights BP (Adam) CIFAR10 learning-rate sweep.")
parser.add_argument("--lr", type=float, default=1e-4, help="Optimizer learning rate.")
parser.add_argument(
    "--activation", type=str, default="relu",
    choices=["relu", "hard_sigmoid"],
    help="Hidden-layer activation.",
)
args = parser.parse_args()

# ------------------------------------------------------------------
# Experiment configuration
# ------------------------------------------------------------------
n_trials = 5
seed_list = [10 * j for j in range(n_trials)]

dataset_name = "CIFAR10"
architecture = [3072, 1000, 10]
n_epochs = 15
batch_size = 20

hidden_activation = args.activation   # relu, hard_sigmoid, or hardtanh (see --activation)
optimizer_name = "adam"         # "adam" or "sgd"
lr = args.lr
weight_decay = 0.0
data_root = "data"

normalization_mean = [0.4914, 0.4822, 0.4465]
normalization_std = [3 * 0.2023, 3 * 0.1994, 3 * 0.2010]

# Train/validation/test split: 50k train -> 45k train / 5k validation, plus the
# official 10k test set. The split seed is FIXED (not a trial seed) so all
# algorithms and trials use identical sets; validation is for LR selection.
n_validation = 5000
validation_split_seed = 0

device = get_device()
print("Device:", device)

# ------------------------------------------------------------------
# Matched, unique save paths
# ------------------------------------------------------------------
run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
arch_str = "x".join(map(str, architecture))
lr_str = f"{lr:.0e}"
wd_str = f"{weight_decay:.0e}"

experiment_name = (
    f"FixedInputWeightsBP_{dataset_name}_{hidden_activation}_{optimizer_name}"
    f"_lr{lr_str}_wd{wd_str}_bs{batch_size}_ep{n_epochs}_arch{arch_str}"
)

results_dir = "../Results/CIFAR10/FixedInputWeightsBP_Adam"
os.makedirs(results_dir, exist_ok=True)

save_prefix = f"{results_dir}/{experiment_name}_{run_id}"
save_path = f"{save_prefix}_accuracy_arrays.npz"
hyperparam_path = f"{save_prefix}_hyperparams.json"

print("Saving arrays to:", save_path)
print("Saving hyperparameters to:", hyperparam_path)

# ------------------------------------------------------------------
# Save hyperparameters once
# ------------------------------------------------------------------
hyperparams = {
    "run_id": run_id,
    "experiment_name": experiment_name,
    "method": "fixed_input_weights_backprop_output_layer_only",
    "description": "Backpropagation baseline with hidden layer fixed; only output layer trained.",
    "dataset": dataset_name,
    "n_trials": n_trials,
    "seed_list": seed_list,
    "architecture": architecture,
    "n_epochs": n_epochs,
    "batch_size": batch_size,
    "hidden_activation": hidden_activation,
    "optimizer_name": optimizer_name,
    "lr": lr,
    "weight_decay": weight_decay,
    "data_root": data_root,
    "n_validation": n_validation,
    "n_train_after_split": 50000 - n_validation,
    "validation_split_seed": validation_split_seed,
    "device": str(device),
    "torch_version": torch.__version__,
    "torchvision_version": torchvision.__version__,
    "initial_evaluation": True,
    "train_eval_batch_size": 256,
    "val_eval_batch_size": 256,
    "test_eval_batch_size": 256,
    "input_shape": [3, 32, 32],
    "input_dimension": 3072,
    "frozen_layers": ["fc1.weight", "fc1.bias"],
    "trained_layers": ["fc2.weight", "fc2.bias"],
}

with open(hyperparam_path, "w") as f:
    json.dump(hyperparams, f, indent=2)

# ------------------------------------------------------------------
# Run trials
# ------------------------------------------------------------------
fixed_input_weights_bp_results = []

fixed_input_weights_bp_trn_acc_list_of_list = []
fixed_input_weights_bp_val_acc_list_of_list = []
fixed_input_weights_bp_test_acc_list_of_list = []
fixed_input_weights_bp_trn_loss_list_of_list = []
fixed_input_weights_bp_val_loss_list_of_list = []
fixed_input_weights_bp_test_loss_list_of_list = []
n_trainable_params_list = []
n_frozen_params_list = []

for trial_, seed in enumerate(seed_list):
    print("\n" + "=" * 80)
    print(f"Fixed-Input-Weights BP CIFAR10 trial {trial_ + 1}/{n_trials}, seed={seed}")
    print("=" * 80)

    result = train_fixed_input_weights_bp_one_seed(
        seed=seed,
        n_epochs=n_epochs,
        architecture=architecture,
        batch_size=batch_size,
        hidden_activation=hidden_activation,
        optimizer_name=optimizer_name,
        lr=lr,
        weight_decay=weight_decay,
        data_root=data_root,
        n_validation=n_validation,
        validation_split_seed=validation_split_seed,
        device=device,
    )

    fixed_input_weights_bp_results.append(result)

    fixed_input_weights_bp_trn_acc_list_of_list.append(result["trn_acc"])
    fixed_input_weights_bp_val_acc_list_of_list.append(result["val_acc"])
    fixed_input_weights_bp_test_acc_list_of_list.append(result["test_acc"])
    fixed_input_weights_bp_trn_loss_list_of_list.append(result["trn_loss"])
    fixed_input_weights_bp_val_loss_list_of_list.append(result["val_loss"])
    fixed_input_weights_bp_test_loss_list_of_list.append(result["test_loss"])
    n_trainable_params_list.append(result["n_trainable_params"])
    n_frozen_params_list.append(result["n_frozen_params"])

    # Save after each trial for safety.
    np.savez_compressed(
        save_path,
        seed_list=np.asarray(seed_list[:len(fixed_input_weights_bp_trn_acc_list_of_list)]),
        architecture=np.asarray(architecture),
        n_epochs=np.asarray(n_epochs),
        batch_size=np.asarray(batch_size),
        fixed_input_weights_bp_trn_acc=np.asarray(fixed_input_weights_bp_trn_acc_list_of_list, dtype=float),
        fixed_input_weights_bp_val_acc=np.asarray(fixed_input_weights_bp_val_acc_list_of_list, dtype=float),
        fixed_input_weights_bp_test_acc=np.asarray(fixed_input_weights_bp_test_acc_list_of_list, dtype=float),
        fixed_input_weights_bp_trn_loss=np.asarray(fixed_input_weights_bp_trn_loss_list_of_list, dtype=float),
        fixed_input_weights_bp_val_loss=np.asarray(fixed_input_weights_bp_val_loss_list_of_list, dtype=float),
        fixed_input_weights_bp_test_loss=np.asarray(fixed_input_weights_bp_test_loss_list_of_list, dtype=float),
        n_trainable_params=np.asarray(n_trainable_params_list, dtype=int),
        n_frozen_params=np.asarray(n_frozen_params_list, dtype=int),
    )

print(f"Saved Fixed-Input-Weights BP CIFAR10 arrays to: {save_path}")
print(f"Saved Fixed-Input-Weights BP CIFAR10 hyperparameters to: {hyperparam_path}")
print("fixed_input_weights_bp_trn_acc shape :", np.asarray(fixed_input_weights_bp_trn_acc_list_of_list).shape)
print("fixed_input_weights_bp_val_acc shape :", np.asarray(fixed_input_weights_bp_val_acc_list_of_list).shape)
print("fixed_input_weights_bp_test_acc shape:", np.asarray(fixed_input_weights_bp_test_acc_list_of_list).shape)
print("Trainable params:", n_trainable_params_list)
print("Frozen params   :", n_frozen_params_list)