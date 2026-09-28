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
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

from torch_utils import set_all_seeds, get_device, optimizer_step

class BPMNISTMLP(nn.Module):
    """
    Backpropagation baseline with architecture [784, 500, 10].

    Default hidden activation is ReLU. If you want closer activation compatibility
    with your PEM runs, set hidden_activation="hard_sigmoid".
    """
    def __init__(self, architecture=(784, 500, 10), hidden_activation="relu"):
        super().__init__()

        self.architecture = list(architecture)
        self.fc1 = nn.Linear(self.architecture[0], self.architecture[1])
        self.fc2 = nn.Linear(self.architecture[1], self.architecture[2])
        self.hidden_activation = hidden_activation

    def hidden_nonlinearity(self, z):
        if self.hidden_activation == "relu":
            return F.relu(z)
        elif self.hidden_activation == "hard_sigmoid":
            # compatible with common hard-sigmoid style in [0, 1]
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
def evaluate_bp(model, loader, device):
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

def train_bp_one_seed(
    seed,
    n_epochs=15,
    architecture=(784, 500, 10),
    batch_size=20,
    hidden_activation="relu",
    optimizer_name="adam",
    lr=1e-3,
    weight_decay=0.0,
    data_root="data",
    n_validation=10000,
    validation_split_seed=0,
    device=None,
):
    set_all_seeds(seed)

    if device is None:
        device = get_device()

    transform = torchvision.transforms.Compose([
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize(mean=(0.0,), std=(1.0,))
    ])

    mnist_dset_train = torchvision.datasets.MNIST(
        data_root,
        train=True,
        transform=transform,
        target_transform=None,
        download=True,
    )

    mnist_dset_test = torchvision.datasets.MNIST(
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
    n_train = len(mnist_dset_train) - n_validation
    split_generator = torch.Generator().manual_seed(validation_split_seed)
    train_subset, val_subset = torch.utils.data.random_split(
        mnist_dset_train, [n_train, n_validation], generator=split_generator
    )

    train_loader = torch.utils.data.DataLoader(
        train_subset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
    )

    # Separate non-shuffled train loader for stable evaluation.
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
        mnist_dset_test,
        batch_size=256,
        shuffle=False,
        num_workers=0,
    )

    model = BPMNISTMLP(
        architecture=architecture,
        hidden_activation=hidden_activation,
    ).to(device)

    if optimizer_name.lower() == "adam":
        optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    elif optimizer_name.lower() == "sgd":
        optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=weight_decay)
    else:
        raise ValueError(f"Unknown optimizer_name={optimizer_name}")

    trn_acc_list = []
    val_acc_list = []
    tst_acc_list = []
    trn_loss_list = []
    val_loss_list = []
    tst_loss_list = []

    # Initial evaluation before training, matching PEM convention.
    trn0 = evaluate_bp(model, train_eval_loader, device)
    val0 = evaluate_bp(model, val_loader, device)
    tst0 = evaluate_bp(model, test_loader, device)

    trn_acc_list.append(trn0["acc"])
    val_acc_list.append(val0["acc"])
    tst_acc_list.append(tst0["acc"])
    trn_loss_list.append(trn0["loss"])
    val_loss_list.append(val0["loss"])
    tst_loss_list.append(tst0["loss"])

    print(
        f"Seed {seed}, epoch 0: "
        f"train acc={trn0['acc']:.4f}, val acc={val0['acc']:.4f}, "
        f"test acc={tst0['acc']:.4f}"
    )

    for epoch in range(n_epochs):
        model.train()

        pbar = tqdm(train_loader, desc=f"BP seed={seed}, epoch {epoch+1}/{n_epochs}")

        for x, y in pbar:
            x = x.to(device)
            y = y.to(device)

            optimizer.zero_grad(set_to_none=True)

            logits = model(x)
            loss = F.cross_entropy(logits, y)

            loss.backward()
            optimizer_step(optimizer)

            pbar.set_postfix({"batch_loss": loss.item()})

        trn = evaluate_bp(model, train_eval_loader, device)
        val = evaluate_bp(model, val_loader, device)
        tst = evaluate_bp(model, test_loader, device)

        trn_acc_list.append(trn["acc"])
        val_acc_list.append(val["acc"])
        tst_acc_list.append(tst["acc"])
        trn_loss_list.append(trn["loss"])
        val_loss_list.append(val["loss"])
        tst_loss_list.append(tst["loss"])

        print(
            f"Seed {seed}, epoch {epoch+1}: "
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
    }

# ------------------------------------------------------------------
# Command-line arguments
# ------------------------------------------------------------------
parser = argparse.ArgumentParser(description="BP (SGD) MNIST learning-rate sweep.")
parser.add_argument("--lr", type=float, default=1e-3, help="Optimizer learning rate.")
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

architecture = [784, 500, 10]
n_epochs = 15
batch_size = 20

hidden_activation = args.activation   # relu, hard_sigmoid, or hardtanh (see --activation)
optimizer_name = "sgd"         # "adam" or "sgd"
lr = args.lr
weight_decay = 0.0
data_root = "data"

# Train/validation/test split: 60k train -> 50k train / 10k validation, plus the
# official 10k test set. The split seed is FIXED (not a trial seed) so all
# algorithms and trials use identical sets; validation is for LR selection.
n_validation = 10000
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
    f"BP_MNIST_{hidden_activation}_{optimizer_name}"
    f"_lr{lr_str}_wd{wd_str}_bs{batch_size}_ep{n_epochs}_arch{arch_str}"
)

results_dir = "../Results/MNIST/BP_SGD"
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
    "method": "backpropagation",
    "dataset": "MNIST",
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
    "n_train_after_split": 60000 - n_validation,
    "validation_split_seed": validation_split_seed,
    "device": str(device),
    "torch_version": torch.__version__,
    "initial_evaluation": True,
    "train_eval_batch_size": 256,
    "val_eval_batch_size": 256,
    "test_eval_batch_size": 256,
}

with open(hyperparam_path, "w") as f:
    json.dump(hyperparams, f, indent=2)

# ------------------------------------------------------------------
# Run trials
# ------------------------------------------------------------------
bp_results = []

bp_trn_acc_list_of_list = []
bp_val_acc_list_of_list = []
bp_test_acc_list_of_list = []
bp_trn_loss_list_of_list = []
bp_val_loss_list_of_list = []
bp_test_loss_list_of_list = []

for trial_, seed in enumerate(seed_list):
    print("\n" + "=" * 80)
    print(f"BP trial {trial_ + 1}/{n_trials}, seed={seed}")
    print("=" * 80)

    result = train_bp_one_seed(
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

    bp_results.append(result)

    bp_trn_acc_list_of_list.append(result["trn_acc"])
    bp_val_acc_list_of_list.append(result["val_acc"])
    bp_test_acc_list_of_list.append(result["test_acc"])
    bp_trn_loss_list_of_list.append(result["trn_loss"])
    bp_val_loss_list_of_list.append(result["val_loss"])
    bp_test_loss_list_of_list.append(result["test_loss"])

    # Save after each trial for safety.
    np.savez_compressed(
        save_path,
        seed_list=np.asarray(seed_list[:len(bp_trn_acc_list_of_list)]),
        architecture=np.asarray(architecture),
        n_epochs=np.asarray(n_epochs),
        batch_size=np.asarray(batch_size),
        bp_trn_acc=np.asarray(bp_trn_acc_list_of_list, dtype=float),
        bp_val_acc=np.asarray(bp_val_acc_list_of_list, dtype=float),
        bp_test_acc=np.asarray(bp_test_acc_list_of_list, dtype=float),
        bp_trn_loss=np.asarray(bp_trn_loss_list_of_list, dtype=float),
        bp_val_loss=np.asarray(bp_val_loss_list_of_list, dtype=float),
        bp_test_loss=np.asarray(bp_test_loss_list_of_list, dtype=float),
    )

print(f"Saved BP arrays to: {save_path}")
print(f"Saved BP hyperparameters to: {hyperparam_path}")
print("bp_trn_acc shape :", np.asarray(bp_trn_acc_list_of_list).shape)
print("bp_val_acc shape :", np.asarray(bp_val_acc_list_of_list).shape)
print("bp_test_acc shape:", np.asarray(bp_test_acc_list_of_list).shape)