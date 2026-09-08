#!/usr/bin/env python3
"""Train firing-rate XOR models and write results/firing_rate.csv."""

from __future__ import annotations

import argparse
import random
from pathlib import Path

import numpy as np
import pandas as pd
import torch

SCRIPT_DIR = Path(__file__).resolve().parent
RESULTS_PATH = SCRIPT_DIR / "results" / "firing_rate.csv"
SEEDS = [55160, 4209, 59052, 24024, 33022, 51564, 28278, 15273]
HIDDEN_SIZE = 64
LR = 0.1
N_EPOCHS = 500


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def gate(x: torch.Tensor) -> torch.Tensor:
    return (x > 0).type_as(x)


class SimpleHebbianModel:
    def __init__(
        self,
        hidden_size: int,
        ws_lr: float,
        device: torch.device,
        *,
        clamp_weights: bool = False,
    ):
        self.device = device
        self.ws_lr = float(ws_lr)
        self.clamp_weights = bool(clamp_weights)
        self.batch_size = 4

        self.wf0 = torch.nn.Linear(2, hidden_size, bias=False, device=device)
        self.wf1 = torch.nn.Linear(hidden_size, 1, bias=False, device=device)

        self.hidden_size = hidden_size

    def _clamp_weights(self) -> None:
        with torch.no_grad():
            self.wf0.weight.data.clamp_(-1.0, 1.0)
            self.wf1.weight.data.clamp_(-1.0, 1.0)

    def _hebb_update_w(
        self, layer: torch.nn.Linear, pre: torch.Tensor, post: torch.Tensor
    ) -> None:
        dw = torch.bmm(pre.unsqueeze(2) - 1 / 2, post.unsqueeze(1) - 1 / 2).sum(dim=0).t()
        layer.weight.data.add_(dw * self.ws_lr)

    def forward(self, inputs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        u1 = self.wf0(inputs)
        x1 = u1 * gate(u1)
        x2 = self.wf1(x1)
        return x1, x2

    def predict(self, inputs: torch.Tensor) -> torch.Tensor:
        _, x2 = self.forward(inputs)
        return x2

    def learn(self, inputs: torch.Tensor, targets: torch.Tensor) -> None:
        x1, _ = self.forward(inputs)

        # Hebbian rule: each connection w_ij updated by pre_i * post_j.
        self._hebb_update_w(self.wf0, inputs, x1)
        self._hebb_update_w(self.wf1, x1, targets)
        if self.clamp_weights:
            self._clamp_weights()


class ThreeQXORModel:
    def __init__(self, hidden_size: int, ws_lr: float, device: torch.device):
        self.device = device
        self.ws_lr = float(ws_lr)
        self.vf_coeff = 0.5
        self.vb_coeff = 0.5
        self.num_inference_iterations = 100
        self.batch_size = 4

        self.wf0 = torch.nn.Linear(2, hidden_size, bias=False, device=device)
        self.wf1 = torch.nn.Linear(hidden_size, 1, bias=False, device=device)
        self.wb1 = torch.nn.Linear(hidden_size, 2, bias=False, device=device)
        self.wb2 = torch.nn.Linear(1, hidden_size, bias=False, device=device)

        self.hidden_size = hidden_size
        self.reset_state()

    def reset_state(self) -> None:
        self.x0 = torch.zeros(self.batch_size, 2, device=self.device)
        self.x1 = torch.zeros(
            self.batch_size,
            self.hidden_size,
            device=self.device,
        )
        self.x2 = torch.zeros(self.batch_size, 1, device=self.device)
        self.u1 = torch.zeros_like(self.x1)
        self.u2 = torch.zeros_like(self.x2)
        self.vf1 = torch.zeros_like(self.x1)
        self.vf2 = torch.zeros_like(self.x2)
        self.vb0 = torch.zeros_like(self.x0)
        self.vb1 = torch.zeros_like(self.x1)

    def update_vs(self) -> None:
        with torch.no_grad():
            self.vf1 = self.wf0(self.x0)
            self.vf2 = self.wf1(self.x1)
            self.vb1 = self.wb2(self.x2)
            self.vb0 = self.wb1(self.x1)

    def update_hidden(self) -> None:
        self.u1 = (
            self.vf_coeff * self.vf1 + self.vb_coeff * self.vb1
        ) / (self.vf_coeff + self.vb_coeff)
        self.x1 = self.u1 * gate(self.u1)

    def update_output(self) -> None:
        self.u2 = self.vf2
        self.x2 = self.u2

    def _update_w(
        self, layer: torch.nn.Linear, pre: torch.Tensor, post: torch.Tensor
    ) -> None:
        dw = torch.bmm(pre.unsqueeze(2), post.unsqueeze(1)).sum(dim=0).t()
        layer.weight.data.add_(dw * self.ws_lr)

    def update_ws(self) -> None:
        ef1 = (self.x1 - self.vf1) * (2.0 * self.vf_coeff)
        ef2 = (self.x2 - self.vf2) * (2.0 * self.vf_coeff)
        eb0 = (self.x0 - self.vb0) * (2.0 * self.vb_coeff)
        eb1 = (self.x1 - self.vb1) * (2.0 * self.vb_coeff)

        self._update_w(self.wf0, self.x0, ef1 * gate(self.x1))
        self._update_w(self.wf1, self.x1, ef2)
        self._update_w(self.wb1, self.x1, eb0)
        self._update_w(self.wb2, self.x2, eb1 * gate(self.x1))

    def predict(self, inputs: torch.Tensor) -> torch.Tensor:
        self.reset_state()
        self.x0.copy_(inputs)
        for _ in range(self.num_inference_iterations):
            self.update_vs()
            self.update_hidden()
            self.update_output()
        return self.x2.clone()

    def learn(self, inputs: torch.Tensor, targets: torch.Tensor) -> None:
        self.x0.copy_(inputs)
        self.x2.copy_(targets)
        for _ in range(self.num_inference_iterations):
            self.update_vs()
            self.update_hidden()
        self.update_ws()


def mse(prediction: torch.Tensor, targets: torch.Tensor) -> float:
    return ((prediction - targets) ** 2).mean().item()


def train_one_seed(
    seed: int,
    *,
    hidden_size: int,
    lr: float,
    n_epochs: int,
    device: torch.device,
) -> list[dict]:
    seed_everything(seed)
    inputs = torch.tensor(
        [[0.0, 0.0], [0.0, 1.0], [1.0, 0.0], [1.0, 1.0]],
        device=device,
    )
    targets = torch.tensor([[0.0], [1.0], [1.0], [0.0]], device=device)

    models = {
        "predictive_dendrites": ThreeQXORModel(hidden_size, lr, device),
        "hebbian": SimpleHebbianModel(hidden_size, lr, device),
        "bounded_hebbian": SimpleHebbianModel(
            hidden_size, lr, device, clamp_weights=True
        ),
    }

    rows: list[dict] = []
    for step in range(1, n_epochs + 1):
        for name, model in models.items():
            prediction = model.predict(inputs)
            rows.append(
                {
                    "seed": seed,
                    "lr": lr,
                    "step": step,
                    "model": name,
                    "mse": mse(prediction, targets),
                }
            )
        for model in models.values():
            model.learn(inputs, targets)
    return rows


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train firing-rate XOR models and write a long-form CSV."
    )
    parser.add_argument("--results-path", type=Path, default=RESULTS_PATH)
    parser.add_argument("--lr", type=float, default=LR)
    parser.add_argument("--hidden-size", type=int, default=HIDDEN_SIZE)
    parser.add_argument("--n-epochs", type=int, default=N_EPOCHS)
    parser.add_argument("--num-seeds", type=int, default=len(SEEDS))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    torch.set_num_threads(1)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    seeds = SEEDS[: args.num_seeds]

    rows: list[dict] = []
    for seed in seeds:
        print(f"Training seed={seed} lr={args.lr} on {device}")
        rows.extend(
            train_one_seed(
                seed,
                hidden_size=args.hidden_size,
                lr=args.lr,
                n_epochs=args.n_epochs,
                device=device,
            )
        )

    args.results_path.parent.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(rows, columns=["seed", "lr", "step", "model", "mse"])
    df.to_csv(args.results_path, index=False)
    print(f"Wrote {len(df)} rows to {args.results_path}")


if __name__ == "__main__":
    main()
