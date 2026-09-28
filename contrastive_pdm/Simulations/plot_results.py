from __future__ import annotations

import json
import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from matplotlib.lines import Line2D

HERE = Path(__file__).resolve().parent
ROOT = HERE / "Results"
FIGDIR = HERE / "Figures"

MIN_RUNID = "20260921"
N_SEEDS = 5
RUNID = re.compile(r"_(\d{8}_\d{6})_accuracy_arrays\.npz$")
ACC = {
    "train": ["bp_trn_acc", "fixed_input_weights_bp_trn_acc", "trn_acc"],
    "val": ["bp_val_acc", "fixed_input_weights_bp_val_acc", "val_acc"],
    "test": ["bp_test_acc", "fixed_input_weights_bp_test_acc", "test_acc"],
}
DATASETS = {
    "MNIST": dict(
        beta=0.05,
        ylim_err=(0.0, 0.4),  # Fig. 4a
        ylim_err_supp=(0.0, 0.3),  # Supp. Fig. 6
    ),
    "CIFAR10": dict(
        beta=0.0025,
        ylim_err=(0.45, 0.95),  # Fig. 4a
        ylim_err_supp=(0.4, 0.8),  # Supp. Fig. 6
    ),
}

# Paper legend names / Fig. 4 colours. Contrastive (weak clamp) is light blue.
METHODS = {
    "PEP_FullClamp_NoFreePhase": dict(name="Standard", color="#1f77b4", ls="-", sweep=True),
    "PEP_WeakClamp_NoFreePhase": dict(name="Standard (weak clamp)", color="#e377c2", ls="-", sweep=True),
    "CPEP_WeakClamp": dict(name="Contrastive (weak clamp)", color="#17becf", ls="-", sweep=True),
    "CPEP_FullClamp": dict(name="Contrastive (full clamp)", color="#8c564b", ls="-", sweep=True),
    "FixedInputWeightsBP_Adam": dict(name="Fixed input weights (Adam)", color="#9467bd", ls="--", sweep=False),
    "FixedInputWeightsBP_SGD": dict(name="Fixed input weights (SGD)", color="#9467bd", ls="-", sweep=False),
    "BP_Adam": dict(name="Backpropagation (Adam)", color="black", ls="-", sweep=False),
    "BP_SGD": dict(name="Backpropagation (SGD)", color="black", ls=":", sweep=False),
}
FIG4A = [
    "PEP_FullClamp_NoFreePhase",
    "CPEP_WeakClamp",
    "FixedInputWeightsBP_Adam",
    "BP_Adam",
]
FIG4A_LABEL = {
    "PEP_FullClamp_NoFreePhase": "Standard",
    "CPEP_WeakClamp": "Contrastive",
    "FixedInputWeightsBP_Adam": "Fixed input weights",
    "BP_Adam": "Backpropagation",
}
SUPP6 = [
    "PEP_FullClamp_NoFreePhase",
    "PEP_WeakClamp_NoFreePhase",
    "CPEP_WeakClamp",
    "CPEP_FullClamp",
    "FixedInputWeightsBP_Adam",
    "FixedInputWeightsBP_SGD",
    "BP_Adam",
    "BP_SGD",
]


def _arr(z, keys, optional=False):
    for k in keys:
        if k in z:
            return np.atleast_2d(z[k].astype(float))
    if optional:
        return None
    raise KeyError(keys)


def _label(hp):
    if "lr_config_label" in hp:
        return f"lr={hp['lr_config_label']}, nlr={float(hp['neural_lr_start']):g}"
    if "lr" in hp:
        return f"lr={float(hp['lr']):g}"
    return "default"


def load_folder(dataset, algo, sweep, beta):
    folder = ROOT / dataset / algo
    keep = {}
    if not folder.exists():
        return []
    for npz in sorted(folder.glob("*_accuracy_arrays.npz")):
        m = RUNID.search(npz.name)
        runid = m.group(1) if m else None
        hp_path = Path(str(npz).replace("_accuracy_arrays.npz", "_hyperparams.json"))
        hp = json.loads(hp_path.read_text()) if hp_path.exists() else {}
        if sweep:
            if runid is None or runid < MIN_RUNID:
                continue
            if hp.get("activation", "relu") != "relu":
                continue
            if bool(hp.get("use_gating", "_gated" in npz.name)):
                continue
            if "beta" in hp and abs(float(hp["beta"]) - beta) > 1e-9:
                continue
            if hp.get("use_random_sign_beta", False):
                continue
        with np.load(npz) as z:
            test = _arr(z, ACC["test"])
            val = _arr(z, ACC["val"], optional=True)
            train = _arr(z, ACC["train"])
        if test.shape[0] < N_SEEDS:
            print(f"skip incomplete ({test.shape[0]}/{N_SEEDS}): {dataset}/{npz.name}")
            continue
        used_test = val is None
        if used_test:
            val = test
        lab = _label(hp)
        if lab not in keep or (runid or "") > keep[lab]["runid"]:
            keep[lab] = dict(
                label=lab,
                runid=runid or "",
                hparams=hp,
                arrays=dict(train=train, val=val, test=test),
                val=float(val[:, -1].mean()),
                test=float(test[:, -1].mean()),
                n=int(test.shape[0]),
                used_test=used_test,
            )
    return sorted(keep.values(), key=lambda r: r["val"], reverse=True)


def select_best():
    best = {}
    for dataset, spec in DATASETS.items():
        best[dataset] = {}
        print(f"\n=== {dataset}  β={spec['beta']:g} ===")
        for key, meta in METHODS.items():
            rows = load_folder(dataset, key, meta["sweep"], spec["beta"])
            if not rows:
                print(f"  WARNING: no runs for {key}")
                continue
            best[dataset][key] = rows[0]
            b = rows[0]
            print(
                f"  {meta['name']:<32} {b['label']:<22} "
                f"val={b['val']:.4f}  test={b['test']:.4f}  err={1 - b['test']:.4f}  (n={b['n']})"
            )
    return best


def plot_curves_new(best, dataset, keys, labels=None, ylim=None, save=None, legend=False, first_on_top=False):
    sns.set_theme(context="talk")
    fig, ax = plt.subplots(figsize=(7, 4))
    selected = best[dataset]
    present = [k for k in keys if k in selected]
    for key in keys:
        if key not in selected:
            print(f"skip missing: {dataset} {key}")
    n = len(present)
    for i, key in enumerate(present):
        arr = 1.0 - selected[key]["arrays"]["test"]
        mean = arr.mean(axis=0)
        n_seeds = arr.shape[0]
        spread = arr.std(axis=0, ddof=1) / np.sqrt(n_seeds) if n_seeds > 1 else np.zeros_like(mean)
        epochs = np.arange(len(mean))
        s = METHODS[key]
        name = (labels or {}).get(key, s["name"])
        rank = (n - 1 - i) if first_on_top else i
        z_fill = 2 * rank + 1
        z_line = 2 * rank + 2
        ax.plot(epochs, mean, color=s["color"], ls=s["ls"], label=name, linewidth=2, zorder=z_line)
        ax.fill_between(epochs, mean - spread, mean + spread, color=s["color"], alpha=0.2, zorder=z_fill)
    ax.set_xlabel("Training epoch", fontsize=30, labelpad=10)
    ax.set_ylabel("Test error", fontsize=30, labelpad=10)
    if ylim is not None:
        ax.set_ylim(*ylim)
    ax.tick_params(axis="both", labelsize=25)
    ax.grid(True, alpha=0.3)
    if legend:
        ax.legend(fontsize=11, frameon=False)
    fig.tight_layout()
    if save is not None:
        fig.savefig(save, dpi=150, bbox_inches="tight")
        print("saved", save)
    plt.close(fig)


def plot_legend(keys, labels=None, ncol=1, save=None):
    sns.set_theme(context="talk")
    handles, names = [], []
    for key in keys:
        s = METHODS[key]
        handles.append(Line2D([0], [0], color=s["color"], ls=s["ls"], linewidth=2))
        names.append((labels or {}).get(key, s["name"]))
    nrows = int(np.ceil(len(keys) / ncol))
    fig, ax = plt.subplots(figsize=(7, 0.55 * nrows + 0.4))
    ax.axis("off")
    ax.legend(
        handles,
        names,
        loc="center",
        frameon=False,
        fontsize=25,
        ncol=ncol,
        handlelength=2.8,
        handletextpad=0.6,
    )
    fig.tight_layout()
    if save is not None:
        fig.savefig(save, dpi=150, bbox_inches="tight")
        print("saved", save)
    plt.close(fig)


def main():
    FIGDIR.mkdir(parents=True, exist_ok=True)
    best = select_best()

    print("Fig. 4a")
    for dataset, spec in DATASETS.items():
        plot_curves_new(
            best,
            dataset,
            FIG4A,
            labels=FIG4A_LABEL,
            ylim=spec["ylim_err"],
            save=FIGDIR / f"Fig4a_{dataset}_test_error.pdf",
            first_on_top=True,
        )
    plot_legend(FIG4A, labels=FIG4A_LABEL, ncol=1, save=FIGDIR / "Fig4a_legend.pdf")

    print("Supplementary Fig. 6")
    for dataset, spec in DATASETS.items():
        keys = [k for k in SUPP6 if not (dataset == "CIFAR10" and k == "PEP_WeakClamp_NoFreePhase")]
        plot_curves_new(
            best,
            dataset,
            keys,
            ylim=spec["ylim_err_supp"],
            save=FIGDIR / f"SuppFig6_{dataset}_test_error.pdf",
            first_on_top=True,
        )
    plot_legend(SUPP6, ncol=1, save=FIGDIR / "SuppFig6_legend.pdf")


if __name__ == "__main__":
    main()
