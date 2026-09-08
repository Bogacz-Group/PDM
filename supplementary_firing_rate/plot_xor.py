"""
Plot XOR training results: losses and accuracies for models with and without
unit-norm first layer. Training and test metrics are in separate plots.
"""

import os
import argparse
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Match style used in replications/plot_toy_linear.py
LABELS = {"no_norm": "Standard", "unit_norm": r"Normalised $w^{2, f}$"}
COLORS = {"no_norm": "blue", "unit_norm": "red"}


def _base_dir(results_dir, width, n_hidden, act_fn):
    base = os.path.join(
        results_dir,
        f"width_{width}",
        f"n_hidden_{n_hidden}",
        f"act_{act_fn}",
    )
    return base


def _seed_dirs(base, unit_norm: bool, seed: int | None):
    """Return list of seed directories under base/unit_norm_{True,False}."""
    root = os.path.join(base, f"unit_norm_{unit_norm}")
    if seed is not None:
        return [os.path.join(root, f"seed_{seed}")]
    if not os.path.isdir(root):
        return []
    seeds = []
    for name in sorted(os.listdir(root)):
        if not name.startswith("seed_"):
            continue
        p = os.path.join(root, name)
        if os.path.isdir(p):
            seeds.append(p)
    return seeds


def _maybe_load(path):
    if os.path.isfile(path):
        return np.load(path)
    return None


def load_metrics(save_dir, print_every):
    """
    Load train/test losses and accs from save_dir.
    Returns dict with arrays and step indices for test (eval every print_every).
    """
    train_losses = np.load(os.path.join(save_dir, "train_losses.npy"))
    test_losses = np.load(os.path.join(save_dir, "test_losses.npy"))
    train_accs = np.load(os.path.join(save_dir, "train_accs.npy"))
    test_accs = np.load(os.path.join(save_dir, "test_accs.npy"))
    test_infer_losses = _maybe_load(os.path.join(save_dir, "test_infer_losses.npy"))
    test_infer_accs = _maybe_load(os.path.join(save_dir, "test_infer_accs.npy"))
    n = len(train_losses)
    # Test metrics are logged at steps 0, print_every, 2*print_every, ..., and final step
    n_test = len(test_losses)
    test_steps = np.minimum(
        np.arange(n_test) * print_every,
        n - 1,
    )
    if n_test > 1 and test_steps[-1] != n - 1:
        test_steps[-1] = n - 1
    return {
        "train_losses": train_losses,
        "test_losses": test_losses,
        "test_infer_losses": test_infer_losses,
        "train_accs": train_accs,
        "test_accs": test_accs,
        "test_infer_accs": test_infer_accs,
        "train_steps": np.arange(n),
        "test_steps": test_steps,
    }


def _stack_to_common_length(arrs):
    """Trim arrays to a common (min) length, returning stacked array and common length."""
    lengths = [int(a.shape[0]) for a in arrs]
    L = min(lengths)
    if any(l != L for l in lengths):
        print(f"Warning: metric lengths differ {lengths}; trimming to {L}.")
    return np.stack([a[:L] for a in arrs], axis=0), L


def _aggregate_metrics(seed_dirs, print_every):
    """Load and aggregate metrics across seeds: mean and standard error."""
    if not seed_dirs:
        return None

    per_seed = []
    for d in seed_dirs:
        if not os.path.isdir(d):
            continue
        try:
            per_seed.append(load_metrics(d, print_every))
        except FileNotFoundError:
            continue

    if not per_seed:
        return None

    n_seeds = len(per_seed)

    # Train metrics: align by trimming to min length across seeds
    train_losses_stack, train_len = _stack_to_common_length(
        [m["train_losses"] for m in per_seed]
    )
    train_accs_stack, _ = _stack_to_common_length([m["train_accs"] for m in per_seed])

    # Test metrics: align by trimming to min length across seeds
    test_losses_stack, test_len = _stack_to_common_length(
        [m["test_losses"] for m in per_seed]
    )
    test_accs_stack, _ = _stack_to_common_length([m["test_accs"] for m in per_seed])

    infer_loss_seeds = [m["test_infer_losses"] for m in per_seed if m["test_infer_losses"] is not None]
    infer_acc_seeds = [m["test_infer_accs"] for m in per_seed if m["test_infer_accs"] is not None]
    has_infer = (
        len(infer_loss_seeds) == n_seeds
        and len(infer_acc_seeds) == n_seeds
    )
    if has_infer:
        test_infer_losses_stack, infer_len = _stack_to_common_length(infer_loss_seeds)
        test_infer_accs_stack, _ = _stack_to_common_length(infer_acc_seeds)
    else:
        test_infer_losses_stack = test_infer_accs_stack = None
        infer_len = 0

    # Steps: recompute deterministically from common lengths
    train_steps = np.arange(train_len)
    test_steps = np.minimum(np.arange(test_len) * print_every, train_len - 1)
    if test_len > 1 and test_steps[-1] != train_len - 1:
        test_steps[-1] = train_len - 1
    infer_steps = np.minimum(np.arange(infer_len) * print_every, train_len - 1)
    if infer_len > 1 and infer_steps[-1] != train_len - 1:
        infer_steps[-1] = train_len - 1

    def mean_stderr(stack):
        mean = stack.mean(axis=0)
        if stack.shape[0] <= 1:
            stderr = np.zeros_like(mean)
        else:
            stderr = stack.std(axis=0, ddof=1) / np.sqrt(stack.shape[0])
        return mean, stderr

    train_losses_mean, train_losses_se = mean_stderr(train_losses_stack)
    train_accs_mean, train_accs_se = mean_stderr(train_accs_stack)
    test_losses_mean, test_losses_se = mean_stderr(test_losses_stack)
    test_accs_mean, test_accs_se = mean_stderr(test_accs_stack)
    if has_infer:
        test_infer_losses_mean, test_infer_losses_se = mean_stderr(test_infer_losses_stack)
        test_infer_accs_mean, test_infer_accs_se = mean_stderr(test_infer_accs_stack)
    else:
        test_infer_losses_mean = test_infer_losses_se = None
        test_infer_accs_mean = test_infer_accs_se = None

    return {
        "n_seeds": n_seeds,
        "train_steps": train_steps,
        "test_steps": test_steps,
        "infer_steps": infer_steps,
        "has_infer": has_infer,
        "train_losses_mean": train_losses_mean,
        "train_losses_se": train_losses_se,
        "test_losses_mean": test_losses_mean,
        "test_losses_se": test_losses_se,
        "test_infer_losses_mean": test_infer_losses_mean,
        "test_infer_losses_se": test_infer_losses_se,
        "train_accs_mean": train_accs_mean,
        "train_accs_se": train_accs_se,
        "test_accs_mean": test_accs_mean,
        "test_accs_se": test_accs_se,
        "test_infer_accs_mean": test_infer_accs_mean,
        "test_infer_accs_se": test_infer_accs_se,
    }


def _styled_plot(out_dir, filename, x, y_mean, y_se, x2, y2_mean, y2_se, ylabel):
    """Two-series plot with standard error bands, matching replications/plot_toy_linear.py."""
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(
        x2,
        y2_mean,
        label=LABELS["no_norm"],
        color=COLORS["no_norm"],
        linewidth=2,
    )
    ax.fill_between(
        x2,
        y2_mean - y2_se,
        y2_mean + y2_se,
        color=COLORS["no_norm"],
        alpha=0.2,
    )
    ax.plot(
        x,
        y_mean,
        label=LABELS["unit_norm"],
        color=COLORS["unit_norm"],
        linewidth=2,
    )
    ax.fill_between(
        x,
        y_mean - y_se,
        y_mean + y_se,
        color=COLORS["unit_norm"],
        alpha=0.2,
    )
    ax.set_xlabel("Training step", fontsize=30, labelpad=10)
    ax.set_ylabel(ylabel, fontsize=30, labelpad=10)
    ax.legend(fontsize=18)
    ax.tick_params(axis="both", labelsize=25)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(left=0)
    plt.tight_layout()
    save_path = os.path.join(out_dir, filename)
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved {save_path}")


def plot(results_dir, width, n_hidden, act_fn, seed, print_every, out_dir):
    sns.set_theme(context="talk")
    base = _base_dir(results_dir, width, n_hidden, act_fn)
    unit_seed_dirs = _seed_dirs(base, unit_norm=True, seed=seed)
    no_unit_seed_dirs = _seed_dirs(base, unit_norm=False, seed=seed)

    if not unit_seed_dirs or not no_unit_seed_dirs:
        raise FileNotFoundError(
            "Expected results in both:\n"
            f"  {os.path.join(base, 'unit_norm_True')}\n"
            f"  {os.path.join(base, 'unit_norm_False')}\n"
            "Run train_xor.py first with matching --results_dir, --width, --n_hidden, --act_fn."
        )

    m_unit = _aggregate_metrics(unit_seed_dirs, print_every)
    m_no_unit = _aggregate_metrics(no_unit_seed_dirs, print_every)
    if m_unit is None or m_no_unit is None:
        raise FileNotFoundError(
            "Could not load metrics for one (or both) settings. "
            "Check that each `seed_*` directory contains `train_losses.npy`, `test_losses.npy`, "
            "`train_accs.npy`, and `test_accs.npy`."
        )

    os.makedirs(out_dir, exist_ok=True)

    _styled_plot(
        out_dir, "train_losses.pdf",
        m_unit["train_steps"], m_unit["train_losses_mean"], m_unit["train_losses_se"],
        m_no_unit["train_steps"], m_no_unit["train_losses_mean"], m_no_unit["train_losses_se"],
        "Train loss",
    )
    _styled_plot(
        out_dir, "test_losses.pdf",
        m_unit["test_steps"], m_unit["test_losses_mean"], m_unit["test_losses_se"],
        m_no_unit["test_steps"], m_no_unit["test_losses_mean"], m_no_unit["test_losses_se"],
        "Test loss",
    )
    _styled_plot(
        out_dir, "train_accs.pdf",
        m_unit["train_steps"], m_unit["train_accs_mean"], m_unit["train_accs_se"],
        m_no_unit["train_steps"], m_no_unit["train_accs_mean"], m_no_unit["train_accs_se"],
        "Accuracy",
    )
    _styled_plot(
        out_dir, "test_accs.pdf",
        m_unit["test_steps"], m_unit["test_accs_mean"], m_unit["test_accs_se"],
        m_no_unit["test_steps"], m_no_unit["test_accs_mean"], m_no_unit["test_accs_se"],
        "Test accuracy",
    )
    if m_unit["has_infer"] and m_no_unit["has_infer"]:
        _styled_plot(
            out_dir, "test_infer_losses.pdf",
            m_unit["infer_steps"], m_unit["test_infer_losses_mean"], m_unit["test_infer_losses_se"],
            m_no_unit["infer_steps"], m_no_unit["test_infer_losses_mean"], m_no_unit["test_infer_losses_se"],
            "Test loss",
        )
        _styled_plot(
            out_dir, "test_infer_accs.pdf",
            m_unit["infer_steps"], m_unit["test_infer_accs_mean"], m_unit["test_infer_accs_se"],
            m_no_unit["infer_steps"], m_no_unit["test_infer_accs_mean"], m_no_unit["test_infer_accs_se"],
            "Test accuracy",
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Plot XOR train/test losses and accuracies (unit norm vs no unit norm).")
    parser.add_argument("--results_dir", type=str, default="results/xor", help="Base results directory from train_xor.py")
    parser.add_argument("--width", type=int, default=2)
    parser.add_argument("--n_hidden", type=int, default=1)
    parser.add_argument("--act_fn", type=str, default="relu")
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="If set, plot only this seed; otherwise aggregate over all available `seed_*` dirs.",
    )
    parser.add_argument("--print_every", type=int, default=100, help="Must match value used in train_xor.py for correct test step axis.")
    parser.add_argument("--out_dir", type=str, default=None, help="Output directory for PDFs; default: results_dir/plots_width_W_n_hidden_H_act_A_seed_S")

    args = parser.parse_args()
    out_dir = args.out_dir
    if out_dir is None:
        seed_str = f"{args.seed}" if args.seed is not None else "all"
        out_dir = os.path.join(
            args.results_dir,
            "plots",
            f"width_{args.width}_n_hidden_{args.n_hidden}_act_{args.act_fn}_seed_{seed_str}",
        )
    plot(
        results_dir=args.results_dir,
        width=args.width,
        n_hidden=args.n_hidden,
        act_fn=args.act_fn,
        seed=args.seed,
        print_every=args.print_every,
        out_dir=out_dir,
    )
