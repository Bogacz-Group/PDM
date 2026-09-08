import os
import argparse
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


def _read_n_train_iters_saved(save_dir):
    p = os.path.join(save_dir, "n_train_iters.npy")
    return int(np.load(p)) if os.path.isfile(p) else None


def _add_iteration_zero_black_borders(ax, df, facet_grid=None, linewidth=1.2):
    """
    Style scatter markers: edge color matches fill (no white halo), with a thin black
    border on iteration-0 points and the iteration-0 legend marker.
    """
    if df.empty or "Iteration" not in df.columns:
        return

    mask = df["Iteration"].to_numpy() == 0
    collections = [c for c in ax.collections if len(c.get_offsets()) > 0]
    if not collections:
        return

    for coll in collections:
        n = len(coll.get_offsets())
        coll.set_edgecolor("face")

        linewidths = np.asarray(coll.get_linewidths(), dtype=float)
        default_lw = float(linewidths.flat[0]) if linewidths.size else 0.5
        if default_lw <= 0:
            default_lw = 0.5
        coll.set_linewidths(default_lw)

        if not mask.any() or n != len(df):
            continue

        facecolors = np.asarray(coll.get_facecolors())
        if facecolors.shape[0] == 1:
            facecolors = np.tile(facecolors, (n, 1))
        facecolors = facecolors.copy()
        if facecolors.shape[1] == 3:
            facecolors = np.column_stack([facecolors, np.ones(n)])

        edgecolors = facecolors.copy()
        edgecolors[mask] = (0.0, 0.0, 0.0, 1.0)
        coll.set_edgecolors(edgecolors)

        linewidths = np.full(n, default_lw)
        linewidths[mask] = linewidth
        coll.set_linewidths(linewidths)

    legend = getattr(facet_grid, "_legend", None) if facet_grid is not None else ax.get_legend()
    if legend is None:
        return

    for handle, text in zip(legend.legend_handles, legend.get_texts()):
        if text.get_text() == "0":
            handle.set_markeredgecolor("black")
            handle.set_markeredgewidth(linewidth)
            break


_REL_PLOT_KW = dict(linewidth=0, alpha=0.5, aspect=1.15, edgecolor="face")


def _legacy_val_log_count(n_train_iters: int, print_every: int) -> int:
    steps = list(range(0, n_train_iters, print_every))
    if steps and steps[-1] != n_train_iters - 1:
        steps.append(n_train_iters - 1)
    return len(steps)


def _infer_optimizer_steps(train_len: int, val_len: int, print_every: int) -> int:
    for n in (train_len, train_len - 1):
        if n < 1:
            continue
        L = _legacy_val_log_count(n, print_every)
        if val_len == L + 1 and train_len == n + 1:
            return n
        if val_len == L and train_len == n:
            return n
    return train_len


def _true_n_train_iters_entry(entry: dict, print_every: int) -> int:
    saved = entry.get("n_train_iters_saved")
    if saved is not None:
        return int(saved)
    train = entry["train_losses"]
    val = entry["val_losses"]
    tr_len = int(train.shape[-1])
    val_len = int(val.shape[-1])
    return _infer_optimizer_steps(tr_len, val_len, print_every)


def test_steps_from_length(n_train_iters: int, print_every: int, n_test_points: int):
    """
    X-coordinates for val loss checkpoints. With init-prefixed logs (train_toy_linear):
    x=0 pre-train, then legacy loop indices + 1 (completed updates).
    """
    steps = list(range(0, n_train_iters, print_every))
    if steps and steps[-1] != n_train_iters - 1:
        steps.append(n_train_iters - 1)
    steps = np.asarray(steps, dtype=int)
    if len(steps) + 1 == n_test_points:
        return np.concatenate([[0], steps + 1])
    if len(steps) == n_test_points:
        return steps
    return np.linspace(0, n_train_iters - 1, n_test_points, dtype=int)


def _trajectories_to_data_list(all_trajectories, plot_every=1, max_iteration=None):
    """Build list of (x, y, iteration) for learning dynamics plot.

    Data are ordered by training iteration first, then by weight init, so that for
    each iteration we plot all initialisations together (matching plot_toy_nonlinear).
    """
    data_list = []

    # Find max trajectory length across all inits
    max_len = 0
    weight_histories = []
    for traj in all_trajectories:
        gen_w = np.asarray(traj["gen_weights"])      # shape (T,)
        amort_w = np.asarray(traj["amort_weights"])  # shape (T,)
        weight_histories.append((gen_w, amort_w))
        if gen_w.shape[0] > max_len:
            max_len = gen_w.shape[0]

    # Loop over training iterations first, then over inits
    for i in range(max_len):
        # i=0 is initialization (iteration 0, before any training)
        # i>0 corresponds to the state after i training updates (iteration i).
        #
        # IMPORTANT: do NOT map i=1 to iteration 0, otherwise iteration 0 appears
        # twice (init + after first update) and points look duplicated.
        iteration = i

        if max_iteration is not None and iteration > max_iteration:
            break
        if i != 0 and iteration % plot_every != 0:
            continue

        for gen_w, amort_w in weight_histories:
            if i >= gen_w.shape[0]:
                continue
            data_list.append(
                {
                    "x": float(gen_w[i]),
                    "y": float(amort_w[i]),
                    "Iteration": iteration,
                }
            )

    return data_list


def plot_trajectories(results_dir, title="PD", plot_every=500, max_iteration=None):
    """Learning dynamics scatter matching ordering in plot_toy_nonlinear."""
    trajectories_path = os.path.join(results_dir, "all_trajectories.npy")
    all_trajectories = np.load(trajectories_path, allow_pickle=True)

    data_list = _trajectories_to_data_list(
        all_trajectories, plot_every=plot_every, max_iteration=max_iteration
    )
    if not data_list:
        print(f"No trajectory points to plot in {results_dir}")
        return

    df = pd.DataFrame(data_list)

    max_iter = (
        max_iteration
        if max_iteration is not None
        else int(df["Iteration"].max())
    )
    legend_iterations = np.linspace(0, max_iter, 5, dtype=int).tolist()

    sns.set_theme(context="talk")

    g = sns.relplot(data=df, x="x", y="y", hue="Iteration", **_REL_PLOT_KW)

    ax = g.axes[0, 0]
    _add_iteration_zero_black_borders(ax, df, facet_grid=g)

    if len(g.figure.axes) > 1:
        cbar_ax = g.figure.axes[-1]
        cbar_ax.set_yticks(legend_iterations)
        cbar_ax.set_yticklabels([str(it) for it in legend_iterations])
        # Move the iteration colorbar higher up on the right side.
        bbox = ax.get_position()
        left = min(0.98 - cbar_ax.get_position().width, bbox.x1 + 0.01)
        height = cbar_ax.get_position().height
        top = 0.99
        bottom = max(0.01, top - height)
        cbar_ax.set_position([left, bottom, cbar_ax.get_position().width, height])
        cbar_ax.set_ylabel("Iteration")

    ax.set_xlabel(f"$w^{{2, f}}$", fontsize=35, labelpad=10)
    ax.set_ylabel(f"$w^{{2, b}}$", fontsize=35, labelpad=10)

    ax.set_xticks([-1, -0.5, 0, 0.5, 1])
    ax.set_yticks([-1, -0.5, 0, 0.5, 1])

    ax.set_xticklabels(["-1", "", "0", "", "1"])
    ax.set_yticklabels(["-1", "", "0", "", "1"])

    ax.tick_params(axis="both", labelsize=30)
    ax.set_title(title, fontsize=30, pad=20)

    plot_path = os.path.join(results_dir, "learning_dynamics.pdf")
    plt.savefig(plot_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Plot saved to {plot_path}")


def _path_prefix(slope=3.0, noise_std=0.0, width=1, n_hidden=1, act_fn="linear"):
    """Subdir path used by train_toy_linear.py for saved runs. Defaults match train_toy_linear.py."""
    return [f"slope_{slope}", f"noise_{noise_std}", f"width_{width}", f"n_hidden_{n_hidden}", f"act_{act_fn}"]


def _load_single_runs_losses(
    results_dir,
    n_seeds=3,
    slope=3.0,
    noise_std=0.0,
    width=1,
    n_hidden=1,
    act_fn="linear",
):
    """
    Load train/val losses for single_runs (non-grid) experiments created by train_toy_linear.py.

    Directory layout:
        results_dir/single_runs/{no_norm,unit_norm}/slope_*/noise_*/width_*/n_hidden_*/act_*/seed_*/
    """
    prefix = _path_prefix(slope, noise_std, width, n_hidden, act_fn)
    data = {}
    for norm_str in ["no_norm", "unit_norm"]:
        train_list = []
        val_list = []
        n_saved = None
        for seed in range(n_seeds):
            save_dir = os.path.join(
                results_dir,
                "single_runs",
                norm_str,
                *prefix,
                f"seed_{seed}",
            )
            train_path = os.path.join(save_dir, "train_losses.npy")
            val_path = os.path.join(save_dir, "val_losses.npy")
            if not (os.path.exists(train_path) and os.path.exists(val_path)):
                continue
            train_list.append(np.load(train_path))
            val_list.append(np.load(val_path))
            if n_saved is None:
                ns = _read_n_train_iters_saved(save_dir)
                if ns is not None:
                    n_saved = ns
        if train_list and val_list:
            data[norm_str] = {
                "train_losses": np.array(train_list),
                "val_losses": np.array(val_list),
                "n_train_iters_saved": n_saved,
            }
        else:
            data[norm_str] = None
    return data


def _load_single_runs_losses_over_lrs(
    results_root,
    n_seeds=3,
    slope=3.0,
    noise_std=0.0,
    width=1,
    n_hidden=1,
    act_fn="linear",
):
    """
    Load train/val losses for single_runs experiments across all learning rates.

    Expects directory layout created by train_toy_linear.py:
        results_root/lr_{param_lr}/single_runs/{no_norm,unit_norm}/...
    """
    prefix = _path_prefix(slope, noise_std, width, n_hidden, act_fn)
    lr_data = {"no_norm": {}, "unit_norm": {}}

    if not os.path.isdir(results_root):
        return lr_data

    for entry in os.listdir(results_root):
        if not entry.startswith("lr_"):
            continue
        lr_str = entry[len("lr_") :]
        try:
            lr_val = float(lr_str)
        except ValueError:
            continue

        lr_root = os.path.join(results_root, entry, "single_runs")
        if not os.path.isdir(lr_root):
            continue

        for norm_str in ["no_norm", "unit_norm"]:
            train_list = []
            val_list = []
            n_saved = None
            for seed in range(n_seeds):
                save_dir = os.path.join(
                    lr_root,
                    norm_str,
                    *prefix,
                    f"seed_{seed}",
                )
                train_path = os.path.join(save_dir, "train_losses.npy")
                val_path = os.path.join(save_dir, "val_losses.npy")
                if not (os.path.exists(train_path) and os.path.exists(val_path)):
                    continue
                train_list.append(np.load(train_path))
                val_list.append(np.load(val_path))
                if n_saved is None:
                    ns = _read_n_train_iters_saved(save_dir)
                    if ns is not None:
                        n_saved = ns

            if train_list and val_list:
                lr_data[norm_str][lr_val] = {
                    "train_losses": np.array(train_list),
                    "val_losses": np.array(val_list),
                    "n_train_iters_saved": n_saved,
                }

    return lr_data


def _compute_best_lrs_from_single_runs(
    results_root,
    n_seeds=3,
    slope=3.0,
    noise_std=0.0,
    width=1,
    n_hidden=1,
    act_fn="linear",
):
    """
    For each norm setting, compute:
      - best learning rate (lowest mean best val loss over seeds)
      - mean best val loss per learning rate.
    """
    lr_data = _load_single_runs_losses_over_lrs(
        results_root,
        n_seeds=n_seeds,
        slope=slope,
        noise_std=noise_std,
        width=width,
        n_hidden=n_hidden,
        act_fn=act_fn,
    )

    best_val_means = {"no_norm": {}, "unit_norm": {}}
    best_lr = {}

    for norm_str in ["no_norm", "unit_norm"]:
        for lr_val, entry in lr_data[norm_str].items():
            val_losses = entry["val_losses"]  # shape (n_seeds, T)
            if val_losses.size == 0:
                continue
            best_per_seed = val_losses.min(axis=1)  # best over time per seed
            best_val_means[norm_str][lr_val] = float(best_per_seed.mean())

        if best_val_means[norm_str]:
            best_lr[norm_str] = min(
                best_val_means[norm_str],
                key=lambda lr_: best_val_means[norm_str][lr_],
            )

    return lr_data, best_lr, best_val_means


def plot_train_loss_single_runs(
    results_dir,
    n_seeds=3,
    slope=3.0,
    noise_std=0.0,
    width=1,
    n_hidden=1,
    act_fn="linear",
    log_scale=False,
):
    """Plot train loss over training for single_runs (mean ± std over seeds) for both norm settings.

    Saved as train_losses.pdf so that it replaces the previous grid-based plot.
    """
    sns.set_theme(context="talk")
    fig, ax = plt.subplots(figsize=(7, 4))
    data = _load_single_runs_losses(
        results_dir,
        n_seeds=n_seeds,
        slope=slope,
        noise_std=noise_std,
        width=width,
        n_hidden=n_hidden,
        act_fn=act_fn,
    )
    colors = {"no_norm": "blue", "unit_norm": "red"}
    labels = {"no_norm": "Standard", "unit_norm": r"Norm $w^{2, f}$"}
    n_plotted = 0

    for norm_str in ["no_norm", "unit_norm"]:
        entry = data.get(norm_str)
        if entry is None:
            continue
        all_train_losses = entry["train_losses"]
        if all_train_losses.size == 0:
            continue
        n_plotted += 1
        mean_loss = np.mean(all_train_losses, axis=0)
        n_seeds_effective = all_train_losses.shape[0]
        std_error = np.std(all_train_losses, axis=0) / np.sqrt(n_seeds_effective)
        steps = np.arange(len(mean_loss))

        ax.plot(steps, mean_loss, color=colors[norm_str], label=labels[norm_str], linewidth=2)
        ax.fill_between(
            steps,
            mean_loss - std_error,
            mean_loss + std_error,
            color=colors[norm_str],
            alpha=0.2,
        )

    if n_plotted == 0:
        print("No single_runs train loss data found. Run train_toy_linear.py with non-grid experiments first.")

    ax.set_xlabel("Training step", fontsize=30, labelpad=10)
    ax.set_ylabel("Training loss", fontsize=30, labelpad=10)
    if log_scale:
        ax.set_yscale("log")
    ax.tick_params(axis="both", labelsize=25)
    if ax.get_legend_handles_labels()[1]:
        ax.legend(fontsize=18)
    ax.grid(True, alpha=0.3)

    plot_path = os.path.join(results_dir, "train_losses.pdf")
    plt.savefig(plot_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Plot saved to {plot_path}")


def plot_val_loss_single_runs(
    results_dir,
    n_seeds=3,
    slope=3.0,
    noise_std=0.0,
    width=1,
    n_hidden=1,
    act_fn="linear",
    print_every=100,
    log_scale=False,
):
    """Plot validation loss over training for single_runs (mean ± std over seeds) for both norm settings.

    Saved as val_losses.pdf so that it replaces the previous grid-based plot.
    """
    sns.set_theme(context="talk")
    fig, ax = plt.subplots(figsize=(7, 4))
    data = _load_single_runs_losses(
        results_dir,
        n_seeds=n_seeds,
        slope=slope,
        noise_std=noise_std,
        width=width,
        n_hidden=n_hidden,
        act_fn=act_fn,
    )
    colors = {"no_norm": "blue", "unit_norm": "red"}
    labels = {"no_norm": "Standard", "unit_norm": r"Norm $w^{2, f}$"}
    n_plotted = 0

    for norm_str in ["no_norm", "unit_norm"]:
        entry = data.get(norm_str)
        if entry is None:
            continue
        all_val_losses = entry["val_losses"]
        if all_val_losses.size == 0:
            continue
        n_plotted += 1
        mean_loss = np.mean(all_val_losses, axis=0)
        n_seeds_effective = all_val_losses.shape[0]
        std_error = np.std(all_val_losses, axis=0) / np.sqrt(n_seeds_effective)
        nti = _true_n_train_iters_entry(entry, print_every)
        steps = test_steps_from_length(nti, print_every, len(mean_loss))

        ax.plot(steps, mean_loss, color=colors[norm_str], label=labels[norm_str], linewidth=2)
        ax.fill_between(
            steps,
            mean_loss - std_error,
            mean_loss + std_error,
            color=colors[norm_str],
            alpha=0.2,
        )

    if n_plotted == 0:
        print("No single_runs val loss data found. Run train_toy_linear.py with non-grid experiments first.")

    ax.set_xlabel("Training step", fontsize=30, labelpad=10)
    ax.set_ylabel("Validation loss", fontsize=30, labelpad=10)
    if log_scale:
        ax.set_yscale("log")
    ax.tick_params(axis="both", labelsize=25)
    if ax.get_legend_handles_labels()[1]:
        ax.legend(fontsize=18)
    ax.grid(True, alpha=0.3)

    plot_path = os.path.join(results_dir, "val_losses.pdf")
    plt.savefig(plot_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Plot saved to {plot_path}")


def plot_train_loss_best_lr_single_runs(
    results_dir,
    n_seeds=3,
    slope=3.0,
    noise_std=0.0,
    width=1,
    n_hidden=1,
    act_fn="linear",
    log_scale=False,
):
    """
    Plot train loss over training for single_runs at the best learning rate
    per model (mean ± std error over seeds), matching the previous setup.
    Saved as train_losses.pdf.
    """
    sns.set_theme(context="talk")
    fig, ax = plt.subplots(figsize=(7, 4))

    lr_data, best_lr, _ = _compute_best_lrs_from_single_runs(
        results_dir,
        n_seeds=n_seeds,
        slope=slope,
        noise_std=noise_std,
        width=width,
        n_hidden=n_hidden,
        act_fn=act_fn,
    )

    colors = {"no_norm": "blue", "unit_norm": "red"}
    labels = {"no_norm": "Standard", "unit_norm": r"Norm $w^{2, f}$"}
    n_plotted = 0

    for norm_str in ["no_norm", "unit_norm"]:
        if norm_str not in best_lr:
            continue
        lr_val = best_lr[norm_str]
        entry = lr_data[norm_str][lr_val]
        all_train_losses = entry["train_losses"]
        if all_train_losses.size == 0:
            continue
        n_plotted += 1
        mean_loss = np.mean(all_train_losses, axis=0)
        n_seeds_effective = all_train_losses.shape[0]
        std_error = np.std(all_train_losses, axis=0, ddof=1) / np.sqrt(
            n_seeds_effective
        )
        steps = np.arange(len(mean_loss))

        ax.plot(steps, mean_loss, color=colors[norm_str], label=labels[norm_str], linewidth=2)
        ax.fill_between(
            steps,
            mean_loss - std_error,
            mean_loss + std_error,
            color=colors[norm_str],
            alpha=0.2,
        )

    if n_plotted == 0:
        print(
            "No single_runs train loss data found across learning rates. "
            "Run train_toy_linear.py with non-grid experiments first."
        )

    ax.set_xlabel("Training step", fontsize=30, labelpad=10)
    ax.set_ylabel("Training loss", fontsize=30, labelpad=10)
    if log_scale:
        ax.set_yscale("log")
    ax.tick_params(axis="both", labelsize=25)
    if ax.get_legend_handles_labels()[1]:
        ax.legend(fontsize=18)
    ax.grid(True, alpha=0.3)

    plot_path = os.path.join(results_dir, "best_train_losses.pdf")
    plt.savefig(plot_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Plot saved to {plot_path}")


def plot_val_loss_best_lr_single_runs(
    results_dir,
    n_seeds=3,
    slope=3.0,
    noise_std=0.0,
    width=1,
    n_hidden=1,
    act_fn="linear",
    print_every=100,
    log_scale=False,
):
    """
    Plot validation loss over training for single_runs at the best learning rate
    per model (mean ± std error over seeds), matching the previous setup.
    Saved as val_losses.pdf.
    """
    sns.set_theme(context="talk")
    fig, ax = plt.subplots(figsize=(7, 4))

    lr_data, best_lr, _ = _compute_best_lrs_from_single_runs(
        results_dir,
        n_seeds=n_seeds,
        slope=slope,
        noise_std=noise_std,
        width=width,
        n_hidden=n_hidden,
        act_fn=act_fn,
    )

    colors = {"no_norm": "blue", "unit_norm": "red"}
    labels = {"no_norm": "Standard", "unit_norm": r"Norm $w^{2, f}$"}
    n_plotted = 0

    for norm_str in ["no_norm", "unit_norm"]:
        if norm_str not in best_lr:
            continue
        lr_val = best_lr[norm_str]
        entry = lr_data[norm_str][lr_val]
        all_val_losses = entry["val_losses"]
        if all_val_losses.size == 0:
            continue
        n_plotted += 1
        mean_loss = np.mean(all_val_losses, axis=0)
        n_seeds_effective = all_val_losses.shape[0]
        std_error = np.std(all_val_losses, axis=0, ddof=1) / np.sqrt(
            n_seeds_effective
        )
        nti = _true_n_train_iters_entry(entry, print_every)
        steps = test_steps_from_length(nti, print_every, len(mean_loss))

        ax.plot(steps, mean_loss, color=colors[norm_str], label=labels[norm_str], linewidth=2)
        ax.fill_between(
            steps,
            mean_loss - std_error,
            mean_loss + std_error,
            color=colors[norm_str],
            alpha=0.2,
        )

    if n_plotted == 0:
        print(
            "No single_runs val loss data found across learning rates. "
            "Run train_toy_linear.py with non-grid experiments first."
        )

    ax.set_xlabel("Training step", fontsize=30, labelpad=10)
    ax.set_ylabel("Validation loss", fontsize=30, labelpad=10)
    if log_scale:
        ax.set_yscale("log")
    ax.tick_params(axis="both", labelsize=25)
    if ax.get_legend_handles_labels()[1]:
        ax.legend(fontsize=18)
    ax.grid(True, alpha=0.3)

    plot_path = os.path.join(results_dir, "best_val_losses.pdf")
    plt.savefig(plot_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Plot saved to {plot_path}")


def plot_best_lr_loss_curves_single_runs(
    results_dir,
    n_seeds=3,
    slope=3.0,
    noise_std=0.0,
    width=1,
    n_hidden=1,
    act_fn="linear",
    print_every=100,
    log_scale=False,
):
    """
    Plot train/val loss (mean ± std error over seeds) for the best learning rate
    of each model (no_norm vs unit_norm), based on single_runs only.
    """
    sns.set_theme(context="talk")

    lr_data, best_lr, _ = _compute_best_lrs_from_single_runs(
        results_dir,
        n_seeds=n_seeds,
        slope=slope,
        noise_std=noise_std,
        width=width,
        n_hidden=n_hidden,
        act_fn=act_fn,
    )

    if not best_lr:
        print(
            "No single_runs data found across learning rates. "
            "Run train_toy_linear.py with non-grid experiments first."
        )
        return

    fig, axes = plt.subplots(2, 1, figsize=(8, 6), sharex=True, constrained_layout=True)

    colors = {"no_norm": "blue", "unit_norm": "red"}
    labels = {"no_norm": "Standard", "unit_norm": r"Norm $w^{2, f}$"}

    for ax, norm_str in zip(axes, ["no_norm", "unit_norm"]):
        if norm_str not in best_lr:
            ax.set_visible(False)
            continue

        lr_val = best_lr[norm_str]
        entry = lr_data[norm_str][lr_val]
        train_losses = entry["train_losses"]  # (n_seeds, T_train)
        val_losses = entry["val_losses"]  # (n_seeds, T_val)

        n_seeds_effective = train_losses.shape[0]

        train_mean = np.mean(train_losses, axis=0)
        train_stderr = np.std(train_losses, axis=0, ddof=1) / np.sqrt(
            n_seeds_effective
        )
        train_steps = np.arange(len(train_mean))

        val_mean = np.mean(val_losses, axis=0)
        val_stderr = np.std(val_losses, axis=0, ddof=1) / np.sqrt(n_seeds_effective)
        nti = _true_n_train_iters_entry(entry, print_every)
        val_steps = test_steps_from_length(nti, print_every, len(val_mean))

        ax.plot(
            train_steps,
            train_mean,
            color=colors[norm_str],
            label=f"{labels[norm_str]} train (mean)",
            linewidth=2,
        )
        ax.fill_between(
            train_steps,
            train_mean - train_stderr,
            train_mean + train_stderr,
            color=colors[norm_str],
            alpha=0.2,
        )

        ax.plot(
            val_steps,
            val_mean,
            color="black",
            label=f"{labels[norm_str]} val (mean)",
            linewidth=2,
            linestyle="--",
        )
        ax.fill_between(
            val_steps,
            val_mean - val_stderr,
            val_mean + val_stderr,
            color="black",
            alpha=0.15,
        )

        if log_scale:
            ax.set_yscale("log")

        ax.set_ylabel("Loss", fontsize=20)
        ax.set_title(
            f"{labels[norm_str]} (best lr = {lr_val:.1e})",
            fontsize=22,
        )
        ax.grid(True, alpha=0.3)
        ax.tick_params(axis="both", labelsize=14)
        if ax.get_legend_handles_labels()[1]:
            ax.legend(fontsize=10)

    axes[-1].set_xlabel("Training step", fontsize=20)

    plot_path = os.path.join(results_dir, "best_lr_loss_curves.pdf")
    plt.savefig(plot_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Plot saved to {plot_path}")


def plot_best_val_vs_lr_single_runs(
    results_dir,
    n_seeds=3,
    slope=3.0,
    noise_std=0.0,
    width=1,
    n_hidden=1,
    act_fn="linear",
    log_scale_y=False,
):
    """
    Plot best (lowest) validation performance versus learning rate for each model
    (no_norm vs unit_norm), based on single_runs only.
    """
    sns.set_theme(context="talk")

    _, _, best_val_means = _compute_best_lrs_from_single_runs(
        results_dir,
        n_seeds=n_seeds,
        slope=slope,
        noise_std=noise_std,
        width=width,
        n_hidden=n_hidden,
        act_fn=act_fn,
    )

    if not (best_val_means["no_norm"] or best_val_means["unit_norm"]):
        print(
            "No single_runs validation data found across learning rates. "
            "Run train_toy_linear.py with non-grid experiments first."
        )
        return

    fig, ax = plt.subplots(figsize=(7, 4))

    for norm_str, color, label in [
        ("no_norm", "blue", "Standard"),
        ("unit_norm", "red", r"Norm $w^{2, f}$"),
    ]:
        if not best_val_means[norm_str]:
            continue

        lrs = sorted(best_val_means[norm_str].keys())
        vals = [best_val_means[norm_str][lr] for lr in lrs]

        ax.plot(lrs, vals, "-o", color=color, label=label, linewidth=2)

    ax.set_xscale("log")
    if log_scale_y:
        ax.set_yscale("log")

    ax.set_xlabel("Learning rate", fontsize=30)
    ax.set_ylabel("Best val. loss", fontsize=30)
    ax.grid(True, which="both", alpha=0.3)
    ax.tick_params(axis="both", labelsize=9)
    if ax.get_legend_handles_labels()[1]:
        ax.legend(fontsize=18)

    plot_path = os.path.join(results_dir, "best_val_vs_lr.pdf")
    plt.savefig(plot_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Plot saved to {plot_path}")


def plot_val_loss_comparison_single_init(
    results_dir,
    init_gen=1.0,
    init_amort=1.0,
    n_seeds=3,
    slope=3.0,
    noise_std=0.0,
    width=1,
    n_hidden=1,
    act_fn="linear",
    print_every=100,
    log_scale=False,
):
    """Plot validation loss over training for both norm settings (mean ± std over seeds, one init).
    init_gen/init_amort must match a run from train_toy_linear.py (default grid: -1, -0.5, 0, 0.5, 1)."""
    sns.set_theme(context="talk")
    fig, ax = plt.subplots(figsize=(7, 4))
    prefix = _path_prefix(slope, noise_std, width, n_hidden, act_fn)
    colors = {"no_norm": "blue", "unit_norm": "red"}
    labels = {"no_norm": "Standard", "unit_norm": r"Norm $w^{2, f}$"}
    n_plotted = 0

    for norm_str in ["no_norm", "unit_norm"]:
        all_val_losses = []
        n_saved = None
        train_len_ref = None
        for seed in range(n_seeds):
            save_dir = os.path.join(
                results_dir, norm_str, *prefix,
                f"init_gen_{init_gen:.2f}", f"init_amort_{init_amort:.2f}", f"seed_{seed}"
            )
            val_losses_path = os.path.join(save_dir, "val_losses.npy")
            if os.path.exists(val_losses_path):
                val_losses = np.load(val_losses_path)
                all_val_losses.append(val_losses)
                if n_saved is None:
                    n_saved = _read_n_train_iters_saved(save_dir)
                if train_len_ref is None:
                    tp = os.path.join(save_dir, "train_losses.npy")
                    if os.path.isfile(tp):
                        train_len_ref = len(np.load(tp))

        if len(all_val_losses) == 0:
            print(f"Skipping {norm_str} (no val losses found for init_gen={init_gen}, init_amort={init_amort})")
            continue

        n_plotted += 1
        all_val_losses = np.array(all_val_losses)
        mean_loss = np.mean(all_val_losses, axis=0)
        n_seeds_effective = all_val_losses.shape[0]
        std_error = np.std(all_val_losses, axis=0) / np.sqrt(n_seeds_effective)
        if n_saved is not None:
            nti = n_saved
        elif train_len_ref is not None:
            nti = _infer_optimizer_steps(
                train_len_ref, len(mean_loss), print_every
            )
        else:
            nti = _infer_optimizer_steps(
                len(mean_loss), len(mean_loss), print_every
            )
        steps = test_steps_from_length(nti, print_every, len(mean_loss))

        ax.plot(steps, mean_loss, color=colors[norm_str], label=labels[norm_str], linewidth=2)
        ax.fill_between(steps, mean_loss - std_error, mean_loss + std_error, color=colors[norm_str], alpha=0.2)

    if n_plotted == 0:
        print("No validation loss data found. Run train_toy_linear.py first (same --init_gen/--init_amort if using non-default).")

    ax.set_xlabel("Training step", fontsize=30, labelpad=10)
    ax.set_ylabel("Validation loss", fontsize=30, labelpad=10)
    if log_scale:
        ax.set_yscale("log")
    ax.tick_params(axis="both", labelsize=25)
    if ax.get_legend_handles_labels()[1]:
        ax.legend(fontsize=18)
    ax.grid(True, alpha=0.3)

    plot_path = os.path.join(results_dir, "val_losses.pdf")
    plt.savefig(plot_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Plot saved to {plot_path}")


def plot_train_loss_comparison_single_init(
    results_dir,
    init_gen=1.0,
    init_amort=1.0,
    n_seeds=3,
    slope=3.0,
    noise_std=0.0,
    width=1,
    n_hidden=1,
    act_fn="linear",
    log_scale=False,
):
    """Plot train loss over training for both norm settings (mean ± std over seeds, one init).
    init_gen/init_amort must match a run from train_toy_linear.py (default grid: -1, -0.5, 0, 0.5, 1)."""
    sns.set_theme(context="talk")
    fig, ax = plt.subplots(figsize=(7, 4))
    prefix = _path_prefix(slope, noise_std, width, n_hidden, act_fn)
    colors = {"no_norm": "blue", "unit_norm": "red"}
    labels = {"no_norm": "Standard", "unit_norm": r"Norm $w^{2, f}$"}
    n_plotted = 0

    for norm_str in ["no_norm", "unit_norm"]:
        all_train_losses = []
        for seed in range(n_seeds):
            save_dir = os.path.join(
                results_dir, norm_str, *prefix,
                f"init_gen_{init_gen:.2f}", f"init_amort_{init_amort:.2f}", f"seed_{seed}"
            )
            train_losses_path = os.path.join(save_dir, "train_losses.npy")
            if os.path.exists(train_losses_path):
                train_losses = np.load(train_losses_path)
                all_train_losses.append(train_losses)

        if len(all_train_losses) == 0:
            print(f"Skipping {norm_str} (no train losses found for init_gen={init_gen}, init_amort={init_amort})")
            continue

        n_plotted += 1
        all_train_losses = np.array(all_train_losses)
        mean_loss = np.mean(all_train_losses, axis=0)
        n_seeds_effective = all_train_losses.shape[0]
        std_error = np.std(all_train_losses, axis=0) / np.sqrt(n_seeds_effective)
        steps = np.arange(len(mean_loss))

        ax.plot(steps, mean_loss, color=colors[norm_str], label=labels[norm_str], linewidth=2)
        ax.fill_between(steps, mean_loss - std_error, mean_loss + std_error, color=colors[norm_str], alpha=0.2)

    if n_plotted == 0:
        print("No train loss data found. Run train_toy_linear.py first (same --init_gen/--init_amort if using non-default).")

    ax.set_xlabel("Training step", fontsize=30, labelpad=10)
    ax.set_ylabel("Training loss", fontsize=30, labelpad=10)
    if log_scale:
        ax.set_yscale("log")
    ax.tick_params(axis="both", labelsize=25)
    if ax.get_legend_handles_labels()[1]:
        ax.legend(fontsize=18)
    ax.grid(True, alpha=0.3)

    plot_path = os.path.join(results_dir, "train_losses.pdf")
    plt.savefig(plot_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Plot saved to {plot_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--results_dir", type=str, default="results/toy_linear")
    parser.add_argument("--init_gen", type=float, default=1.0)
    parser.add_argument("--init_amort", type=float, default=1.0)
    parser.add_argument("--n_seeds", type=int, default=3)
    parser.add_argument("--slope", type=float, default=3.0)
    parser.add_argument("--noise_std", type=float, default=0.5)
    parser.add_argument("--width", type=int, default=1)
    parser.add_argument("--n_hidden", type=int, default=1)
    parser.add_argument("--act_fn", type=str, default="linear")
    parser.add_argument("--print_every", type=int, default=100, help="Match train_toy_linear.py (x-axis step)")
    parser.add_argument("--plot_every_no_norm", type=int, default=10)
    parser.add_argument("--plot_every_unit_norm", type=int, default=10)
    parser.add_argument("--max_iteration_no_norm", type=int, default=3500)
    parser.add_argument("--max_iteration_unit_norm", type=int, default=500)
    parser.add_argument(
        "--linear_loss",
        action="store_true",
        help="Plot train/test losses on a linear scale instead of log scale (default).",
    )
    args = parser.parse_args()
    
    titles = {
        "no_norm": "Standard",
        "unit_norm": r"Normalised $w^{2, f}$",
    }

    plot_params = {
        "no_norm": {
            "plot_every": args.plot_every_no_norm,
            "max_iteration": args.max_iteration_no_norm,
        },
        "unit_norm": {
            "plot_every": args.plot_every_unit_norm,
            "max_iteration": args.max_iteration_unit_norm,
        },
    }

    # ------------------------------------------------------------------
    # Learning dynamics plots: one per learning rate and model
    # ------------------------------------------------------------------
    if os.path.isdir(args.results_dir):
        for entry in sorted(os.listdir(args.results_dir)):
            if not entry.startswith("lr_"):
                continue
            lr_subdir = os.path.join(args.results_dir, entry)
            for norm_str in ["no_norm", "unit_norm"]:
                subdir = os.path.join(lr_subdir, norm_str)
                traj_path = os.path.join(subdir, "all_trajectories.npy")
                if not os.path.exists(traj_path):
                    print(f"Skipping {entry}/{norm_str} (no trajectories found)")
                    continue

                print(f"\nPlotting learning dynamics for {entry}, {norm_str}...")
                plot_trajectories(
                    subdir,
                    title=titles[norm_str],
                    plot_every=plot_params[norm_str]["plot_every"],
                    max_iteration=plot_params[norm_str]["max_iteration"],
                )
    
    # Plot train and validation losses: single init only (mean ± std over seeds for that init)
    loss_kw = dict(
        slope=args.slope,
        noise_std=args.noise_std,
        width=args.width,
        n_hidden=args.n_hidden,
        act_fn=args.act_fn,
    )
    # For the linear toy, we now define the main train/val loss plots purely from
    # non-grid single_runs (aggregated over seeds), and save them as train_losses.pdf
    # and val_losses.pdf, replacing the older grid-based versions.
    single_runs_kw = dict(
        slope=args.slope,
        noise_std=args.noise_std,
        width=args.width,
        n_hidden=args.n_hidden,
        act_fn=args.act_fn,
    )
    print("\nPlotting best train losses (single_runs; best lr per model, same setup as before)...")
    plot_train_loss_best_lr_single_runs(
        args.results_dir,
        n_seeds=args.n_seeds,
        **single_runs_kw,
        log_scale=not args.linear_loss,
    )
    print("\nPlotting best val losses (single_runs; best lr per model, same setup as before)...")
    plot_val_loss_best_lr_single_runs(
        args.results_dir,
        n_seeds=args.n_seeds,
        print_every=args.print_every,
        **single_runs_kw,
        log_scale=not args.linear_loss,
    )
    print("\nPlotting best val. loss versus learning rate (single_runs)...")
    plot_best_val_vs_lr_single_runs(
        args.results_dir,
        n_seeds=args.n_seeds,
        **single_runs_kw,
        log_scale_y=not args.linear_loss,
    )
