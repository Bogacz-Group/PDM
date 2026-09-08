import os
import argparse
import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


# Default model subdirs and display names (must match train_toy_nonlinear.py)
MODEL_DIRS = [
    "standard_pdm",
    "unit_norm_pdm",
    "lateral_pdm",
]
MODEL_LABELS = [
    r"Standard",
    r"Normalised $w^{2, f}$",
    r"Normalised $w^{2, f}$ + lateral",
]
MODEL_COLORS = ["#1f77b4", "#d62728", "#2ca02c"]  # standard=blue, unit_norm=red, lateral=green


def _seed_result_paths(results_dir, model_dir, filename, n_seeds=None):
    """
    Return paths to a result file under seed_* subdirs.

    If n_seeds is set, only seed_0 .. seed_{n_seeds - 1} are considered (matching
    train_toy_nonlinear.py). Otherwise all seed_* directories are included.
    """
    if n_seeds is not None:
        paths = []
        for seed in range(n_seeds):
            p = os.path.join(results_dir, f"seed_{seed}", model_dir, filename)
            if os.path.isfile(p):
                paths.append(p)
        return paths
    pattern = os.path.join(results_dir, "seed_*", model_dir, filename)
    return sorted(glob.glob(pattern))


def _read_n_train_iters_saved(dir_path):
    p = os.path.join(dir_path, "n_train_iters.npy")
    return int(np.load(p)) if os.path.isfile(p) else None


def _iter_models(plot_standard_pd: bool = True):
    """
    Iterate (model_dir, label, color) tuples.

    If plot_standard_pd is False, skip the standard PD baseline (MODEL_DIRS[0]).
    """
    start = 0 if plot_standard_pd else 1
    return zip(MODEL_DIRS[start:], MODEL_LABELS[start:], MODEL_COLORS[start:])


def has_grid_trajectories(results_dir):
    """True if any model has all_trajectories.npy (from --grid_experiment)."""
    return get_dynamics_results_dir(results_dir) is not None


def has_activity_histories(results_dir):
    """
    True if any model has hidden_activity_history.npy saved, either in:
      - results_dir/seed_*/<model>/hidden_activity_history.npy, or
      - results_dir/<model>/hidden_activity_history.npy.
    """
    # Check seed_* layout first.
    for key in MODEL_DIRS:
        pattern = os.path.join(results_dir, "seed_*", key, "hidden_activity_history.npy")
        if glob.glob(pattern):
            return True
    # Fallback: single-run layout.
    for key in MODEL_DIRS:
        path = os.path.join(results_dir, key, "hidden_activity_history.npy")
        if os.path.exists(path):
            return True
    return False


def _load_hidden_activities_from_seeds(results_dir, model_dir, n_seeds=None):
    """
    Load hidden activity histories from multiple seed subdirs:
    results_dir/seed_*/<model_dir>/hidden_activity_history.npy.

    Returns an array of shape (n_seeds, T, H) where T is the number of
    recorded training iterations and H is the number of hidden units.
    If no data is found, returns None.
    """
    files = _seed_result_paths(
        results_dir, model_dir, "hidden_activity_history.npy", n_seeds=n_seeds
    )
    if not files:
        return None
    acts_list = [np.load(f) for f in files]
    # Ensure all trajectories have the same length; truncate to the shortest if needed.
    min_len = min(a.shape[0] for a in acts_list)
    acts_list = [a[:min_len] for a in acts_list]
    return np.stack(acts_list, axis=0)


def _load_hidden_activities_single_run(results_dir, model_dir):
    """
    Load hidden activity history for a single-run layout:
    results_dir/<model_dir>/hidden_activity_history.npy.

    Returns array of shape (1, T, H) for consistency, or None if missing.
    """
    path = os.path.join(results_dir, model_dir, "hidden_activity_history.npy")
    if not os.path.exists(path):
        return None
    arr = np.load(path)
    if arr.ndim == 2:
        arr = arr[np.newaxis, ...]
    return arr


def get_dynamics_results_dir(results_dir):
    """
    Return the directory that contains all_trajectories.npy for learning dynamics.
    - If trajectories exist at results_dir/<model>/all_trajectories.npy, return results_dir.
    - If they exist at results_dir/grid_seed_*/<model>/all_trajectories.npy, return the first grid_seed_* path.
    - Otherwise return None.
    """
    for key in MODEL_DIRS:
        path_top = os.path.join(results_dir, key, "all_trajectories.npy")
        if os.path.exists(path_top):
            return results_dir
    grid_dirs = sorted(glob.glob(os.path.join(results_dir, "grid_seed_*")))
    for gdir in grid_dirs:
        if os.path.isdir(gdir):
            if any(
                os.path.exists(os.path.join(gdir, key, "all_trajectories.npy"))
                for key in MODEL_DIRS
            ):
                return gdir
    return None


def _load_losses_from_seeds(results_dir, model_dir, n_seeds=None):
    """
    Load train/val losses from multiple seed subdirs:
    results_dir/seed_*/<model_dir>/train_losses.npy and val_losses.npy.
    Falls back to test_losses.npy for backward compatibility.
    Returns dict with (n_seeds, n_steps) arrays or None if nothing found.
    """
    train_files = _seed_result_paths(
        results_dir, model_dir, "train_losses.npy", n_seeds=n_seeds
    )
    if not train_files:
        return None
    train_list = []
    val_list = []
    n_saved = None
    for tf in train_files:
        dir_path = os.path.dirname(tf)
        if n_saved is None:
            ns = _read_n_train_iters_saved(dir_path)
            if ns is not None:
                n_saved = ns
        train_list.append(np.load(tf))
        # Prefer new val_losses.npy, fall back to legacy test_losses.npy
        val_path = os.path.join(dir_path, "val_losses.npy")
        if not os.path.exists(val_path):
            val_path = os.path.join(dir_path, "test_losses.npy")
        if not os.path.exists(val_path):
            return None
        val_list.append(np.load(val_path))
    train_stack = np.array(train_list)
    val_stack = np.array(val_list)
    return {
        "train_losses": train_stack,
        "val_losses": val_stack,
        "n_train_iters_saved": n_saved,
    }


def _load_losses_from_grid(results_dir, model_dir):
    """
    Load train/val losses from grid run subdirs
    (results_dir/<model_dir>/init_w11_*/init_w12_*/train_losses.npy etc.).
    Falls back to test_losses.npy for backward compatibility.
    Returns (n_runs, n_steps) arrays for mean ± std plotting, or None if no grid loss files found.
    """
    pattern = os.path.join(results_dir, model_dir, "init_w11_*", "init_w12_*", "train_losses.npy")
    train_files = sorted(glob.glob(pattern))
    if not train_files:
        return None
    train_list = []
    val_list = []
    n_saved = None
    for tf in train_files:
        dir_path = os.path.dirname(tf)
        if n_saved is None:
            ns = _read_n_train_iters_saved(dir_path)
            if ns is not None:
                n_saved = ns
        train_list.append(np.load(tf))
        # Prefer new val_losses.npy, fall back to legacy test_losses.npy
        val_path = os.path.join(dir_path, "val_losses.npy")
        if not os.path.exists(val_path):
            val_path = os.path.join(dir_path, "test_losses.npy")
        if not os.path.exists(val_path):
            return None
        val_list.append(np.load(val_path))
    train_stack = np.array(train_list)
    val_stack = np.array(val_list)
    return {
        "train_losses": train_stack,
        "val_losses": val_stack,
        "n_train_iters_saved": n_saved,
    }


def load_losses(results_dir, print_every=100, warn_missing=True, n_seeds=None):
    """
    Load train_losses.npy and val_losses.npy for each model.
    First tries top-level files (single run). If missing, loads from grid run
    subdirs (init_w11_*/init_w12_*/). All arrays are returned as (n_runs, n_steps)
    so plotting can show mean ± 1 std across runs.

    Returns:
        data: dict mapping model_key -> {"train_losses": (n_runs, n_steps), "val_losses": (n_runs, n_steps)}
        or None for a model if no data found. Falls back to legacy test_losses.npy when needed.
    """
    data = {}
    for key in MODEL_DIRS:
        # 1) Prefer multiple seeds layout: results_dir/seed_*/<model>/
        seed_data = _load_losses_from_seeds(results_dir, key, n_seeds=n_seeds)
        if seed_data is not None:
            data[key] = seed_data
            continue

        # 2) Backwards compatible: top-level single run in results_dir/<model>/
        path = os.path.join(results_dir, key)
        train_path = os.path.join(path, "train_losses.npy")
        # Prefer new val_losses.npy, fall back to legacy test_losses.npy
        val_path = os.path.join(path, "val_losses.npy")
        if not os.path.exists(val_path):
            val_path = os.path.join(path, "test_losses.npy")
        if os.path.exists(train_path) and os.path.exists(val_path):
            train = np.load(train_path)
            val = np.load(val_path)
            # Single run: store as (1, n_steps) for consistent mean/std in plotting
            if train.ndim == 1:
                train = train[np.newaxis, :]
                val = val[np.newaxis, :]
            data[key] = {
                "train_losses": train,
                "val_losses": val,
                "n_train_iters_saved": _read_n_train_iters_saved(path),
            }
            continue

        # 3) Fallback: load all runs from grid subdirs
        grid_data = _load_losses_from_grid(results_dir, key)
        if grid_data is not None:
            data[key] = grid_data
            continue

        if warn_missing:
            print(f"Warning: missing data for {key} (expected {train_path}, {val_path})")
        data[key] = None
    return data


def load_losses_single_init(results_dir, init_w11=1.0, init_w12=1.0):
    """
    Load train/val losses for a single grid point (init_w11, init_w12) for each model.
    Used for val loss comparison at one init. Returns same structure as load_losses
    with (1, n_steps) per model, or None for a model if that init dir is missing.
    """
    data = {}
    for key in MODEL_DIRS:
        dir_path = os.path.join(
            results_dir, key,
            f"init_w11_{init_w11:.2f}",
            f"init_w12_{init_w12:.2f}",
        )
        train_path = os.path.join(dir_path, "train_losses.npy")
        # Prefer new val_losses.npy, fall back to legacy test_losses.npy
        val_path = os.path.join(dir_path, "val_losses.npy")
        if not os.path.exists(val_path):
            val_path = os.path.join(dir_path, "test_losses.npy")
        if not os.path.exists(train_path) or not os.path.exists(val_path):
            data[key] = None
            continue
        train = np.load(train_path)
        val = np.load(val_path)
        if train.ndim == 1:
            train = train[np.newaxis, :]
            val = val[np.newaxis, :]
        data[key] = {
            "train_losses": train,
            "val_losses": val,
            "n_train_iters_saved": _read_n_train_iters_saved(dir_path),
        }
    return data


def _mean_std(arr):
    """From (n_runs, n_steps) return mean and std along axis=0. Handles single run (std=0)."""
    if arr.ndim == 1:
        return arr, np.zeros_like(arr)
    mean = np.mean(arr, axis=0)
    std = np.std(arr, axis=0)
    return mean, std


def _mean_stderr(arr):
    """From (n_runs, n_steps) return mean and standard error along axis=0. Handles single run (stderr=0)."""
    if arr.ndim == 1:
        return arr, np.zeros_like(arr)
    mean = np.mean(arr, axis=0)
    # Standard error of the mean: std / sqrt(n_runs)
    std = np.std(arr, axis=0)
    n = arr.shape[0]
    stderr = std / np.sqrt(max(n, 1))
    return mean, stderr


def _median_iqr(arr):
    """From (n_runs, n_steps) return median and (q1, q3) along axis=0 for IQR. Handles single run (q1=q3=median)."""
    if arr.ndim == 1:
        return arr, arr, arr
    median = np.median(arr, axis=0)
    q1 = np.percentile(arr, 25, axis=0)
    q3 = np.percentile(arr, 75, axis=0)
    return median, q1, q3


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


def _true_n_train_iters(entry: dict, print_every: int) -> int:
    saved = entry.get("n_train_iters_saved")
    if saved is not None:
        return int(saved)
    train = entry["train_losses"]
    val = entry["val_losses"]
    tr_len = int(train.shape[-1])
    val_len = int(val.shape[-1])
    return _infer_optimizer_steps(tr_len, val_len, print_every)


def test_steps_from_length(n_train_iters, print_every, n_test_points):
    """
    X-coordinates for logged val loss. Legacy: loop step indices 0, print_every, ...
    With init-prefixed logs: x=0 (pre-train), then (legacy_index + 1).
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


def plot_train_losses(
    data,
    results_dir,
    out_name="train_losses.pdf",
    plot_standard_pd=True,
    log_scale=False,
    max_steps=None,
    use_median=False,
    use_stderr=False,
):
    """Plot train loss vs step with an uncertainty band.

    Options:
      - Default (no flags): mean ± standard error of the mean
      - use_stderr: mean ± standard error of the mean
      - use_median: median with IQR (25th–75th percentile)
      - Otherwise (if extended): mean ± 1 std
    """
    sns.set_theme(context="talk")
    fig, ax = plt.subplots(figsize=(7, 4))
    for (key, label, color) in _iter_models(plot_standard_pd=plot_standard_pd):
        if data.get(key) is None:
            continue
        train = data[key]["train_losses"]
        if use_median:
            center, low, high = _median_iqr(train)
        elif use_stderr:
            center, stderr = _mean_stderr(train)
            low, high = center - stderr, center + stderr
        else:
            center, std = _mean_std(train)
            low, high = center - std, center + std
        if max_steps is not None:
            n = min(len(center), max_steps)
            center = center[:n]
            low = low[:n]
            high = high[:n]
        steps = np.arange(len(center))
        ax.plot(steps, center, label=label, color=color, alpha=0.9, linewidth=2)
        if train.ndim > 1 and train.shape[0] > 1:
            ax.fill_between(steps, low, high, color=color, alpha=0.18)
    ax.set_xlabel("Training step", fontsize=30, labelpad=10)
    ax.set_ylabel("Training loss", fontsize=30, labelpad=10)
    ax.legend(loc="upper right", fontsize=20)
    ax.tick_params(axis="both", labelsize=25)
    ax.set_yscale("log" if log_scale else "linear")
    ax.grid(True, alpha=0.3)
    ax.set_xlim(left=0)
    plt.tight_layout()
    save_path = os.path.join(results_dir, out_name)
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved {save_path}")


def _trajectories_to_data_list(all_trajectories, plot_every=1, max_iteration=None):
    """Build list of (x, y, iteration) for learning dynamics plot (first hidden neuron w2_f).

    Data are ordered by training iteration first, then by weight init, so that for each
    iteration we plot all inits together (rather than all iterations per init).
    """
    data_list = []
    track_weights_every = 1

    # Find the maximum trajectory length across all inits
    max_len = 0
    weight_histories = []
    for traj in all_trajectories:
        gen_weights = np.asarray(traj["gen_weight_history"])  # (n_steps+1, 2)
        weight_histories.append(gen_weights)
        if gen_weights.shape[0] > max_len:
            max_len = gen_weights.shape[0]

    # Loop over training iterations first, then over inits
    for i in range(max_len):
        iteration = 0 if i == 0 else (i - 1) * track_weights_every
        if max_iteration is not None and iteration > max_iteration:
            break
        if i != 0 and iteration % plot_every != 0:
            continue

        for w in weight_histories:
            if i >= w.shape[0]:
                continue
            data_list.append(
                {
                    "x": float(w[i, 0]),
                    "y": float(w[i, 1]),
                    "Iteration": iteration,
                }
            )

    return data_list


def _seed_weight_histories_to_data_list(weight_histories, plot_every=1, max_iteration=None):
    """
    Given a list of single-run weight histories (one per seed), build a list of
    (x, y, iteration) points by looping over training iterations first, then
    over seeds.

    Each element of `weight_histories` is expected to be broadcastable to
    shape (T, 2) where the last dimension contains (w11^{2, f}, w12^{2, f}).
    """
    # Normalise shapes and track maximum trajectory length.
    norm_histories = []
    max_len = 0
    for w in weight_histories:
        arr = np.asarray(w)
        if arr.ndim >= 2 and arr.shape[-1] == 2 and arr.ndim > 2:
            arr = arr[:, 0, :]
        if arr.ndim != 2 or arr.shape[1] != 2:
            raise ValueError(f"Expected weight history of shape (T, 2), got {arr.shape}")
        norm_histories.append(arr)
        if arr.shape[0] > max_len:
            max_len = arr.shape[0]

    data_list = []
    for i in range(max_len):
        iteration = i
        if max_iteration is not None and iteration > max_iteration:
            break
        if i != 0 and iteration % plot_every != 0:
            continue
        for arr in norm_histories:
            if i >= arr.shape[0]:
                continue
            data_list.append(
                {
                    "x": float(arr[i, 0]),
                    "y": float(arr[i, 1]),
                    "Iteration": iteration,
                }
            )
    return data_list


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


# Shared relplot kwargs for learning/activity dynamics scatter plots.
_REL_PLOT_KW = dict(linewidth=0, alpha=0.5, aspect=1.15, edgecolor="face")


def _position_iteration_colorbar_top_right(
    ax, cbar_ax, width=0.03, height=1, pad_x=0.01, pad_y=0.0
):
    """
    Move seaborn's hue colorbar ("legend") to the top-right outside the main axes.

    We position in figure-relative coordinates using the main axes bbox.
    """
    bbox = ax.get_position()  # [left, bottom, width, height] in figure coords
    left = min(0.98 - width, bbox.x1 + pad_x)

    # Place the colorbar at a fixed top of the figure so it is always
    # aligned near the top-right corner (independent of seaborn's layout).
    top = 0.99 - pad_y
    bottom = max(0.02, top - height)

    cbar_ax.set_position([left, bottom, width, height])
    cbar_ax.set_ylabel("Iteration")


def plot_learning_dynamics(
    results_dir,
    plot_every=100,
    max_iteration=None,
    out_name_prefix="learning_dynamics",
):
    """
    Plot w2_f learning dynamics for each model in separate figures (same style as plot_toy_nonlinear).
    Saves one PDF per model: learning_dynamics_standard_pdm.pdf, etc.
    Expects all_trajectories.npy in results_dir/<model>/ from --grid_experiment run.
    """
    sns.set_theme(context="talk")
    os.makedirs(results_dir, exist_ok=True)
    for model_dir, label in zip(MODEL_DIRS, MODEL_LABELS):
        traj_path = os.path.join(results_dir, model_dir, "all_trajectories.npy")
        if not os.path.exists(traj_path):
            print(f"Skipping {model_dir} (no all_trajectories.npy)")
            continue
        all_trajectories = np.load(traj_path, allow_pickle=True)
        if len(all_trajectories) == 0:
            print(f"Skipping {model_dir} (empty trajectories)")
            continue
        data_list = _trajectories_to_data_list(
            all_trajectories, plot_every=plot_every, max_iteration=max_iteration
        )
        if not data_list:
            print(f"Skipping {model_dir} (no points)")
            continue
        df = pd.DataFrame(data_list)
        max_iter = max_iteration if max_iteration is not None else int(df["Iteration"].max())
        legend_iterations = np.linspace(0, max_iter, 5, dtype=int).tolist()
        # Same as plot_toy_nonlinear: relplot for identical color scheme and layout
        g = sns.relplot(data=df, x="x", y="y", hue="Iteration", **_REL_PLOT_KW)
        ax = g.axes[0, 0]
        _add_iteration_zero_black_borders(ax, df, facet_grid=g)
        ax.set_title(label, fontsize=30, pad=20)
        if len(g.figure.axes) > 1:
            cbar_ax = g.figure.axes[-1]
            cbar_ax.set_yticks(legend_iterations)
            cbar_ax.set_yticklabels([str(it) for it in legend_iterations])
            _position_iteration_colorbar_top_right(ax, cbar_ax)
        ax.set_xlabel(r"$w_{11}^{2, f}$", fontsize=30, labelpad=10)
        ax.set_ylabel(r"$w_{12}^{2, f}$", fontsize=30, labelpad=10)
        ax.set_xticks([-1, -0.5, 0, 0.5, 1])
        ax.set_yticks([-1, -0.5, 0, 0.5, 1])
        ax.set_xticklabels(["-1", "", "0", "", "1"])
        ax.set_yticklabels(["-1", "", "0", "", "1"])
        ax.tick_params(axis="both", labelsize=30)
        plot_path = os.path.join(results_dir, f"{out_name_prefix}_{model_dir}.pdf")
        plt.savefig(plot_path, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"Saved {plot_path}")


def plot_learning_dynamics_single_runs(
    results_dir,
    plot_every=1,
    max_iteration=None,
    out_name_prefix="learning_dynamics_seeds",
    n_seeds=None,
):
    """
    Plot w2_f learning dynamics for each model using SINGLE-RUN (seed_*) results.

    Expects, for each learning rate directory `results_dir`, files of the form
        results_dir/seed_*/<model>/gen_weight_history.npy.

    For every model where at least one such file exists, all seeds are combined
    into a single plot:
        learning_dynamics_seeds_<model>.pdf
    """
    sns.set_theme(context="talk")
    os.makedirs(results_dir, exist_ok=True)

    any_found_any_model = False
    for model_dir, label in zip(MODEL_DIRS, MODEL_LABELS):
        files = _seed_result_paths(
            results_dir, model_dir, "gen_weight_history.npy", n_seeds=n_seeds
        )
        if not files:
            continue

        weight_histories = []
        for fpath in files:
            weights = np.load(fpath)
            weight_histories.append(weights)

        try:
            data_list_all = _seed_weight_histories_to_data_list(
                weight_histories, plot_every=plot_every, max_iteration=max_iteration
            )
        except ValueError as e:
            print(f"Skipping {model_dir}: {e}")
            continue

        if not data_list_all:
            continue

        any_found_any_model = True
        df = pd.DataFrame(data_list_all)
        max_iter = (
            max_iteration
            if max_iteration is not None
            else int(df["Iteration"].max())
        )
        legend_iterations = np.linspace(0, max_iter, 5, dtype=int).tolist()

        g = sns.relplot(data=df, x="x", y="y", hue="Iteration", **_REL_PLOT_KW)
        ax = g.axes[0, 0]
        _add_iteration_zero_black_borders(ax, df, facet_grid=g)
        ax.set_title(label, fontsize=30, pad=20)
        if len(g.figure.axes) > 1:
            cbar_ax = g.figure.axes[-1]
            cbar_ax.set_yticks(legend_iterations)
            cbar_ax.set_yticklabels([str(it) for it in legend_iterations])
            _position_iteration_colorbar_top_right(ax, cbar_ax)
        ax.set_xlabel(r"$w_{11}^{2, f}$", fontsize=30, labelpad=10)
        ax.set_ylabel(r"$w_{12}^{2, f}$", fontsize=30, labelpad=10)
        ax.set_xticks([-1, -0.5, 0, 0.5, 1])
        ax.set_yticks([-1, -0.5, 0, 0.5, 1])
        ax.set_xticklabels(["-1", "", "0", "", "1"])
        ax.set_yticklabels(["-1", "", "0", "", "1"])
        ax.tick_params(axis="both", labelsize=30)

        plot_path = os.path.join(results_dir, f"{out_name_prefix}_{model_dir}.pdf")
        plt.savefig(plot_path, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"Saved {plot_path}")

    if not any_found_any_model:
        print(
            f"No single-run gen_weight_history.npy files found under {results_dir}/seed_*/<model>/; "
            "skipping single-run learning dynamics plots."
        )


def plot_activity_dynamics(
    results_dir,
    out_name_prefix="activity_dynamics",
    max_iteration=None,
    n_seeds=None,
):
    """
    Plot hidden-unit activity dynamics over training for each model, aggregating across seeds.

    For each training iteration t, we take the mean hidden activities over seeds and batches
    (x_1^2(t), x_2^2(t)) and plot them as points in the plane with color indicating
    the training iteration, analogous to the learning dynamics plots (but in activity space
    instead of weight space).

    Expects hidden_activity_history.npy saved under either:
      - results_dir/seed_*/<model>/hidden_activity_history.npy (multi-seed layout), or
      - results_dir/<model>/hidden_activity_history.npy (single run).

    For each model, saves one PDF: activity_dynamics_<model>.pdf.
    If max_iteration is not None, only iterations t <= max_iteration are plotted.
    """
    sns.set_theme(context="talk")
    os.makedirs(results_dir, exist_ok=True)

    for model_dir, label, color in zip(MODEL_DIRS, MODEL_LABELS, MODEL_COLORS):
        # Prefer seed-based aggregation, fall back to single-run layout.
        acts = _load_hidden_activities_from_seeds(
            results_dir, model_dir, n_seeds=n_seeds
        )
        if acts is None:
            acts = _load_hidden_activities_single_run(results_dir, model_dir)
        if acts is None:
            print(f"Skipping {model_dir} (no hidden_activity_history.npy found)")
            continue
        # acts: (n_seeds, T, H). Typically H = 2 for these experiments.
        n_seeds, T, H = acts.shape

        # Mean over seeds to get a single trajectory in activity space.
        mean_acts = acts.mean(axis=0)  # (T, H)

        # Optionally crop to max_iteration.
        if max_iteration is not None:
            T_eff = min(T, max_iteration + 1)
        else:
            T_eff = T

        # Build dataframe of (x, y, Iteration) similar to _trajectories_to_data_list.
        data_list = []
        for t in range(T_eff):
            if H == 1:
                x_val = float(mean_acts[t, 0])
                y_val = float(mean_acts[t, 0])
            else:
                x_val = float(mean_acts[t, 0])
                y_val = float(mean_acts[t, 1])
            data_list.append(
                {
                    "x": x_val,
                    "y": y_val,
                    "Iteration": t,
                }
            )
        if not data_list:
            print(f"Skipping {model_dir} (no activity points)")
            continue

        df = pd.DataFrame(data_list)
        max_iter = int(df["Iteration"].max())
        legend_iterations = np.linspace(0, max_iter, 5, dtype=int).tolist()

        # Use relplot for consistent style with learning dynamics.
        g = sns.relplot(data=df, x="x", y="y", hue="Iteration", **_REL_PLOT_KW)
        ax = g.axes[0, 0]
        _add_iteration_zero_black_borders(ax, df, facet_grid=g)
        ax.set_title(label, fontsize=30, pad=20)

        # Adjust colorbar ticks to show a few key iterations.
        if len(g.figure.axes) > 1:
            cbar_ax = g.figure.axes[-1]
            cbar_ax.set_yticks(legend_iterations)
            cbar_ax.set_yticklabels([str(it) for it in legend_iterations])
            _position_iteration_colorbar_top_right(ax, cbar_ax)

        ax.set_xlabel(r"$x_1^2$", fontsize=30, labelpad=10)
        ax.set_ylabel(r"$x_2^2$", fontsize=30, labelpad=10)
        ax.tick_params(axis="both", labelsize=25)
        ax.grid(True, alpha=0.3)

        plot_path = os.path.join(results_dir, f"{out_name_prefix}_{model_dir}.pdf")
        plt.savefig(plot_path, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"Saved {plot_path}")


def _has_grid_runs(data):
    """True if any model has multiple runs (grid experiment)."""
    for key in MODEL_DIRS:
        if data.get(key) is None:
            continue
        val = data[key]["val_losses"]
        if val.ndim > 1 and val.shape[0] > 1:
            return True
    return False


def _load_losses_over_lrs(results_root, n_seeds=None):
    """
    Load train/val losses aggregated over *seeds* for each learning rate and model.
    Expects directory layout:
        results_root/lr_*/seed_*/<model>/...
    and ignores any grid/non-seed layouts inside the lr_* directories.
    """
    lr_data = {key: {} for key in MODEL_DIRS}
    if not os.path.isdir(results_root):
        return lr_data
    for entry in sorted(os.listdir(results_root)):
        if not entry.startswith("lr_"):
            continue
        lr_str = entry[len("lr_") :]
        try:
            lr_val = float(lr_str)
        except ValueError:
            continue
        lr_dir = os.path.join(results_root, entry)
        if not os.path.isdir(lr_dir):
            continue
        # For each model, load ONLY seed-based runs under this lr directory.
        for key in MODEL_DIRS:
            seed_data = _load_losses_from_seeds(lr_dir, key, n_seeds=n_seeds)
            if seed_data is None:
                continue
            lr_data[key][lr_val] = seed_data
    return lr_data


def _compute_best_lrs_from_lrs(results_root, n_seeds=None):
    """
    For each model, compute:
      - best learning rate (lowest mean best val loss over runs)
      - mean best val loss per learning rate.
    """
    lr_data = _load_losses_over_lrs(results_root, n_seeds=n_seeds)
    best_val_means = {key: {} for key in MODEL_DIRS}
    best_lr = {}

    for key in MODEL_DIRS:
        for lr_val, entry in lr_data[key].items():
            val_losses = entry["val_losses"]  # shape (n_runs, T) or (T,)
            if val_losses.size == 0:
                continue
            if val_losses.ndim == 1:
                val_losses = val_losses[np.newaxis, :]
            best_per_run = val_losses.min(axis=1)  # best over time per run
            best_val_means[key][lr_val] = float(best_per_run.mean())

        if best_val_means[key]:
            best_lr[key] = min(
                best_val_means[key],
                key=lambda lr_: best_val_means[key][lr_],
            )

    return lr_data, best_lr, best_val_means


def plot_train_loss_best_lr_over_lrs(
    results_root,
    log_scale=False,
    use_median=False,
    n_seeds=None,
):
    """
    Plot train loss over training at the best learning rate per model,
    based on lr_* subdirs.

    If use_median is False (default): mean ± standard error over runs.
    If use_median is True: median with IQR (25th–75th percentile).
    Saved as best_train_losses.pdf.
    """
    sns.set_theme(context="talk")

    lr_data, best_lr, _ = _compute_best_lrs_from_lrs(results_root, n_seeds=n_seeds)
    if not best_lr:
        print(
            "No loss data found across learning rates. "
            "Run train_toy_nonlinear.py with lr sweeps first."
        )
        return

    fig, ax = plt.subplots(figsize=(7, 4))

    n_plotted = 0
    for model_dir, label, color in zip(MODEL_DIRS, MODEL_LABELS, MODEL_COLORS):
        if model_dir not in best_lr:
            continue
        lr_val = best_lr[model_dir]
        entry = lr_data[model_dir][lr_val]
        train_losses = entry["train_losses"]  # (n_runs, T) or (T,)
        if train_losses.size == 0:
            continue
        if train_losses.ndim == 1:
            train_losses = train_losses[np.newaxis, :]

        steps = np.arange(train_losses.shape[1])
        if use_median:
            center, q1, q3 = _median_iqr(train_losses)
            ax.plot(
                steps,
                center,
                color=color,
                label=label,
                linewidth=2,
            )
            ax.fill_between(
                steps,
                q1,
                q3,
                color=color,
                alpha=0.2,
            )
        else:
            n_runs = train_losses.shape[0]
            mean_loss = np.mean(train_losses, axis=0)
            stderr = np.std(train_losses, axis=0, ddof=1) / np.sqrt(max(n_runs, 1))
            ax.plot(
                steps,
                mean_loss,
                color=color,
                label=label,
                linewidth=2,
            )
            ax.fill_between(
                steps,
                mean_loss - stderr,
                mean_loss + stderr,
                color=color,
                alpha=0.2,
            )
        n_plotted += 1

    if n_plotted == 0:
        print(
            "No train loss data found across learning rates. "
            "Run train_toy_nonlinear.py with lr sweeps first."
        )
        return

    ax.set_xlabel("Training step", fontsize=30, labelpad=10)
    ax.set_ylabel("Training loss", fontsize=30, labelpad=10)
    if log_scale:
        ax.set_yscale("log")
    ax.tick_params(axis="both", labelsize=25)
    if ax.get_legend_handles_labels()[1]:
        ax.legend(fontsize=18)
    ax.grid(True, alpha=0.3)

    plot_path = os.path.join(results_root, "best_train_losses.pdf")
    plt.savefig(plot_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved {plot_path}")


def plot_val_loss_best_lr_over_lrs(
    results_root,
    print_every=100,
    log_scale=False,
    use_median=False,
    n_seeds=None,
):
    """
    Plot validation loss over training at the best learning rate per model,
    based on lr_* subdirs.

    If use_median is False (default): mean ± standard error over runs.
    If use_median is True: median with IQR (25th–75th percentile).
    Saved as best_val_losses.pdf.
    """
    sns.set_theme(context="talk")

    lr_data, best_lr, _ = _compute_best_lrs_from_lrs(results_root, n_seeds=n_seeds)
    if not best_lr:
        print(
            "No validation data found across learning rates. "
            "Run train_toy_nonlinear.py with lr sweeps first."
        )
        return

    fig, ax = plt.subplots(figsize=(7, 4))

    n_plotted = 0
    for model_dir, label, color in zip(MODEL_DIRS, MODEL_LABELS, MODEL_COLORS):
        if model_dir not in best_lr:
            continue
        lr_val = best_lr[model_dir]
        entry = lr_data[model_dir][lr_val]
        val_losses = entry["val_losses"]  # (n_runs, T_val) or (T_val,)
        train_losses = entry["train_losses"]
        if val_losses.size == 0:
            continue
        if val_losses.ndim == 1:
            val_losses = val_losses[np.newaxis, :]
        if train_losses.ndim == 1:
            train_losses = train_losses[np.newaxis, :]

        n_train_iters = _true_n_train_iters(entry, print_every)
        steps = test_steps_from_length(n_train_iters, print_every, val_losses.shape[1])

        if use_median:
            center, q1, q3 = _median_iqr(val_losses)
            ax.plot(
                steps,
                center,
                color=color,
                label=label,
                linewidth=2,
            )
            ax.fill_between(
                steps,
                q1,
                q3,
                color=color,
                alpha=0.2,
            )
        else:
            n_runs = val_losses.shape[0]
            mean_loss = np.mean(val_losses, axis=0)
            stderr = np.std(val_losses, axis=0, ddof=1) / np.sqrt(max(n_runs, 1))
            ax.plot(
                steps,
                mean_loss,
                color=color,
                label=label,
                linewidth=2,
            )
            ax.fill_between(
                steps,
                mean_loss - stderr,
                mean_loss + stderr,
                color=color,
                alpha=0.2,
            )
        n_plotted += 1

    if n_plotted == 0:
        print(
            "No validation loss data found across learning rates. "
            "Run train_toy_nonlinear.py with lr sweeps first."
        )
        return

    ax.set_xlabel("Training step", fontsize=30, labelpad=10)
    ax.set_ylabel("Validation loss", fontsize=30, labelpad=10)
    if log_scale:
        ax.set_yscale("log")
    ax.tick_params(axis="both", labelsize=25)
    if ax.get_legend_handles_labels()[1]:
        ax.legend(fontsize=18)
    ax.grid(True, alpha=0.3)

    plot_path = os.path.join(results_root, "best_val_losses.pdf")
    plt.savefig(plot_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved {plot_path}")


def plot_best_lr_loss_curves_over_lrs(
    results_root,
    print_every=100,
    log_scale=False,
    n_seeds=None,
):
    """
    Plot train/val loss (mean ± std error over runs) for the best learning rate
    of each model (Standard, Norm w2f, Norm w2f + lateral), based on lr_* subdirs.
    Saves a single PDF with one panel per model.
    """
    sns.set_theme(context="talk")

    lr_data, best_lr, _ = _compute_best_lrs_from_lrs(results_root, n_seeds=n_seeds)
    if not best_lr:
        print(
            "No loss data found across learning rates. "
            "Run train_toy_nonlinear.py with lr sweeps first."
        )
        return

    n_models_with_best = sum(1 for k in MODEL_DIRS if k in best_lr)
    fig, axes = plt.subplots(
        n_models_with_best,
        1,
        figsize=(8, 3 * n_models_with_best),
        sharex=True,
        constrained_layout=True,
    )
    if n_models_with_best == 1:
        axes = [axes]

    axis_idx = 0
    for model_dir, label, color in zip(MODEL_DIRS, MODEL_LABELS, MODEL_COLORS):
        if model_dir not in best_lr:
            continue
        ax = axes[axis_idx]
        axis_idx += 1

        lr_val = best_lr[model_dir]
        entry = lr_data[model_dir][lr_val]
        train_losses = entry["train_losses"]  # (n_runs, T_train) or (T_train,)
        val_losses = entry["val_losses"]  # (n_runs, T_val) or (T_val,)

        if train_losses.ndim == 1:
            train_losses = train_losses[np.newaxis, :]
        if val_losses.ndim == 1:
            val_losses = val_losses[np.newaxis, :]

        n_runs = train_losses.shape[0]

        train_mean = np.mean(train_losses, axis=0)
        train_stderr = np.std(train_losses, axis=0, ddof=1) / np.sqrt(max(n_runs, 1))
        train_steps = np.arange(len(train_mean))

        val_mean = np.mean(val_losses, axis=0)
        val_stderr = np.std(val_losses, axis=0, ddof=1) / np.sqrt(max(n_runs, 1))
        n_train_iters = _true_n_train_iters(entry, print_every)
        val_steps = test_steps_from_length(n_train_iters, print_every, len(val_mean))

        ax.plot(
            train_steps,
            train_mean,
            color=color,
            label=f"{label} train (mean)",
            linewidth=2,
        )
        ax.fill_between(
            train_steps,
            train_mean - train_stderr,
            train_mean + train_stderr,
            color=color,
            alpha=0.2,
        )

        ax.plot(
            val_steps,
            val_mean,
            color="black",
            label=f"{label} val (mean)",
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

        ax.set_ylabel("Loss", fontsize=30)
        ax.set_title(
            f"{label} (best lr = {lr_val:.1e})",
            fontsize=22,
        )
        ax.grid(True, alpha=0.3)
        ax.tick_params(axis="both", labelsize=9)
        if ax.get_legend_handles_labels()[1]:
            ax.legend(fontsize=18)

    axes[-1].set_xlabel("Training step", fontsize=30)

    plot_path = os.path.join(results_root, "best_lr_loss_curves.pdf")
    plt.savefig(plot_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved {plot_path}")


def plot_best_val_vs_lr_over_lrs(
    results_root,
    log_scale_y=False,
    n_seeds=None,
):
    """
    Plot best (lowest) validation performance versus learning rate for each model,
    based on lr_* subdirectories.
    """
    sns.set_theme(context="talk")

    _, _, best_val_means = _compute_best_lrs_from_lrs(results_root, n_seeds=n_seeds)
    if not any(best_val_means[key] for key in MODEL_DIRS):
        print(
            "No validation data found across learning rates. "
            "Run train_toy_nonlinear.py with lr sweeps first."
        )
        return

    fig, ax = plt.subplots(figsize=(7, 4))

    for model_dir, label, color in zip(MODEL_DIRS, MODEL_LABELS, MODEL_COLORS):
        if not best_val_means[model_dir]:
            continue
        lrs = sorted(best_val_means[model_dir].keys())
        vals = [best_val_means[model_dir][lr] for lr in lrs]
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

    plot_path = os.path.join(results_root, "best_val_vs_lr.pdf")
    plt.savefig(plot_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved {plot_path}")


def _run_plots_for_results_dir(results_dir, args):
    """
    Run all requested plots for a single results directory.

    Assumes that `results_dir` either contains:
    - seed_*/<model>/... (single runs), or
    - <model>/init_w11_*/init_w12_*/... (grid runs), or
    - <model>/... (single run without seeds).
    """
    plot_standard_pd = not args.no_standard_pd
    log_loss_scale = args.log_loss_scale
    max_steps = args.max_steps
    # Error-bar choice:
    # - If --use_median: median + IQR
    # - Else if --use_stderr OR no flag at all: mean ± standard error (default)
    # - (mean ± std is reserved for potential future options)
    use_median = args.use_median
    # Default to stderr when neither flag is set
    use_stderr = args.use_stderr or (not args.use_median and not args.use_stderr)

    os.makedirs(results_dir, exist_ok=True)

    data = load_losses(
        results_dir,
        warn_missing=not has_grid_trajectories(results_dir),
        n_seeds=args.n_seeds,
    )
    # For dynamics and grid runs, work relative to this results_dir (may already
    # be an lr_* subfolder).
    base_dir_for_dynamics = results_dir
    has_trajectories = has_grid_trajectories(base_dir_for_dynamics)
    dynamics_dir = get_dynamics_results_dir(base_dir_for_dynamics)  # top-level or grid_seed_*
    has_activities = has_activity_histories(results_dir)
    has_grid = _has_grid_runs(data)

    # Detect whether we have explicit seed_* subdirectories. When seeds exist,
    # we want loss plots to aggregate over seeds (mean/SEM) rather than over
    # grid initialisations, so we skip the single-init grid logic in that case.
    seed_pattern = os.path.join(results_dir, "seed_*")
    has_seed_dirs = any(os.path.isdir(d) for d in glob.glob(seed_pattern))

    # For loss curves:
    #   - If we have grid data but NO seed_* dirs: prefer a specific init
    #     (init_w11, init_w12), falling back to aggregating all inits.
    #   - If seed_* dirs exist: always aggregate over seeds via `data`.
    if has_grid and not has_seed_dirs:
        data_loss = load_losses_single_init(results_dir, init_w11=args.init_w11, init_w12=args.init_w12)
        if not any(v is not None for v in data_loss.values()):
            print(
                f"Warning: no data for single init ({args.init_w11}, {args.init_w12}) in {results_dir}, "
                "falling back to aggregating all initialisations for loss plots."
            )
            data_loss = data
    else:
        data_loss = data

    # Plot dynamics from grid trajectories whenever we have them (or user asked --dynamics)
    if (args.dynamics or has_trajectories) and dynamics_dir is not None:
        plot_learning_dynamics(
            dynamics_dir,
            plot_every=args.plot_every,
            max_iteration=args.max_iteration,
        )

    # Additionally, plot learning dynamics from single-run (seed_*) weight histories
    # by default whenever seed_* directories exist, or when explicitly requested.
    if args.single_run_dynamics or has_seed_dirs:
        plot_learning_dynamics_single_runs(
            results_dir,
            plot_every=args.plot_every,
            max_iteration=args.max_iteration,
            n_seeds=args.n_seeds,
        )

    # Plot activity dynamics over training whenever we have activity histories
    # (or user explicitly requested it).
    if args.activity_dynamics or has_activities:
        plot_activity_dynamics(
            results_dir,
            max_iteration=args.max_iteration,
            n_seeds=args.n_seeds,
        )

    if all(v is None for v in data.values()):
        if has_trajectories:
            print(
                f"No loss data found in {results_dir}. "
                "Plotted learning dynamics. For loss curves, run train_toy_nonlinear.py without --grid_experiment."
            )
        else:
            print(f"No data found in {results_dir}. Run train_toy_nonlinear.py first (single run or --grid_experiment).")
        return

    # Note: we intentionally do not plot per-lr train/val loss curves here.
    # All loss visualisations are handled by the summary best-* plots over lrs.

def main():
    parser = argparse.ArgumentParser(description="Plot three-model comparison (Standard PDM, Unit-norm PDM, lateral pdm).")
    parser.add_argument("--results_dir", type=str, default="results/toy_nonlinear",
                        help="Directory containing standard_pdm/, unit_norm_pdm/, lateral_pdm/ (or lr_<lr>/seed_*/ when --lr set).")
    parser.add_argument("--lr", type=float, default=None,
                        help="Learning rate: plot from results_dir/lr_<lr>/ (omit to use top-level results_dir, e.g. seed_*/ or single run).")
    parser.add_argument("--print_every", type=int, default=100,
                        help="Print/eval interval used during training (for test loss x-axis).")
    parser.add_argument("--init_w11", type=float, default=1.0,
                        help="init_w11 for single-init test loss plot (grid experiment only).")
    parser.add_argument("--init_w12", type=float, default=1.0,
                        help="init_w12 for single-init test loss plot (grid experiment only).")
    parser.add_argument("--train_only", action="store_true", help="Plot only train loss.")
    parser.add_argument("--test_only", action="store_true", help="Plot only test loss.")
    parser.add_argument("--no_standard_pd", action="store_true", default=False)
    parser.add_argument("--log_loss_scale", action="store_true", default=True,
                        help="Use log scale for loss y-axis (default: linear).")
    parser.add_argument("--max_steps", type=int, default=None,
                        help="Maximum training step to show in loss plots (crop curves).")
    parser.add_argument("--dynamics", action="store_true",
                        help="Plot learning dynamics (w2_f trajectories). Requires --grid_experiment run.")
    parser.add_argument("--plot_every", type=int, default=10,
                        help="Subsample trajectory points by this many iterations (for dynamics plot).")
    parser.add_argument("--max_iteration", type=int, default=1000,
                        help="Max training iteration to show in dynamics plot (default: all).")
    parser.add_argument("--activity_dynamics", action="store_true",
                        help="Plot hidden activity dynamics over training for each model (non-grid/single-run experiments).")
    parser.add_argument(
        "--single_run_dynamics",
        action="store_true",
        help=(
            "Plot learning dynamics in weight space from single-run (seed_*) results, "
            "using gen_weight_history.npy under seed_*/<model>/."
        ),
    )
    parser.add_argument("--use_median", action="store_true",
                        help="Plot median with IQR (25th–75th percentile) instead of mean ± 1 std for train/test losses.")
    parser.add_argument("--use_stderr", action="store_true",
                        help="Plot mean ± standard error of the mean (SEM) instead of mean ± 1 std for train/test losses.")
    parser.add_argument(
        "--n_seeds",
        type=int,
        default=None,
        help=(
            "Number of seeds to aggregate over (uses seed_0 .. seed_{n_seeds-1}). "
            "Default: all seed_* directories found."
        ),
    )
    args = parser.parse_args()

    # Determine which results directories to process.
    # If --lr is provided, only process that learning rate under results_dir/lr_<lr>/.
    # Otherwise, if subfolders results_dir/lr_* exist, process all of them. As a
    # fallback (for older runs), process results_dir directly.
    if args.lr is not None:
        results_dirs = [os.path.join(args.results_dir, f"lr_{args.lr}")]
    else:
        lr_pattern = os.path.join(args.results_dir, "lr_*")
        lr_dirs = sorted(d for d in glob.glob(lr_pattern) if os.path.isdir(d))
        if lr_dirs:
            results_dirs = lr_dirs
        else:
            results_dirs = [args.results_dir]

    # Note: aggregation over seed_* directories for each lr_* is handled inside
    # load_losses() via _load_losses_from_seeds, which returns (n_seeds, n_steps)
    # arrays so that mean and standard error can be plotted.
    for rd in results_dirs:
        print(f"\nProcessing nonlinear toy results in {rd}")
        _run_plots_for_results_dir(rd, args)

    # If we have multiple learning rates under args.results_dir, also create
    # summary plots that select the best lr per model (based on best val loss)
    # and show:
    #   - best train loss learning curves at the best lr for each model
    #   - best val loss learning curves at the best lr for each model
    #   - best val loss as a function of learning rate.
    if any(os.path.isdir(d) for d in glob.glob(os.path.join(args.results_dir, "lr_*"))):
        print("\nPlotting best train losses across learning rates...")
        plot_train_loss_best_lr_over_lrs(
            args.results_dir,
            log_scale=args.log_loss_scale,
            use_median=args.use_median,
            n_seeds=args.n_seeds,
        )
        print("\nPlotting best val losses across learning rates...")
        plot_val_loss_best_lr_over_lrs(
            args.results_dir,
            print_every=args.print_every,
            log_scale=args.log_loss_scale,
            use_median=args.use_median,
            n_seeds=args.n_seeds,
        )
        print("\nPlotting best val. loss versus learning rate across learning rates...")
        plot_best_val_vs_lr_over_lrs(
            args.results_dir,
            log_scale_y=args.log_loss_scale,
            n_seeds=args.n_seeds,
        )


if __name__ == "__main__":
    main()
