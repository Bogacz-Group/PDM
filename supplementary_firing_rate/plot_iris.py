import os
import glob
import argparse

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


# Matches the modes in train_iris.py
MODE_NAMES = ["no_lateral", "with_lateral", "backprop"]

# Use the same textual legend entries as in plot_toy_nonlinear.py
MODE_LABELS = {
    "no_lateral": r"Norm. $w^{2, f}$",
    "with_lateral": r"Norm. $w^{2, f}$ + lateral",
    "backprop": "Backprop",
}

MODE_COLORS = {
    "no_lateral": "#d62728",
    "with_lateral": "#2ca02c",
    "backprop": "#9467bd",
}

# Distinct series colors for multi-model/thetaf comparisons.
SERIES_COLORS = [
    "#1f77b4",  # blue
    "#ff7f0e",  # orange
    "#2ca02c",  # green
    "#d62728",  # red
    "#9467bd",  # purple
    "#8c564b",  # brown
    "#e377c2",  # pink
    "#7f7f7f",  # gray
    "#bcbd22",  # olive
    "#17becf",  # cyan
]

# Keep a stable color mapping across all generated plots in one run.
GLOBAL_SERIES_COLOR_MAP = {}

# Match the toy-linear/nonlinear palette for key series.
# We key by (norm_name, mode, thetaf) so "Standard" (no_norm_fwd) can use a distinct
# palette from the normalised (norm_fwd) series.
#
# - Toy "Norm w^{2,f}" uses matplotlib red: #d62728
# - Toy "Norm w^{2,f} + lateral" uses matplotlib green: #2ca02c
# - Toy "Standard" uses matplotlib blue: #1f77b4
# Additionally reserve orange for theta_f=0.95 to make it stand out (norm_fwd, no_lateral).
RESERVED_SERIES_COLORS: dict[tuple[str | None, str, float | None], str] = {
    ("norm_fwd", "no_lateral", 0.5): "#d62728",
    ("norm_fwd", "with_lateral", 0.5): "#2ca02c",
    ("norm_fwd", "no_lateral", 0.95): "#ff7f0e",
    ("no_norm_fwd", "no_lateral", 0.5): "#1f77b4",
    # Nonlinear mapping highlight (kept purple as before).
    ("norm_fwd", "with_lateral", 1.0): "#9467bd",
    # Standard thetaf=0.99 should be orange.
    ("no_norm_fwd", "no_lateral", 0.99): "#ff7f0e",
}

# Match typography used in toy linear/nonlinear plots.
AXIS_LABEL_FONTSIZE = 35
TICK_LABEL_FONTSIZE = 30
LEGEND_FONTSIZE = 18
METRIC_PLOT_WIDTH = 16
METRIC_PLOT_HEIGHT = 4
# Shared y-axis floor for all accuracy plots (curves, bars, vs-lr).
ACC_PLOT_YMIN = 0.6
# Selected-model bar plots should show a bit more range.
SELECTED_ACC_PLOT_YMIN = 0.5
# Bar plots (best-* summaries) should be tall enough for rotated labels.
BAR_PLOT_HEIGHT = 4.0
# Keep bar plots relatively narrow even with many labels.
BAR_PLOT_MIN_WIDTH = 10.0
BAR_PLOT_WIDTH_PER_LABEL = 1.45
# Selected-model bar plots: give extra horizontal space.
SELECTED_BAR_PLOT_MIN_WIDTH = 8.0
SELECTED_BAR_PLOT_WIDTH_PER_LABEL = 1.5

# Styling: bar-plot error bars and jittered dots.
BAR_ERRORBAR_LINEWIDTH = 1.0
BAR_ERRORBAR_CAPTHICK = 1.0
JITTER_DOT_EDGE_COLOR = "white"
JITTER_DOT_EDGE_WIDTH = 1.25
# Small padding so jittered dots near limits aren't clipped.
BAR_PLOT_Y_PADDING_FRAC = 0.02

# w2f_data_pc_projections.pdf: fixed PC1/PC2 view with sparse ticks.
W2F_DATA_PC_AXIS_LIM = 1.75
W2F_DATA_PC_AXIS_TICKS = (-3, 0, 3)
W2F_DATA_PC_AUTOSCALE_PAD_FRAC = 0.08


def _autoscale_xy_limits(points2: np.ndarray, *, pad_frac: float = W2F_DATA_PC_AUTOSCALE_PAD_FRAC):
    """
    Compute symmetric x/y limits around 0 from a (N,2) cloud.
    Returns (lim, xticks, yticks) where ticks may be None.
    """
    pts = np.asarray(points2, dtype=np.float32)
    if pts.ndim != 2 or pts.shape[1] != 2:
        return None, None, None
    pts = pts[np.isfinite(pts).all(axis=1)]
    if pts.size == 0:
        return None, None, None
    max_abs = float(np.max(np.abs(pts)))
    if not np.isfinite(max_abs) or max_abs <= 0:
        return None, None, None
    lim = max_abs * (1.0 + float(pad_frac))
    # For autoscale, keep ticks simple: -lim, 0, lim (rounded a bit).
    tick = float(np.round(lim, 2))
    ticks = (-tick, 0.0, tick)
    return lim, ticks, ticks


def _safe_load_npz(path: str) -> dict | None:
    try:
        with np.load(path, allow_pickle=False) as f:
            return {k: f[k] for k in f.files}
    except Exception:
        return None


def _scatter_2d(ax, Z2: np.ndarray, y: np.ndarray, title: str):
    Z2 = np.asarray(Z2)
    y = np.asarray(y)
    if Z2.ndim != 2 or Z2.shape[1] != 2 or Z2.shape[0] != y.shape[0]:
        ax.set_title(f"{title} (missing)")
        ax.axis("off")
        return
    if not np.isfinite(Z2).all():
        ax.set_title(f"{title} (nan)")
        ax.axis("off")
        return
    ax.scatter(
        Z2[:, 0],
        Z2[:, 1],
        c=y,
        cmap="tab10",
        s=40,
        alpha=0.9,
        edgecolors="white",
        linewidths=0.5,
    )
    ax.set_title(title)
    ax.grid(True, alpha=0.25)
    ax.set_xlabel("dim 1")
    ax.set_ylabel("dim 2")


def plot_separability_run(run_dir: str, *, w2f_data_pc_autoscale: bool = False) -> bool:
    """
    Create projection/weight-space PDFs inside a single run directory
    containing separability_metrics.npz.
    """
    npz_path = os.path.join(run_dir, "separability_metrics.npz")
    data = _safe_load_npz(npz_path)
    if data is None:
        return False

    model = str(data.get("model", np.array("unknown")).item())
    y = data.get("y_val")
    if y is None:
        return False
    y = np.asarray(y, dtype=np.int64)

    sns.set_theme(context="talk")

    # Input data projected onto first 2 PCs (train set).
    x_train_pca2 = data.get("x_train_pca2")
    y_train = data.get("y_train")
    Xt = None
    if x_train_pca2 is not None:
        Xt = np.asarray(x_train_pca2, dtype=np.float32)
        if Xt.ndim != 2 or Xt.shape[1] != 2 or not np.isfinite(Xt).all():
            Xt = None
    yt = None
    if y_train is not None:
        yt = np.asarray(y_train, dtype=np.int64).reshape(-1)
        if Xt is None or yt.shape[0] != Xt.shape[0]:
            yt = None

    if Xt is not None:
        fig, ax = plt.subplots(figsize=(7.5, 6.5))
        if yt is None:
            ax.scatter(
                Xt[:, 0],
                Xt[:, 1],
                s=45,
                c="0.4",
                alpha=0.7,
                edgecolors="white",
                linewidths=0.4,
            )
        else:
            palette = np.array(["#1f77b4", "#d62728", "#2ca02c"], dtype=object)  # blue, red, green
            yy = np.clip(yt.astype(int, copy=False), 0, palette.size - 1)
            ax.scatter(
                Xt[:, 0],
                Xt[:, 1],
                s=55,
                c=palette[yy],
                alpha=0.75,
                edgecolors="white",
                linewidths=0.4,
            )
        ax.grid(True, alpha=0.25)
        ax.set_xlabel("PC1", fontsize=AXIS_LABEL_FONTSIZE, labelpad=10)
        ax.set_ylabel("PC2", fontsize=AXIS_LABEL_FONTSIZE, labelpad=10)
        ax.tick_params(axis="both", labelsize=TICK_LABEL_FONTSIZE)
        plt.tight_layout()
        plt.savefig(os.path.join(run_dir, "data_pca2.pdf"), dpi=150, bbox_inches="tight")
        plt.close()

    # 2D projections.
    if model == "pdm":
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        _scatter_2d(axes[0, 0], data.get("pca2_fwd"), y, "PCA2 (fwd)")
        _scatter_2d(axes[0, 1], data.get("lda2_fwd"), y, "LDA2 (fwd)")
        _scatter_2d(axes[1, 0], data.get("pca2_relaxed"), y, "PCA2 (relaxed)")
        _scatter_2d(axes[1, 1], data.get("lda2_relaxed"), y, "LDA2 (relaxed)")
        fig.suptitle("Hidden activity projections")
        plt.tight_layout(rect=(0, 0, 1, 0.97))
    else:
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        _scatter_2d(axes[0], data.get("pca2"), y, "PCA2")
        _scatter_2d(axes[1], data.get("lda2"), y, "LDA2")
        fig.suptitle("Hidden activity projections")
        plt.tight_layout(rect=(0, 0, 1, 0.95))
    plt.savefig(os.path.join(run_dir, "activity_projections.pdf"), dpi=150, bbox_inches="tight")
    plt.close()

    # Input->hidden weights projected onto first 2 PCs of the *input data*.
    w2f_data_pca2 = data.get("w2f_data_pca2")
    w2f_data_pca2_steps = data.get("w2f_data_pca2_steps")
    data_pca_evr = data.get("data_pca_explained_var_ratio")
    if w2f_data_pca2 is not None and w2f_data_pca2_steps is not None:
        Wp = np.asarray(w2f_data_pca2, dtype=np.float32)  # (S, H, 2)
        steps = np.asarray(w2f_data_pca2_steps, dtype=np.int64)  # (S,)
        if Wp.ndim == 3 and Wp.shape[-1] == 2 and steps.ndim == 1 and steps.shape[0] == Wp.shape[0]:
            evr = None
            if data_pca_evr is not None:
                evr = np.asarray(data_pca_evr, dtype=np.float32).reshape(-1)
                if evr.size < 2 or not np.isfinite(evr[:2]).all():
                    evr = None

            # Single-panel learning-dynamics style plot (match plot_toy_linear):
            # use seaborn hue->color scale axis (not plt.colorbar).
            S = int(Wp.shape[0])
            H = int(Wp.shape[1])
            pts = Wp.reshape(S * H, 2)
            iters = np.repeat(steps.astype(np.int64), H)

            import pandas as pd

            df = pd.DataFrame(
                {
                    "x": pts[:, 0].astype(np.float32),
                    "y": pts[:, 1].astype(np.float32),
                    "Iteration": iters,
                }
            )

            max_iter = int(df["Iteration"].max()) if len(df) else 0
            legend_iterations = np.linspace(0, max_iter, 5, dtype=int).tolist()

            sns.set_theme(context="talk")
            g = sns.relplot(
                data=df,
                x="x",
                y="y",
                hue="Iteration",
                linewidth=0,
                alpha=0.5,
                aspect=1.15,
            )
            ax = g.axes[0, 0]

            if Xt is not None:
                # Overlay the (train) data in the same PC basis.
                # Put it ABOVE the weight dots so it's always visible.
                if yt is None:
                    ax.scatter(
                        Xt[:, 0],
                        Xt[:, 1],
                        s=38,
                        c="0.6",
                        alpha=0.32,
                        edgecolors="none",
                        zorder=3,
                    )
                else:
                    # Explicit 3-class palette (Iris): blue, red, green.
                    # Keep it discrete (not a continuous colormap) so all classes are distinct.
                    palette = np.array(["#1f77b4", "#d62728", "#2ca02c"], dtype=object)
                    yy = yt.astype(int, copy=False)
                    yy = np.clip(yy, 0, palette.size - 1)
                    point_colors = palette[yy]
                    ax.scatter(
                        Xt[:, 0],
                        Xt[:, 1],
                        s=44,
                        c=point_colors,
                        alpha=0.38,
                        edgecolors="white",
                        linewidths=0.4,
                        zorder=3,
                    )

            # Style the hue scale axis like plot_toy_linear.
            if len(g.figure.axes) > 1:
                cbar_ax = g.figure.axes[-1]
                cbar_ax.set_yticks(legend_iterations)
                cbar_ax.set_yticklabels([str(it) for it in legend_iterations])
                cbar_ax.set_ylabel("Iteration")

            ax.grid(True, alpha=0.25)
            ax.set_xlabel("PC1", fontsize=AXIS_LABEL_FONTSIZE, labelpad=10)
            ax.set_ylabel("PC2", fontsize=AXIS_LABEL_FONTSIZE, labelpad=10)
            ax.tick_params(axis="both", labelsize=TICK_LABEL_FONTSIZE)
            if w2f_data_pc_autoscale:
                # Let matplotlib choose limits based on the plotted points.
                # (No manual set_xlim/set_ylim, no forced ticks.)
                ax.relim()
                ax.autoscale(enable=True, axis="both", tight=False)
                ax.autoscale_view()
            else:
                lim = W2F_DATA_PC_AXIS_LIM
                ax.set_xlim(-lim, lim)
                ax.set_ylim(-lim, lim)
                ax.set_xticks(W2F_DATA_PC_AXIS_TICKS)
                ax.set_yticks(W2F_DATA_PC_AXIS_TICKS)

            # No title (user requested).
            plt.savefig(os.path.join(run_dir, "w2f_data_pc_projections.pdf"), dpi=150, bbox_inches="tight")
            plt.close()

            # Also save final-step weights only (more comparable to dissertation-style snapshots).
            W_init = np.asarray(Wp[0], dtype=np.float32)    # (H, 2)
            W_final = np.asarray(Wp[-1], dtype=np.float32)  # (H, 2)

            fig, ax = plt.subplots(figsize=(7.5, 6.5))
            # Background data (optional).
            if Xt is not None:
                if yt is None:
                    ax.scatter(
                        Xt[:, 0], Xt[:, 1],
                        s=40, c="0.75", alpha=0.35,
                        edgecolors="none",
                        zorder=0,
                    )
                else:
                    palette = np.array(["#1f77b4", "#d62728", "#2ca02c"], dtype=object)
                    yy = np.clip(yt.astype(int, copy=False), 0, palette.size - 1)
                    ax.scatter(
                        Xt[:, 0], Xt[:, 1],
                        s=48, c=palette[yy], alpha=0.30,
                        edgecolors="white", linewidths=0.3,
                        zorder=0,
                    )

            ax.scatter(
                W_final[:, 0],
                W_final[:, 1],
                s=55,
                c="black",
                alpha=0.75,
                edgecolors="white",
                linewidths=0.4,
                zorder=2,
            )
            ax.grid(True, alpha=0.25)
            ax.set_xlabel("PC1", fontsize=AXIS_LABEL_FONTSIZE, labelpad=10)
            ax.set_ylabel("PC2", fontsize=AXIS_LABEL_FONTSIZE, labelpad=10)
            ax.tick_params(axis="both", labelsize=TICK_LABEL_FONTSIZE)
            if not w2f_data_pc_autoscale:
                lim = W2F_DATA_PC_AXIS_LIM
                ax.set_xlim(-lim, lim)
                ax.set_ylim(-lim, lim)
                ax.set_xticks(W2F_DATA_PC_AXIS_TICKS)
                ax.set_yticks(W2F_DATA_PC_AXIS_TICKS)
            plt.tight_layout()
            plt.savefig(os.path.join(run_dir, "w2f_data_pc_final.pdf"), dpi=150, bbox_inches="tight")
            plt.close()

            # And an init->final arrow plot (one arrow per hidden unit).
            fig, ax = plt.subplots(figsize=(7.5, 6.5))
            if Xt is not None:
                if yt is None:
                    ax.scatter(
                        Xt[:, 0], Xt[:, 1],
                        s=35, c="0.75", alpha=0.30,
                        edgecolors="none",
                        zorder=0,
                    )
                else:
                    palette = np.array(["#1f77b4", "#d62728", "#2ca02c"], dtype=object)
                    yy = np.clip(yt.astype(int, copy=False), 0, palette.size - 1)
                    ax.scatter(
                        Xt[:, 0], Xt[:, 1],
                        s=42, c=palette[yy], alpha=0.26,
                        edgecolors="white", linewidths=0.25,
                        zorder=0,
                    )
            dW = W_final - W_init
            ax.quiver(
                W_init[:, 0],
                W_init[:, 1],
                dW[:, 0],
                dW[:, 1],
                angles="xy",
                scale_units="xy",
                scale=1.0,
                width=0.003,
                alpha=0.65,
                color="black",
                zorder=2,
            )
            ax.scatter(
                W_init[:, 0], W_init[:, 1],
                s=22, c="black", alpha=0.35,
                edgecolors="none",
                zorder=1,
            )
            ax.scatter(
                W_final[:, 0], W_final[:, 1],
                s=28, c="black", alpha=0.85,
                edgecolors="white", linewidths=0.3,
                zorder=3,
            )
            ax.grid(True, alpha=0.25)
            ax.set_xlabel("PC1", fontsize=AXIS_LABEL_FONTSIZE, labelpad=10)
            ax.set_ylabel("PC2", fontsize=AXIS_LABEL_FONTSIZE, labelpad=10)
            ax.tick_params(axis="both", labelsize=TICK_LABEL_FONTSIZE)
            if not w2f_data_pc_autoscale:
                lim = W2F_DATA_PC_AXIS_LIM
                ax.set_xlim(-lim, lim)
                ax.set_ylim(-lim, lim)
                ax.set_xticks(W2F_DATA_PC_AXIS_TICKS)
                ax.set_yticks(W2F_DATA_PC_AXIS_TICKS)
            plt.tight_layout()
            plt.savefig(os.path.join(run_dir, "w2f_data_pc_init_to_final.pdf"), dpi=150, bbox_inches="tight")
            plt.close()

    return True


def plot_separability_all_runs(
    results_dir: str, norm_name: str | None = None, *, w2f_data_pc_autoscale: bool = False
):
    """
    Scan results tree and generate per-run separability PDFs anywhere a
    separability_metrics.npz is present.
    """
    resolved = _resolve_results_dir(results_dir, norm_name=norm_name)
    pattern = os.path.join(resolved, "**", "separability_metrics.npz")
    paths = glob.glob(pattern, recursive=True)
    n_ok = 0
    for i, p in enumerate(paths):
        if i % 100 == 0:
            print(f"[separability] {i}/{len(paths)}: {os.path.dirname(p)}")
        if plot_separability_run(os.path.dirname(p), w2f_data_pc_autoscale=w2f_data_pc_autoscale):
            n_ok += 1
    print(f"Generated separability plots for {n_ok}/{len(paths)} runs under {resolved}")


def _resolve_results_dir(results_dir: str, norm_name: str | None = None) -> str:
    """
    Resolve `results_dir` for the current on-disk layout.

    New (train_iris.py) layout:
      results_dir/
        norm_fwd/ or no_norm_fwd/
          no_lateral/ with_lateral/ ...

    Old layout:
      results_dir/
        no_lateral/ with_lateral/ ...
    """
    # If caller already points at a directory containing mode subdirs, keep it.
    if os.path.isdir(os.path.join(results_dir, "no_lateral")):
        return results_dir

    candidates = []
    if norm_name is not None:
        candidates.append(norm_name)
    # Prefer norm_fwd if present.
    candidates.extend(["norm_fwd", "no_norm_fwd"])

    for name in candidates:
        candidate = os.path.join(results_dir, name)
        if os.path.isdir(os.path.join(candidate, "no_lateral")):
            return candidate
    return results_dir


def _norm_root_and_name(results_dir: str) -> tuple[str, str | None]:
    """
    If `results_dir` points at .../(norm_fwd|no_norm_fwd), return (parent, norm_name).
    Otherwise return (results_dir, None).
    """
    base = os.path.basename(os.path.normpath(results_dir))
    if base in ("norm_fwd", "no_norm_fwd"):
        return os.path.dirname(os.path.normpath(results_dir)), base
    return results_dir, None


def _offdiag_values(mat: np.ndarray) -> np.ndarray:
    """Flatten the off-diagonal entries of a square matrix."""
    mat = np.asarray(mat)
    if mat.ndim != 2 or mat.shape[0] != mat.shape[1]:
        raise ValueError(f"Expected square 2D matrix, got shape={mat.shape}")
    n = mat.shape[0]
    mask = ~np.eye(n, dtype=bool)
    return np.asarray(mat[mask], dtype=float).reshape(-1)


def _load_final_lateral_values_from_run_dir(run_dir: str) -> np.ndarray | None:
    """
    Return the final (end-of-training) off-diagonal lateral weights for one run dir.
    Expects `tracked_lateral_weights.npy` saved by train_iris.py when use_lateral=True.
    """
    path = os.path.join(run_dir, "tracked_lateral_weights.npy")
    if not os.path.isfile(path):
        return None
    arr = np.load(path)
    if arr.ndim != 3 or arr.shape[-1] != arr.shape[-2]:
        return None
    return _offdiag_values(arr[-1])


def _discover_lateral_run_dirs_for_series(
    results_dir: str,
    *,
    width: int = None,
    thetaf: float = None,
    lr: float | None = None,
    best_selected: bool = False,
) -> list[str]:
    """
    Return per-seed run directories for the with_lateral series.

    - best_selected=True: .../seed_*/best_lr_*/lateral_pdm/
    - lr provided (legacy): .../lr_<lr>/seed_*/lateral_pdm/
    - lr provided (gridsearch layout): .../seed_*/best_lr_<lr>/lateral_pdm/
    """
    mode_root = _mode_base_dir(results_dir, "with_lateral", width=width, thetaf=thetaf)
    if not os.path.isdir(mode_root):
        return []

    if best_selected:
        pattern = os.path.join(mode_root, "seed_*", "best_lr_*", "lateral_pdm")
        return sorted(d for d in glob.glob(pattern) if os.path.isdir(d))

    if lr is None:
        return []

    legacy_lr_dir = os.path.join(mode_root, f"lr_{lr}")
    if os.path.isdir(legacy_lr_dir):
        pattern = os.path.join(legacy_lr_dir, "seed_*", "lateral_pdm")
        return sorted(d for d in glob.glob(pattern) if os.path.isdir(d))

    # GridSearchCV layout: seed_*/best_lr_<lr>/lateral_pdm
    pattern = os.path.join(mode_root, "seed_*", "best_lr_*", "lateral_pdm")
    out = []
    for d in sorted(glob.glob(pattern)):
        if not os.path.isdir(d):
            continue
        best_lr_dir = os.path.dirname(d)  # .../best_lr_<lr>
        base = os.path.basename(best_lr_dir)
        if not base.startswith("best_lr_"):
            continue
        try:
            chosen_lr = float(base.replace("best_lr_", ""))
        except ValueError:
            continue
        if np.isclose(chosen_lr, lr):
            out.append(d)
    return out


def plot_lateral_weight_distributions(
    data: dict,
    *,
    results_dir: str,
    lr,
    width: int = None,
    thetafs: list | None = None,
    save_base_dir: str | None = None,
    source_results_dir_by_series_key: dict[str, str] | None = None,
):
    """
    Plot KDEs of learned lateral weights at the end of training (off-diagonal only).

    For each applicable series (with_lateral), draws one KDE per seed (faint) plus a
    pooled KDE (bold) for that series.
    """
    sns.set_theme(context="talk")
    fig, ax = plt.subplots(figsize=(METRIC_PLOT_WIDTH, METRIC_PLOT_HEIGHT))

    any_plotted = False
    series_colors = {}

    # IMPORTANT: iterate in the same insertion order as the other plots in this file
    # so the global color mapping matches across figures (for non-reserved series).
    for series_key, entry in data.items():
        if entry.get("mode") != "with_lateral":
            continue

        series_results_dir = (
            source_results_dir_by_series_key.get(series_key, results_dir)
            if source_results_dir_by_series_key is not None
            else results_dir
        )

        thetaf = entry.get("thetaf", None)
        norm_name = entry.get("norm_name", None)
        style = _series_style("with_lateral", thetaf, norm_name, series_colors)

        best_selected = isinstance(lr, str) and lr == "gridsearch_best"
        run_dirs = _discover_lateral_run_dirs_for_series(
            series_results_dir,
            width=width,
            thetaf=thetaf,
            lr=None if best_selected else float(lr),
            best_selected=best_selected,
        )
        if not run_dirs:
            continue

        per_seed_vals = []
        for run_dir in run_dirs:
            vals = _load_final_lateral_values_from_run_dir(run_dir)
            if vals is None or vals.size == 0:
                continue
            per_seed_vals.append(vals)
            sns.kdeplot(
                x=vals,
                ax=ax,
                color=style["color"],
                alpha=0.18,
                linewidth=1.0,
                fill=False,
            )

        if not per_seed_vals:
            continue

        pooled = np.concatenate([np.ravel(v) for v in per_seed_vals]).astype(float, copy=False)
        pooled = pooled[np.isfinite(pooled)]
        if pooled.size == 0:
            continue

        sns.kdeplot(
            x=pooled,
            ax=ax,
            color=style["color"],
            alpha=0.95,
            linewidth=2.5,
            fill=False,
            label=style["label"],
        )
        any_plotted = True

    if not any_plotted:
        plt.close()
        return

    ax.set_xlabel("Lateral weights (off-diagonal)", fontsize=AXIS_LABEL_FONTSIZE, labelpad=10)
    ax.set_ylabel("Density", fontsize=AXIS_LABEL_FONTSIZE, labelpad=10)
    ax.tick_params(axis="both", labelsize=TICK_LABEL_FONTSIZE)
    ax.grid(True, alpha=0.3)
    ax.legend(
        loc="upper left",
        bbox_to_anchor=(1.02, 1.0),
        borderaxespad=0.0,
        fontsize=LEGEND_FONTSIZE,
    )
    plt.tight_layout(rect=(0, 0, 0.8, 1))

    base = save_base_dir if save_base_dir is not None else _summary_save_base(results_dir, width=width)
    save_dir = os.path.join(base, f"lr_{lr}")
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, "lateral_weight_distribution.pdf")
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved {save_path}")


def _discover_norm_dirs(results_dir: str, requested: str | None = None) -> list[tuple[str, str | None]]:
    """
    Return list of (resolved_results_dir, norm_name) to plot.

    - If `results_dir` already points at a directory containing mode subdirs, return [(results_dir, None)].
    - If it is a "root" directory (e.g. iris_results/) that contains norm_fwd/ and/or no_norm_fwd/,
      return those that exist (or only `requested` if provided).
    """
    if os.path.isdir(os.path.join(results_dir, "no_lateral")):
        return [(results_dir, None)]

    root, embedded = _norm_root_and_name(results_dir)
    if embedded is not None:
        # Caller passed .../norm_fwd or .../no_norm_fwd
        results_dir = os.path.join(root, embedded)
        return [(results_dir, embedded)]

    candidates = []
    if requested is not None:
        candidates = [requested]
    else:
        candidates = ["norm_fwd", "no_norm_fwd"]

    out: list[tuple[str, str | None]] = []
    for name in candidates:
        candidate = os.path.join(results_dir, name)
        if os.path.isdir(os.path.join(candidate, "no_lateral")):
            out.append((candidate, name))
    return out


def _norm_display_name(norm_name: str | None) -> str:
    if norm_name is None:
        return ""
    if norm_name == "norm_fwd":
        return "Norm fwd"
    if norm_name == "no_norm_fwd":
        return "No norm fwd"
    return str(norm_name)


def _combined_save_base(root_results_dir: str, width: int = None) -> str:
    """Folder for combined norm_fwd vs no_norm_fwd comparison plots."""
    width_label = f"width_{width}" if width is not None else "flat"
    return os.path.join(root_results_dir, "norm_compare", width_label)


def _run_subdir_name(mode: str) -> str:
    """Map each training mode to the per-run directory name it writes."""
    return "backprop" if mode == "backprop" else "lateral_pdm"


def _thetaf_dir_name(thetaf: float) -> str:
    """Directory name used for a specific thetaf value."""
    return f"thetaf_{thetaf:g}"


def _mode_base_dir(
    results_dir: str,
    mode: str,
    width: int = None,
    thetaf: float = None,
):
    """Return the base directory for a mode/width[/thetaf] combination."""
    # Backprop results may be stored either alongside the (norm_*) PDM results
    # or one level above (e.g. iris_results/backprop while PDM is under
    # iris_results/norm_fwd/*). Search both to be robust.
    if mode == "backprop":
        candidates = [os.path.join(results_dir, "backprop")]
        parent = os.path.dirname(os.path.normpath(results_dir))
        if parent and parent != results_dir:
            candidates.append(os.path.join(parent, "backprop"))
        base = next((c for c in candidates if os.path.isdir(c)), candidates[0])
    else:
        base = os.path.join(results_dir, mode)
    if width is not None:
        base = os.path.join(base, f"width_{width}")
    if mode != "backprop" and thetaf is not None:
        base = os.path.join(base, _thetaf_dir_name(thetaf))
    return base


def _summary_save_base(results_dir: str, width: int = None, thetaf: float = None):
    """Save combined comparison plots under the no_lateral width directory."""
    return _mode_base_dir(results_dir, "no_lateral", width=width, thetaf=None)


def _should_plot_series(mode: str, thetaf: float = None) -> bool:
    """Keep every trained mode/thetaf combination on comparison plots."""
    return True


def _series_key(mode: str, thetaf: float = None) -> str:
    """Stable dictionary key for one plotted series."""
    if mode == "backprop":
        return mode
    if thetaf is None:
        return f"{mode}__legacy"
    return f"{mode}__thetaf_{thetaf:g}"


def _series_key_with_norm(norm_name: str | None, mode: str, thetaf: float = None) -> str:
    norm_part = norm_name if norm_name is not None else "flat"
    return f"{norm_part}__{_series_key(mode, thetaf)}"


def _series_label(mode: str, thetaf: float = None) -> str:
    """Legend label for one plotted series."""
    if mode == "backprop" or thetaf is None:
        return MODE_LABELS[mode]
    if mode == "with_lateral" and np.isclose(thetaf, 1.0):
        base_label = "Fixed input weights"
    else:
        base_label = MODE_LABELS[mode]
    return f"{base_label} ($\\theta_f={thetaf:g}$)"


def _series_label_with_norm(mode: str, thetaf: float = None, norm_name: str | None = None) -> str:
    """
    Legend label for one plotted series, adjusting wording for no_norm_fwd.

    Requirements:
    - norm_fwd keeps existing labels ("Norm. ...")
    - no_norm_fwd uses "Standard" / "Standard + lateral" and always shows θ_f in parentheses
      when thetaf is available.
    """
    if norm_name != "no_norm_fwd":
        return _series_label(mode, thetaf)

    if mode == "backprop":
        return MODE_LABELS[mode]

    # no_norm_fwd
    base = "Standard" if mode == "no_lateral" else "Standard + lateral"
    if thetaf is None:
        return base
    return f"{base} ($\\theta_f={thetaf:g}$)"


def _linestyle_for_norm(base_linestyle: str, norm_name: str | None) -> str:
    """
    Encode norm_fwd vs no_norm_fwd as a line-style difference while preserving the
    existing thetaf=1.0 dotted convention.
    """
    if norm_name is None or norm_name == "norm_fwd":
        return base_linestyle
    # no_norm_fwd: use dashed variants
    return "-." if base_linestyle == ":" else "--"


def _series_style(
    mode: str,
    thetaf: float,
    norm_name: str | None = None,
    series_colors: dict | None = None,
):
    """Line style shared across all metric plots."""
    # Color should depend on norm_name so "Standard" can be blue while normalised is red.
    color_key = (norm_name, mode, thetaf)
    if color_key in GLOBAL_SERIES_COLOR_MAP:
        series_color = GLOBAL_SERIES_COLOR_MAP[color_key]
    else:
        reserved = RESERVED_SERIES_COLORS.get(color_key)
        if reserved is not None:
            series_color = reserved
        else:
            # Assign a distinct color, avoiding collisions with reserved colors.
            used = set(GLOBAL_SERIES_COLOR_MAP.values()) | set(RESERVED_SERIES_COLORS.values())
            palette = [c for c in SERIES_COLORS if c not in used] or SERIES_COLORS
            series_color = palette[len(GLOBAL_SERIES_COLOR_MAP) % len(palette)]
        GLOBAL_SERIES_COLOR_MAP[color_key] = series_color
    if series_colors is not None:
        series_colors[color_key] = series_color

    # Backprop is not a "normed vs standard" comparison series; keep it solid.
    base_linestyle = "-" if mode == "backprop" else (
        ":" if thetaf is not None and np.isclose(thetaf, 1.0) else "-"
    )
    linestyle = base_linestyle if mode == "backprop" else _linestyle_for_norm(base_linestyle, norm_name)

    if mode == "backprop":
        return {
            "label": MODE_LABELS[mode],
            "color": series_color,
            "linestyle": linestyle,
        }
    return {
        "label": _series_label_with_norm(mode, thetaf, norm_name),
        "color": series_color,
        "linestyle": linestyle,
    }


def _discover_all_lrs_from_results(
    results_dir: str,
    width: int = None,
    thetafs: list = None,
):
    """Union the learning rates that appear across the plotted series."""
    all_lrs = set()
    thetaf_values = thetafs if thetafs is not None else _discover_thetafs_from_results(
        results_dir, width=width
    )
    for thetaf in thetaf_values:
        all_lrs.update(
            _discover_lrs_from_results(results_dir, width=width, thetaf=thetaf)
        )
    all_lrs.update(
        _discover_lrs_from_results(results_dir, width=width, thetaf=None)
    )
    return sorted(all_lrs)


def _load_single_series_for_lr(
    results_dir: str,
    mode: str,
    lr: float,
    width: int = None,
    thetaf: float = None,
    norm_name: str | None = None,
):
    """Load metrics for one mode/thetaf series at one learning rate."""
    mode_root = _mode_base_dir(results_dir, mode, width=width, thetaf=thetaf)

    train_files = []
    # Legacy layout:
    #   .../<thetaf_*>/lr_<lr>/seed_<seed>/<mode-run-dir>/train_losses.npy
    legacy_lr_dir = os.path.join(mode_root, f"lr_{lr}")
    if os.path.isdir(legacy_lr_dir):
        pattern = os.path.join(
            legacy_lr_dir, "seed_*", _run_subdir_name(mode), "train_losses.npy"
        )
        train_files = sorted(glob.glob(pattern))
    else:
        # GridSearchCV layout:
        #   .../<thetaf_*>/seed_<seed>/best_lr_<lr>/<mode-run-dir>/train_losses.npy
        pattern = os.path.join(
            mode_root,
            "seed_*",
            "best_lr_*",
            _run_subdir_name(mode),
            "train_losses.npy",
        )
        candidate_files = sorted(glob.glob(pattern))
        if not candidate_files:
            return None

        # Filter to only seeds whose chosen best_lr matches `lr`.
        filtered = []
        for tf in candidate_files:
            # tf = .../seed_<s>/best_lr_<lr>/<mode-run-dir>/train_losses.npy
            run_dir = os.path.dirname(tf)  # .../<mode-run-dir>
            best_lr_dir = os.path.dirname(run_dir)  # .../best_lr_<lr>
            base = os.path.basename(best_lr_dir)
            if not base.startswith("best_lr_"):
                continue
            try:
                chosen_lr = float(base.replace("best_lr_", ""))
            except ValueError:
                continue
            if np.isclose(chosen_lr, lr):
                filtered.append(tf)
        train_files = sorted(filtered)

    if not train_files:
        return None

    train_list = []
    val_list = []
    val_acc_list = []
    rankme_list = []
    rankme_fwd_list = []
    for tf in train_files:
        run_dir = os.path.dirname(tf)
        train_arr = np.load(tf)
        val_path = os.path.join(run_dir, "val_losses.npy")
        if not os.path.exists(val_path):
            continue
        val_arr = np.load(val_path)

        val_acc_path = os.path.join(run_dir, "val_accuracies.npy")
        val_acc_arr = np.load(val_acc_path) if os.path.exists(val_acc_path) else None

        rankme_path = os.path.join(run_dir, "rankme.npy")
        rankme_arr = np.load(rankme_path) if os.path.exists(rankme_path) else None

        rankme_fwd_path = os.path.join(run_dir, "rankme_fwd.npy")
        rankme_fwd_arr = (
            np.load(rankme_fwd_path) if os.path.exists(rankme_fwd_path) else None
        )

        train_list.append(train_arr)
        val_list.append(val_arr)
        if val_acc_arr is not None:
            val_acc_list.append(val_acc_arr)
        if rankme_arr is not None:
            rankme_list.append(rankme_arr)
        if rankme_fwd_arr is not None:
            rankme_fwd_list.append(rankme_fwd_arr)

    if not train_list:
        return None

    train_stack = np.array(train_list)
    n_train_iters_saved = None
    for tf in train_files:
        run_dir = os.path.dirname(tf)
        npath = os.path.join(run_dir, "n_train_iters.npy")
        if os.path.isfile(npath):
            n_train_iters_saved = int(np.load(npath))
            break
    return {
        "mode": mode,
        "thetaf": thetaf,
        "norm_name": norm_name,
        "train_losses": train_stack,
        "val_losses": np.array(val_list),
        "val_accuracies": np.array(val_acc_list) if val_acc_list else None,
        "rankmes": np.array(rankme_list) if rankme_list else None,
        "rankmes_fwd": np.array(rankme_fwd_list) if rankme_fwd_list else None,
        "n_train_iters": train_stack.shape[-1],
        "n_train_iters_saved": n_train_iters_saved,
    }


def _mean_stderr(arr: np.ndarray):
    """From (n_runs, n_steps) return mean and standard error along axis=0."""
    if arr.ndim == 1:
        return arr, np.zeros_like(arr)
    mean = np.mean(arr, axis=0)
    std = np.std(arr, axis=0)
    n = arr.shape[0]
    stderr = std / np.sqrt(max(n, 1))
    return mean, stderr


def _last_finite_per_run(arr_2d: np.ndarray) -> np.ndarray:
    """
    For an array shaped (n_runs, n_steps), return the last finite value per run.
    If a run has no finite values, it contributes a NaN.
    """
    if arr_2d.ndim != 2:
        raise ValueError(f"Expected 2D array (n_runs, n_steps), got shape={arr_2d.shape}")
    out = np.full((arr_2d.shape[0],), np.nan, dtype=float)
    for i in range(arr_2d.shape[0]):
        row = arr_2d[i]
        finite = np.isfinite(row)
        if not finite.any():
            continue
        out[i] = float(row[np.flatnonzero(finite)[-1]])
    return out


def _legacy_val_log_count(n_train_iters: int, print_every: int) -> int:
    """Number of val/rankme log points without the pre-train (init) prefix."""
    steps = list(range(0, n_train_iters, print_every))
    if steps and steps[-1] != n_train_iters - 1:
        steps.append(n_train_iters - 1)
    return len(steps)


def _infer_optimizer_steps(train_len: int, val_len: int, print_every: int) -> int:
    """Recover configured n_train_iters from saved curve lengths (legacy vs init-prefixed)."""
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
    """Configured training iterations (not len(train_losses) when init point is stored)."""
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
    X-coordinates for logged val / rankme points.

    Legacy: one point per loop step s in {0, print_every, ...} (after iteration s).

    With init (train_iris after init logging): first point at x=0 (pre-train); then
    x = s + 1 for each legacy s (number of completed weight updates).
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


def _discover_widths_from_results(results_dir: str):
    """Return sorted list of widths (int) found under results_dir/no_lateral/width_*."""
    no_lateral_dir = os.path.join(results_dir, "no_lateral")
    if not os.path.isdir(no_lateral_dir):
        return []
    width_dirs = sorted(
        d for d in glob.glob(os.path.join(no_lateral_dir, "width_*"))
        if os.path.isdir(d)
    )
    widths = []
    for d in width_dirs:
        try:
            widths.append(int(os.path.basename(d).replace("width_", "")))
        except ValueError:
            continue
    return sorted(widths)


def _discover_thetafs_from_results(results_dir: str, width: int = None):
    """
    Return sorted thetaf values found under the PDM mode directories for a width.
    Falls back to [None] for backward-compatible flat layouts with lr_* directly under
    a mode width directory.
    """
    thetafs = set()
    has_legacy_layout = False

    for mode in ("no_lateral", "with_lateral"):
        base = _mode_base_dir(results_dir, mode, width=width)
        if not os.path.isdir(base):
            continue

        thetaf_dirs = sorted(
            d for d in glob.glob(os.path.join(base, "thetaf_*")) if os.path.isdir(d)
        )
        for d in thetaf_dirs:
            try:
                thetafs.add(float(os.path.basename(d).replace("thetaf_", "")))
            except ValueError:
                continue

        lr_dirs = sorted(
            d for d in glob.glob(os.path.join(base, "lr_*")) if os.path.isdir(d)
        )
        if lr_dirs:
            has_legacy_layout = True

    if thetafs:
        return sorted(thetafs)
    return [None] if has_legacy_layout else []


def _width_has_best_lr_training_runs(results_dir: str, width: int) -> bool:
    """True if this width has train_iris grid-search output (seed_*/best_lr_*/<run>/train_losses.npy)."""
    if width is None:
        return False
    wdir = f"width_{width}"
    patterns = [
        os.path.join(
            results_dir,
            "no_lateral",
            wdir,
            "thetaf_*",
            "seed_*",
            "best_lr_*",
            "lateral_pdm",
            "train_losses.npy",
        ),
        os.path.join(
            results_dir,
            "with_lateral",
            wdir,
            "thetaf_*",
            "seed_*",
            "best_lr_*",
            "lateral_pdm",
            "train_losses.npy",
        ),
        os.path.join(
            results_dir,
            "backprop",
            wdir,
            "seed_*",
            "best_lr_*",
            "backprop",
            "train_losses.npy",
        ),
    ]
    return any(glob.glob(p) for p in patterns)


def _discover_lrs_from_results(
    results_dir: str,
    width: int = None,
    thetaf: float = None,
):
    """
    Return sorted list of learning rates found across result trees.
    For PDM modes, if thetaf is set: look under width_<width>/thetaf_<thetaf>/lr_*.
    Backprop ignores thetaf and is searched under width_<width>/lr_*.
    """
    lrs = set()
    for mode in MODE_NAMES:
        base = _mode_base_dir(results_dir, mode, width=width, thetaf=thetaf)
        if not os.path.isdir(base):
            continue

        # When thetaf is None, PDM bases are width_<w>/ only. Do not scan lr_* there if
        # thetaf_* subdirs exist: those lr_* folders are usually plot outputs from this
        # script, not train_iris runs (which live under thetaf_*/...).
        if thetaf is None and mode != "backprop":
            if glob.glob(os.path.join(base, "thetaf_*")):
                continue

        # Legacy layout.
        for d in glob.glob(os.path.join(base, "lr_*")):
            if not os.path.isdir(d):
                continue
            try:
                lrs.add(float(os.path.basename(d).replace("lr_", "")))
            except ValueError:
                continue

        # GridSearchCV layout.
        #   .../seed_<seed>/best_lr_<lr>/
        for d in glob.glob(os.path.join(base, "seed_*", "best_lr_*")):
            if not os.path.isdir(d):
                continue
            try:
                lrs.add(float(os.path.basename(d).replace("best_lr_", "")))
            except ValueError:
                continue

    return sorted(lrs)


def load_losses_for_lr(
    results_dir: str,
    lr: float,
    print_every: int,
    width: int = None,
    thetafs: list = None,
    norm_name: str | None = None,
):
    """
    Load train/val losses (and optional val accuracies) for a given learning rate
    across seeds, for all plotted series.

    Directory layout (from train_iris.py):
        results_dir/
          no_lateral/with_lateral/backprop/
            [width_<width>/]   # if width is not None
            [thetaf_<thetaf>/] # PDM modes only
            lr_<lr>/
              seed_0/, seed_1/, ...
                <mode-run-dir>/train_losses.npy
                <mode-run-dir>/val_losses.npy
    """
    data = {}
    thetaf_values = thetafs if thetafs is not None else _discover_thetafs_from_results(
        results_dir, width=width
    )

    for thetaf in thetaf_values:
        for mode in ("no_lateral", "with_lateral"):
            if not _should_plot_series(mode, thetaf):
                continue
            entry = _load_single_series_for_lr(
                results_dir, mode, lr, width=width, thetaf=thetaf, norm_name=norm_name
            )
            if entry is not None:
                data[_series_key(mode, thetaf)] = entry

    backprop_entry = _load_single_series_for_lr(
        results_dir, "backprop", lr, width=width, thetaf=None, norm_name=norm_name
    )
    if backprop_entry is not None:
        data[_series_key("backprop", None)] = backprop_entry

    return data


def _load_single_series_best_selected(
    results_dir: str,
    mode: str,
    width: int = None,
    thetaf: float = None,
    norm_name: str | None = None,
):
    """
    Load train/test curves aggregated over seeds, using each seed's
    GridSearchCV-chosen best learning rate.
    """
    mode_root = _mode_base_dir(results_dir, mode, width=width, thetaf=thetaf)
    if not os.path.isdir(mode_root):
        return None

    pattern = os.path.join(
        mode_root,
        "seed_*",
        "best_lr_*",
        _run_subdir_name(mode),
        "train_losses.npy",
    )
    train_files = sorted(glob.glob(pattern))
    if not train_files:
        return None

    train_list = []
    val_list = []
    val_acc_list = []
    rankme_list = []
    rankme_fwd_list = []
    n_train_iters_saved = None

    for tf in train_files:
        run_dir = os.path.dirname(tf)  # .../<mode-run-dir>
        train_arr = np.load(tf)

        val_path = os.path.join(run_dir, "val_losses.npy")
        if not os.path.exists(val_path):
            continue
        val_arr = np.load(val_path)

        val_acc_path = os.path.join(run_dir, "val_accuracies.npy")
        val_acc_arr = np.load(val_acc_path) if os.path.exists(val_acc_path) else None

        rankme_path = os.path.join(run_dir, "rankme.npy")
        rankme_arr = np.load(rankme_path) if os.path.exists(rankme_path) else None

        rankme_fwd_path = os.path.join(run_dir, "rankme_fwd.npy")
        rankme_fwd_arr = (
            np.load(rankme_fwd_path) if os.path.exists(rankme_fwd_path) else None
        )

        train_list.append(train_arr)
        val_list.append(val_arr)
        if val_acc_arr is not None:
            val_acc_list.append(val_acc_arr)
        if rankme_arr is not None:
            rankme_list.append(rankme_arr)
        if rankme_fwd_arr is not None:
            rankme_fwd_list.append(rankme_fwd_arr)

        if n_train_iters_saved is None:
            npath = os.path.join(run_dir, "n_train_iters.npy")
            if os.path.isfile(npath):
                n_train_iters_saved = int(np.load(npath))

    if not train_list:
        return None

    train_stack = np.array(train_list)
    return {
        "mode": mode,
        "thetaf": thetaf,
        "norm_name": norm_name,
        "train_losses": train_stack,
        "val_losses": np.array(val_list),
        "val_accuracies": np.array(val_acc_list) if val_acc_list else None,
        "rankmes": np.array(rankme_list) if rankme_list else None,
        "rankmes_fwd": np.array(rankme_fwd_list) if rankme_fwd_list else None,
        "n_train_iters": train_stack.shape[-1],
        "n_train_iters_saved": n_train_iters_saved,
    }


def load_losses_best_selected(
    results_dir: str,
    width: int = None,
    thetafs: list = None,
    norm_name: str | None = None,
):
    """
    Load train/test curves aggregated over seeds, where each seed uses its
    GridSearchCV-chosen best learning rate.
    """
    data = {}
    thetaf_values = thetafs if thetafs is not None else _discover_thetafs_from_results(
        results_dir, width=width
    )

    for thetaf in thetaf_values:
        for mode in ("no_lateral", "with_lateral"):
            if not _should_plot_series(mode, thetaf):
                continue
            entry = _load_single_series_best_selected(
                results_dir, mode, width=width, thetaf=thetaf, norm_name=norm_name
            )
            if entry is not None:
                data[_series_key(mode, thetaf)] = entry

    backprop_entry = _load_single_series_best_selected(
        results_dir, "backprop", width=width, thetaf=None, norm_name=norm_name
    )
    if backprop_entry is not None:
        data[_series_key("backprop", None)] = backprop_entry

    return data


def _load_losses_over_lrs(
    results_dir: str,
    print_every: int,
    width: int = None,
    thetafs: list = None,
):
    """
    Backward-compatible alias for loading metric data over learning rates.
    """
    return _load_losses_over_lrs_iris(
        results_dir, print_every, width=width, thetafs=thetafs
    )


def _compute_best_lrs_from_lrs(
    results_dir: str,
    print_every: int,
    thetafs: list = None,
):
    """
    Backward-compatible alias for computing best learning rates across plotted series.
    """
    return _compute_best_lrs_from_lrs_iris(
        results_dir, print_every, width=None, thetafs=thetafs
    )


def plot_train_losses(
    data,
    results_dir,
    lr,
    log_scale=False,
    max_steps=None,
    width=None,
    thetafs=None,
    save_base_dir: str | None = None,
):
    """Plot train loss vs step (mean ± standard error over seeds)."""
    sns.set_theme(context="talk")
    fig, ax = plt.subplots(figsize=(METRIC_PLOT_WIDTH, METRIC_PLOT_HEIGHT))

    series_colors = {}
    y_for_scale = []
    for entry in data.values():
        train = entry["train_losses"]
        center, stderr = _mean_stderr(train)
        low, high = center - stderr, center + stderr
        y_for_scale.extend([center, low, high])

        if max_steps is not None:
            n = min(len(center), max_steps)
            center = center[:n]
            low = low[:n]
            high = high[:n]

        steps = np.arange(len(center))
        style = _series_style(entry["mode"], entry["thetaf"], entry.get("norm_name"), series_colors)

        ax.plot(
            steps,
            center,
            label=style["label"],
            color=style["color"],
            linestyle=style["linestyle"],
            alpha=0.9,
            linewidth=2,
        )
        if train.ndim > 1 and train.shape[0] > 1:
            ax.fill_between(steps, low, high, color=style["color"], alpha=0.18)

    ax.set_xlabel("Training step", fontsize=AXIS_LABEL_FONTSIZE, labelpad=10)
    ax.set_ylabel("Training loss", fontsize=AXIS_LABEL_FONTSIZE, labelpad=10)
    ax.legend(
        loc="upper left",
        bbox_to_anchor=(1.02, 1.0),
        borderaxespad=0.0,
        fontsize=LEGEND_FONTSIZE,
    )
    ax.tick_params(axis="both", labelsize=TICK_LABEL_FONTSIZE)
    if log_scale and y_for_scale:
        y = np.concatenate([np.ravel(v) for v in y_for_scale])
        y = y[np.isfinite(y)]
        can_log = (y.size > 0) and (np.nanmin(y) > 0.0)
        ax.set_yscale("log" if can_log else "linear")
    else:
        ax.set_yscale("linear")
    ax.grid(True, alpha=0.3)
    ax.set_xlim(left=0)
    plt.tight_layout(rect=(0, 0, 0.8, 1))

    base = save_base_dir if save_base_dir is not None else _summary_save_base(results_dir, width=width)
    save_dir = os.path.join(base, f"lr_{lr}")
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, "train_losses.pdf")
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved {save_path}")


def plot_val_losses(
    data,
    results_dir,
    lr,
    print_every,
    log_scale=False,
    max_steps=None,
    width=None,
    thetafs=None,
    save_base_dir: str | None = None,
):
    """Plot validation (test) loss vs step (mean ± standard error over seeds)."""
    sns.set_theme(context="talk")
    fig, ax = plt.subplots(figsize=(METRIC_PLOT_WIDTH, METRIC_PLOT_HEIGHT))

    series_colors = {}
    y_for_scale = []
    for entry in data.values():
        val = entry["val_losses"]
        n_train_iters = _true_n_train_iters(entry, print_every)

        center, stderr = _mean_stderr(val)
        low, high = center - stderr, center + stderr
        y_for_scale.extend([center, low, high])

        test_steps = test_steps_from_length(n_train_iters, print_every, len(center))
        if max_steps is not None:
            mask = test_steps <= max_steps
            if mask.any():
                test_steps = test_steps[mask]
                center = center[mask]
                low = low[mask]
                high = high[mask]

        style = _series_style(entry["mode"], entry["thetaf"], entry.get("norm_name"), series_colors)

        ax.plot(
            test_steps,
            center,
            label=style["label"],
            color=style["color"],
            linestyle=style["linestyle"],
            alpha=0.9,
            linewidth=2,
        )
        if val.ndim > 1 and val.shape[0] > 1:
            ax.fill_between(test_steps, low, high, color=style["color"], alpha=0.18)

    ax.set_xlabel("Training step", fontsize=AXIS_LABEL_FONTSIZE, labelpad=10)
    ax.set_ylabel("Test loss", fontsize=AXIS_LABEL_FONTSIZE, labelpad=10)
    ax.legend(
        loc="upper left",
        bbox_to_anchor=(1.02, 1.0),
        borderaxespad=0.0,
        fontsize=LEGEND_FONTSIZE,
    )
    ax.tick_params(axis="both", labelsize=TICK_LABEL_FONTSIZE)
    if log_scale and y_for_scale:
        y = np.concatenate([np.ravel(v) for v in y_for_scale])
        y = y[np.isfinite(y)]
        can_log = (y.size > 0) and (np.nanmin(y) > 0.0)
        ax.set_yscale("log" if can_log else "linear")
    else:
        ax.set_yscale("linear")
    ax.grid(True, alpha=0.3)
    ax.set_xlim(left=0)
    plt.tight_layout(rect=(0, 0, 0.8, 1))

    base = save_base_dir if save_base_dir is not None else _summary_save_base(results_dir, width=width)
    save_dir = os.path.join(base, f"lr_{lr}")
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, "test_losses.pdf")
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved {save_path}")


def plot_val_accuracies(
    data,
    results_dir,
    lr,
    print_every,
    max_steps=None,
    width=None,
    thetafs=None,
    save_base_dir: str | None = None,
):
    """Plot validation accuracy vs step (mean ± standard error over seeds)."""
    sns.set_theme(context="talk")
    fig, ax = plt.subplots(figsize=(METRIC_PLOT_WIDTH, METRIC_PLOT_HEIGHT))

    series_colors = {}
    for entry in data.values():
        val_accs = entry.get("val_accuracies", None)
        if val_accs is None:
            continue

        n_train_iters = _true_n_train_iters(entry, print_every)

        center, stderr = _mean_stderr(val_accs)
        low, high = center - stderr, center + stderr

        steps = test_steps_from_length(n_train_iters, print_every, len(center))
        if max_steps is not None:
            mask = steps <= max_steps
            if mask.any():
                steps = steps[mask]
                center = center[mask]
                low = low[mask]
                high = high[mask]

        style = _series_style(entry["mode"], entry["thetaf"], entry.get("norm_name"), series_colors)

        ax.plot(
            steps,
            center,
            label=style["label"],
            color=style["color"],
            linestyle=style["linestyle"],
            alpha=0.9,
            linewidth=2,
        )
        if val_accs.ndim > 1 and val_accs.shape[0] > 1:
            ax.fill_between(steps, low, high, color=style["color"], alpha=0.18)

    ax.set_xlabel("Training step", fontsize=AXIS_LABEL_FONTSIZE, labelpad=10)
    ax.set_ylabel("Test acc.", fontsize=AXIS_LABEL_FONTSIZE, labelpad=10)
    ax.legend(
        loc="upper left",
        bbox_to_anchor=(1.02, 1.0),
        borderaxespad=0.0,
        fontsize=LEGEND_FONTSIZE,
    )
    ax.tick_params(axis="both", labelsize=TICK_LABEL_FONTSIZE)
    ax.set_ylim(ACC_PLOT_YMIN, 1.0)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(left=0)
    plt.tight_layout(rect=(0, 0, 0.8, 1))

    base = save_base_dir if save_base_dir is not None else _summary_save_base(results_dir, width=width)
    save_dir = os.path.join(base, f"lr_{lr}")
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, "test_accuracies.pdf")
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved {save_path}")


def plot_rankme(
    data,
    results_dir: str,
    lr: float,
    print_every: int,
    max_steps: int = None,
    width: int = None,
    thetafs: list = None,
    save_base_dir: str | None = None,
):
    """Plot RankMe during training (mean ± std over seeds) for both models on the same axes."""
    sns.set_theme(context="talk")
    fig, ax = plt.subplots(figsize=(METRIC_PLOT_WIDTH, METRIC_PLOT_HEIGHT))

    series_colors = {}
    for entry in data.values():
        rankmes = entry.get("rankmes")
        if rankmes is None or rankmes.size == 0:
            continue
        n_train_iters = _true_n_train_iters(entry, print_every)
        # Shade mean ± standard error across seeds for consistency with other metrics.
        center, stderr = _mean_stderr(rankmes)
        low, high = center - stderr, center + stderr

        steps = test_steps_from_length(n_train_iters, print_every, len(center))
        if max_steps is not None:
            mask = steps <= max_steps
            if mask.any():
                steps = steps[mask]
                center = center[mask]
                low = low[mask]
                high = high[mask]

        style = _series_style(entry["mode"], entry["thetaf"], entry.get("norm_name"), series_colors)
        ax.plot(
            steps,
            center,
            label=style["label"],
            color=style["color"],
            linestyle=style["linestyle"],
            alpha=0.9,
            linewidth=2,
        )
        if rankmes.ndim > 1 and rankmes.shape[0] > 1:
            ax.fill_between(steps, low, high, color=style["color"], alpha=0.18)

    ax.set_xlabel("Training step", fontsize=AXIS_LABEL_FONTSIZE, labelpad=10)
    ax.set_ylabel("RankMe", fontsize=AXIS_LABEL_FONTSIZE, labelpad=10)
    ax.legend(
        loc="upper left",
        bbox_to_anchor=(1.02, 1.0),
        borderaxespad=0.0,
        fontsize=LEGEND_FONTSIZE,
    )
    ax.tick_params(axis="both", labelsize=TICK_LABEL_FONTSIZE)
    ax.set_xlim(left=0)
    ax.grid(True, alpha=0.3)
    plt.tight_layout(rect=(0, 0, 0.8, 1))

    base = save_base_dir if save_base_dir is not None else _summary_save_base(results_dir, width=width)
    save_dir = os.path.join(base, f"lr_{lr}")
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, "rankme.pdf")
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved {save_path}")


def _load_losses_over_lrs_iris(
    results_dir: str,
    print_every: int,
    width: int = None,
    thetafs: list = None,
):
    """
    Load train/val/val_acc for every plotted series and learning rate for a given width.
    If width is None, discover lr_* under results_dir/no_lateral/ (flat or under width_*).
    Returns lr_data[series_key][lr] = { ... }.
    """
    lrs = _discover_all_lrs_from_results(results_dir, width=width, thetafs=thetafs)
    lr_data = {}
    for lr_val in lrs:
        data = load_losses_for_lr(
            results_dir, lr_val, print_every, width=width, thetafs=thetafs
        )
        for series_key, entry in data.items():
            lr_data.setdefault(series_key, {})
            lr_data[series_key][lr_val] = entry
    return lr_data


def _compute_best_lrs_from_lrs_iris(
    results_dir: str,
    print_every: int,
    width: int = None,
    thetafs: list = None,
):
    """
    For each plotted series: best lr = lr with lowest mean(best val loss over time).
    """
    lr_data = _load_losses_over_lrs_iris(
        results_dir, print_every, width=width, thetafs=thetafs
    )
    best_val_means = {}
    best_val_acc_means = {}
    best_rankme_means = {}
    best_lr = {}
    best_rankme_lr = {}

    for series_key, series_lr_data in lr_data.items():
        best_val_means[series_key] = {}
        best_val_acc_means[series_key] = {}
        best_rankme_means[series_key] = {}
        for lr_val, entry in series_lr_data.items():
            val_losses = entry["val_losses"]
            if val_losses.size == 0:
                continue
            if val_losses.ndim == 1:
                val_losses = val_losses[np.newaxis, :]
            best_per_run = val_losses.min(axis=1)
            best_val_means[series_key][lr_val] = float(best_per_run.mean())

            val_acc = entry.get("val_accuracies")
            if val_acc is not None and val_acc.size > 0:
                if val_acc.ndim == 1:
                    val_acc = val_acc[np.newaxis, :]
                best_acc_per_run = val_acc.max(axis=1)
                best_val_acc_means[series_key][lr_val] = float(best_acc_per_run.mean())

            rankmes = entry.get("rankmes")
            if rankmes is not None and rankmes.size > 0:
                if rankmes.ndim == 1:
                    rankmes = rankmes[np.newaxis, :]
                end_rankme_per_run = _last_finite_per_run(rankmes)
                end_rankme_per_run = end_rankme_per_run[np.isfinite(end_rankme_per_run)]
                if end_rankme_per_run.size > 0:
                    best_rankme_means[series_key][lr_val] = float(
                        np.nanmean(end_rankme_per_run)
                    )

        if best_val_means[series_key]:
            best_lr[series_key] = min(
                best_val_means[series_key],
                key=lambda lr_: best_val_means[series_key][lr_],
            )
        if best_rankme_means[series_key]:
            best_rankme_lr[series_key] = max(
                best_rankme_means[series_key],
                key=lambda lr_: best_rankme_means[series_key][lr_],
            )

    return (
        lr_data,
        best_lr,
        best_rankme_lr,
        best_val_means,
        best_val_acc_means,
        best_rankme_means,
    )


def plot_train_loss_best_lr_over_lrs(
    results_dir: str,
    print_every: int,
    log_scale: bool = False,
    max_steps=None,
    width: int = None,
    thetafs: list = None,
    save_base_dir: str | None = None,
):
    """Plot train loss over training at the best lr for each plotted series."""
    sns.set_theme(context="talk")
    lr_data, best_lr, _, _, _, _ = _compute_best_lrs_from_lrs_iris(
        results_dir, print_every, width=width, thetafs=thetafs
    )
    if not best_lr:
        print("No loss data across learning rates. Run train_iris.py with multiple lrs first.")
        return

    fig, ax = plt.subplots(figsize=(METRIC_PLOT_WIDTH, METRIC_PLOT_HEIGHT))
    series_colors = {}
    y_for_scale = []
    for series_key, lr_val in best_lr.items():
        entry = lr_data[series_key][lr_val]
        train = entry["train_losses"]
        if train.ndim == 1:
            train = train[np.newaxis, :]
        center, stderr = _mean_stderr(train)
        low, high = center - stderr, center + stderr
        y_for_scale.extend([center, low, high])
        if max_steps is not None:
            n = min(len(center), max_steps)
            center, low, high = center[:n], low[:n], high[:n]
        steps = np.arange(len(center))
        style = _series_style(entry["mode"], entry["thetaf"], entry.get("norm_name"), series_colors)
        label = style["label"]
        ax.plot(
            steps,
            center,
            label=label,
            color=style["color"],
            linestyle=style["linestyle"],
            linewidth=2,
        )
        if train.shape[0] > 1:
            ax.fill_between(steps, low, high, color=style["color"], alpha=0.18)

    ax.set_xlabel("Training step", fontsize=AXIS_LABEL_FONTSIZE, labelpad=10)
    ax.set_ylabel("Training loss", fontsize=AXIS_LABEL_FONTSIZE, labelpad=10)
    ax.legend(
        loc="upper left",
        bbox_to_anchor=(1.02, 1.0),
        borderaxespad=0.0,
        fontsize=LEGEND_FONTSIZE,
    )
    ax.tick_params(axis="both", labelsize=TICK_LABEL_FONTSIZE)
    if log_scale and y_for_scale:
        y = np.concatenate([np.ravel(v) for v in y_for_scale])
        y = y[np.isfinite(y)]
        can_log = (y.size > 0) and (np.nanmin(y) > 0.0)
        ax.set_yscale("log" if can_log else "linear")
    else:
        ax.set_yscale("linear")
    ax.grid(True, alpha=0.3)
    ax.set_xlim(left=0)
    plt.tight_layout(rect=(0, 0, 0.8, 1))
    save_base = save_base_dir if save_base_dir is not None else _summary_save_base(results_dir, width=width)
    os.makedirs(save_base, exist_ok=True)
    save_path = os.path.join(save_base, "best_train_losses.pdf")
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved {save_path}")


def plot_val_loss_best_lr_over_lrs(
    results_dir: str,
    print_every: int,
    log_scale: bool = False,
    max_steps=None,
    width: int = None,
    thetafs: list = None,
    save_base_dir: str | None = None,
):
    """Plot val loss over training at the best lr for each plotted series."""
    sns.set_theme(context="talk")
    lr_data, best_lr, _, _, _, _ = _compute_best_lrs_from_lrs_iris(
        results_dir, print_every, width=width, thetafs=thetafs
    )
    if not best_lr:
        print("No validation data across learning rates. Run train_iris.py with multiple lrs first.")
        return

    fig, ax = plt.subplots(figsize=(METRIC_PLOT_WIDTH, METRIC_PLOT_HEIGHT))
    series_colors = {}
    y_for_scale = []
    for series_key, lr_val in best_lr.items():
        entry = lr_data[series_key][lr_val]
        val = entry["val_losses"]
        n_train_iters = _true_n_train_iters(entry, print_every)
        if val.ndim == 1:
            val = val[np.newaxis, :]
        center, stderr = _mean_stderr(val)
        low, high = center - stderr, center + stderr
        y_for_scale.extend([center, low, high])
        steps = test_steps_from_length(n_train_iters, print_every, len(center))
        if max_steps is not None:
            mask = steps <= max_steps
            if mask.any():
                steps, center, low, high = steps[mask], center[mask], low[mask], high[mask]
        style = _series_style(entry["mode"], entry["thetaf"], entry.get("norm_name"), series_colors)
        label = style["label"]
        ax.plot(
            steps,
            center,
            label=label,
            color=style["color"],
            linestyle=style["linestyle"],
            linewidth=2,
        )
        if val.shape[0] > 1:
            ax.fill_between(steps, low, high, color=style["color"], alpha=0.18)

    ax.set_xlabel("Training step", fontsize=AXIS_LABEL_FONTSIZE, labelpad=10)
    ax.set_ylabel("Test loss", fontsize=AXIS_LABEL_FONTSIZE, labelpad=10)
    ax.legend(
        loc="upper left",
        bbox_to_anchor=(1.02, 1.0),
        borderaxespad=0.0,
        fontsize=LEGEND_FONTSIZE,
    )
    ax.tick_params(axis="both", labelsize=TICK_LABEL_FONTSIZE)
    if log_scale and y_for_scale:
        y = np.concatenate([np.ravel(v) for v in y_for_scale])
        y = y[np.isfinite(y)]
        can_log = (y.size > 0) and (np.nanmin(y) > 0.0)
        ax.set_yscale("log" if can_log else "linear")
    else:
        ax.set_yscale("linear")
    ax.grid(True, alpha=0.3)
    ax.set_xlim(left=0)
    plt.tight_layout(rect=(0, 0, 0.8, 1))
    save_base = save_base_dir if save_base_dir is not None else _summary_save_base(results_dir, width=width)
    os.makedirs(save_base, exist_ok=True)
    save_path = os.path.join(save_base, "best_test_losses.pdf")
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved {save_path}")


def plot_val_acc_best_lr_over_lrs(
    results_dir: str,
    print_every: int,
    max_steps=None,
    width: int = None,
    thetafs: list = None,
    save_base_dir: str | None = None,
):
    """Plot val accuracy over training at the best lr for each plotted series."""
    sns.set_theme(context="talk")
    lr_data, best_lr, _, _, _, _ = _compute_best_lrs_from_lrs_iris(
        results_dir, print_every, width=width, thetafs=thetafs
    )
    if not best_lr:
        print("No validation data across learning rates. Run train_iris.py with multiple lrs first.")
        return

    fig, ax = plt.subplots(figsize=(METRIC_PLOT_WIDTH, METRIC_PLOT_HEIGHT))
    series_colors = {}
    for series_key, lr_val in best_lr.items():
        entry = lr_data[series_key][lr_val]
        val_acc = entry.get("val_accuracies")
        if val_acc is None or val_acc.size == 0:
            continue
        if val_acc.ndim == 1:
            val_acc = val_acc[np.newaxis, :]
        n_train_iters = _true_n_train_iters(entry, print_every)
        center, stderr = _mean_stderr(val_acc)
        low, high = center - stderr, center + stderr
        steps = test_steps_from_length(n_train_iters, print_every, len(center))
        if max_steps is not None:
            mask = steps <= max_steps
            if mask.any():
                steps, center, low, high = steps[mask], center[mask], low[mask], high[mask]
        style = _series_style(entry["mode"], entry["thetaf"], entry.get("norm_name"), series_colors)
        label = style["label"]
        ax.plot(
            steps,
            center,
            label=label,
            color=style["color"],
            linestyle=style["linestyle"],
            linewidth=2,
        )
        if val_acc.shape[0] > 1:
            ax.fill_between(steps, low, high, color=style["color"], alpha=0.18)

    ax.set_xlabel("Training step", fontsize=AXIS_LABEL_FONTSIZE, labelpad=10)
    ax.set_ylabel("Test acc.", fontsize=AXIS_LABEL_FONTSIZE, labelpad=10)
    ax.legend(
        loc="upper left",
        bbox_to_anchor=(1.02, 1.0),
        borderaxespad=0.0,
        fontsize=LEGEND_FONTSIZE,
    )
    ax.tick_params(axis="both", labelsize=TICK_LABEL_FONTSIZE)
    ax.set_ylim(ACC_PLOT_YMIN, 1.0)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(left=0)
    plt.tight_layout(rect=(0, 0, 0.8, 1))
    save_base = save_base_dir if save_base_dir is not None else _summary_save_base(results_dir, width=width)
    os.makedirs(save_base, exist_ok=True)
    save_path = os.path.join(save_base, "best_test_accuracies.pdf")
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved {save_path}")


def _bar_series_sort_key(entry: dict):
    """
    Bar order:
    - Standard (no_norm_fwd, no_lateral, thetaf=0.5) always first
    - then all Standard (no_norm_fwd) series
    - then all Normalised (norm_fwd) series
    - backprop last

    Within each norm bucket: ascending θ_f; for each θ_f, no_lateral then with_lateral.
    """
    mode = entry["mode"]
    thetaf = entry["thetaf"]
    norm_name = entry.get("norm_name")
    if mode == "backprop":
        return (99, float("inf"), 0)

    is_standard_05 = (
        norm_name == "no_norm_fwd"
        and mode == "no_lateral"
        and thetaf is not None
        and np.isclose(thetaf, 0.5)
    )
    if is_standard_05:
        return (-1, 0.0, 0)

    # Bucket by norm: standard first, then norm_fwd, then legacy/other.
    if norm_name == "no_norm_fwd":
        norm_order = 0
    elif norm_name == "norm_fwd" or norm_name is None:
        norm_order = 1
    else:
        norm_order = 2

    tf = float(thetaf) if thetaf is not None else 0.0
    # Same θ_f: no_lateral before with_lateral
    mode_order = {"no_lateral": 0, "with_lateral": 1}.get(mode, 99)
    return (norm_order, tf, mode_order)


def _plot_best_test_accuracies_bar_entries(
    entries: list,
    results_dir: str,
    width: int = None,
    save_filename: str = "best_test_accuracies_bar.pdf",
    label_override=None,
    show_xticklabels: bool = True,
    save_base_dir: str | None = None,
    acc_plot_ymin: float = ACC_PLOT_YMIN,
    sort_key=None,
    fig_width: float | None = None,
):
    """
    Bar plot of best test accuracy per series from pre-loaded metric entries.
    Each entry is like those returned by load_losses_for_lr / load_losses_best_selected.
    """
    if not entries:
        return
    if sort_key is None:
        sort_key = _bar_series_sort_key
    entries = sorted(entries, key=sort_key)
    sns.set_theme(context="talk")

    series_colors = {}
    labels = []
    means = []
    stderrs = []
    colors = []
    points_per_series = []

    for entry in entries:
        val_acc = entry.get("val_accuracies")
        if val_acc is None or val_acc.size == 0:
            continue
        if val_acc.ndim == 1:
            val_acc = val_acc[np.newaxis, :]

        best_per_run = np.nanmax(val_acc, axis=1)
        best_per_run = best_per_run[np.isfinite(best_per_run)]
        if best_per_run.size == 0:
            continue

        mean = float(np.nanmean(best_per_run))
        std = float(np.nanstd(best_per_run, axis=0))
        stderr = std / np.sqrt(max(best_per_run.shape[0], 1))

        style = _series_style(entry["mode"], entry["thetaf"], entry.get("norm_name"), series_colors)
        label = (
            label_override(entry) if callable(label_override) else style["label"]
        )

        labels.append(label)
        means.append(mean)
        stderrs.append(stderr)
        colors.append(style["color"])
        points_per_series.append(best_per_run.astype(float))

    if not labels:
        print("No test accuracy data found for any series in bar plot.")
        return

    if fig_width is None:
        fig_width = max(BAR_PLOT_MIN_WIDTH, BAR_PLOT_WIDTH_PER_LABEL * len(labels))
    fig, ax = plt.subplots(figsize=(fig_width, BAR_PLOT_HEIGHT))
    x = np.arange(len(labels))
    ax.bar(
        x,
        means,
        yerr=stderrs,
        capsize=6,
        error_kw={
            "ecolor": "black",
            "elinewidth": BAR_ERRORBAR_LINEWIDTH,
            "capthick": BAR_ERRORBAR_CAPTHICK,
            "zorder": 5,
        },
        color=colors,
        alpha=0.55,
    )

    rng = np.random.default_rng(0)
    for i, pts in enumerate(points_per_series):
        if pts is None or len(pts) == 0:
            continue
        jitter = rng.uniform(-0.18, 0.18, size=len(pts))
        ax.scatter(
            np.full(len(pts), x[i]) + jitter,
            pts,
            s=90,
            color=colors[i],
            edgecolors=JITTER_DOT_EDGE_COLOR,
            alpha=1.0,
            linewidths=JITTER_DOT_EDGE_WIDTH,
            clip_on=False,
            zorder=3,
        )
    # Add small headroom so dots at y=1 aren't cut off.
    ax.set_ylim(acc_plot_ymin, 1.0 + BAR_PLOT_Y_PADDING_FRAC)
    # Match best-loss typography in plot_toy_{nonlinear,linear}.py
    ax.set_ylabel("Test accuracy", fontsize=30, labelpad=10)
    ax.set_xticks(x)
    if show_xticklabels:
        ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=25)
    else:
        ax.set_xticklabels([])
    ax.tick_params(axis="both", labelsize=25)
    ax.grid(True, axis="y", alpha=0.3)
    fig.subplots_adjust(bottom=0.42, left=0.12, right=0.98, top=0.96)

    save_base = save_base_dir if save_base_dir is not None else _summary_save_base(results_dir, width=width)
    os.makedirs(save_base, exist_ok=True)
    save_path = os.path.join(save_base, save_filename)
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved {save_path}")


def _plot_best_rankme_bar_entries(
    entries: list,
    results_dir: str,
    width: int = None,
    save_filename: str = "best_rankme_bar.pdf",
    rankme_key: str = "rankmes",
    ylabel: str = "RankMe",
    label_override=None,
    show_xticklabels: bool = True,
    save_base_dir: str | None = None,
    sort_key=None,
    fig_width: float | None = None,
):
    """Bar plot of final RankMe-like metric per series from pre-loaded entries."""
    if not entries:
        return
    if sort_key is None:
        sort_key = _bar_series_sort_key
    entries = sorted(entries, key=sort_key)
    sns.set_theme(context="talk")

    series_colors = {}
    labels = []
    means = []
    stderrs = []
    colors = []
    points_per_series = []

    for entry in entries:
        rankmes = entry.get(rankme_key)
        if rankmes is None or rankmes.size == 0:
            continue
        if rankmes.ndim == 1:
            rankmes = rankmes[np.newaxis, :]

        end_per_run = _last_finite_per_run(rankmes)
        end_per_run = end_per_run[np.isfinite(end_per_run)]
        if end_per_run.size == 0:
            continue

        mean = float(np.nanmean(end_per_run))
        std = float(np.nanstd(end_per_run, axis=0))
        stderr = std / np.sqrt(max(end_per_run.shape[0], 1))

        style = _series_style(entry["mode"], entry["thetaf"], entry.get("norm_name"), series_colors)
        label = (
            label_override(entry) if callable(label_override) else style["label"]
        )

        labels.append(label)
        means.append(mean)
        stderrs.append(stderr)
        colors.append(style["color"])
        points_per_series.append(end_per_run.astype(float))

    if not labels:
        print("No RankMe data found for any series in bar plot.")
        return

    if fig_width is None:
        fig_width = max(BAR_PLOT_MIN_WIDTH, BAR_PLOT_WIDTH_PER_LABEL * len(labels))
    fig, ax = plt.subplots(figsize=(fig_width, BAR_PLOT_HEIGHT))
    x = np.arange(len(labels))
    ax.bar(
        x,
        means,
        yerr=stderrs,
        capsize=6,
        error_kw={
            "ecolor": "black",
            "elinewidth": BAR_ERRORBAR_LINEWIDTH,
            "capthick": BAR_ERRORBAR_CAPTHICK,
            "zorder": 5,
        },
        color=colors,
        alpha=0.55,
    )

    rng = np.random.default_rng(0)
    for i, pts in enumerate(points_per_series):
        if pts is None or len(pts) == 0:
            continue
        jitter = rng.uniform(-0.18, 0.18, size=len(pts))
        ax.scatter(
            np.full(len(pts), x[i]) + jitter,
            pts,
            s=90,
            color=colors[i],
            edgecolors=JITTER_DOT_EDGE_COLOR,
            alpha=1.0,
            linewidths=JITTER_DOT_EDGE_WIDTH,
            clip_on=False,
            zorder=3,
        )
    # Add some breathing room so dots at extremes aren't clipped.
    ax.margins(y=BAR_PLOT_Y_PADDING_FRAC)
    # Match best-loss typography in plot_toy_{nonlinear,linear}.py
    ax.set_ylabel(ylabel, fontsize=30, labelpad=10)
    ax.set_xticks(x)
    if show_xticklabels:
        ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=25)
    else:
        ax.set_xticklabels([])
    ax.tick_params(axis="both", labelsize=25)
    ax.grid(True, axis="y", alpha=0.3)
    fig.subplots_adjust(bottom=0.42, left=0.12, right=0.98, top=0.96)

    save_base = save_base_dir if save_base_dir is not None else _summary_save_base(results_dir, width=width)
    os.makedirs(save_base, exist_ok=True)
    save_path = os.path.join(save_base, save_filename)
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved {save_path}")


def plot_best_test_accuracies_bar_from_data(
    data: dict,
    results_dir: str,
    width: int = None,
    save_base_dir: str | None = None,
):
    """Bar plot from aggregated series dict (e.g. gridsearch-best line-plot data)."""
    _plot_best_test_accuracies_bar_entries(
        list(data.values()),
        results_dir,
        width=width,
        save_filename="best_test_accuracies_bar.pdf",
        save_base_dir=save_base_dir,
    )


def plot_best_rankme_bar_from_data(
    data: dict,
    results_dir: str,
    width: int = None,
    save_base_dir: str | None = None,
):
    _plot_best_rankme_bar_entries(
        list(data.values()),
        results_dir,
        width=width,
        save_filename="best_rankme_bar.pdf",
        save_base_dir=save_base_dir,
    )


def _is_requested_model_subset(entry: dict) -> bool:
    """
    Filter used for the extra "selected-models-only" bar plots.

    Requested subset:
    - theta_f = 0.5 (w/ and w/o lateral)
    - theta_f = 0.95 w/o lateral
    - nonlinear mapping (implemented as with_lateral, theta_f = 1.0)
    - backprop
    """
    mode = entry.get("mode")
    thetaf = entry.get("thetaf")
    norm_name = entry.get("norm_name")
    if mode == "backprop":
        return True
    if thetaf is None:
        return False

    # In combined-norm plots, keep only ONE "Standard" series: no_norm_fwd, no_lateral, thetaf=0.5.
    # Drop all other standard (no_norm_fwd) series from the selected-model plots.
    if norm_name == "no_norm_fwd":
        return mode == "no_lateral" and (np.isclose(thetaf, 0.5) or np.isclose(thetaf, 0.99))

    # norm_fwd (or legacy single-tree) selection as before.
    if mode == "no_lateral" and np.isclose(thetaf, 0.5):
        return True
    if mode == "with_lateral" and (np.isclose(thetaf, 0.5) or np.isclose(thetaf, 1.0)):
        return True
    return False


def _filter_entries_selected_models(entries: list) -> list:
    """Return only the entries in the requested subset, preserving order."""
    return [e for e in entries if _is_requested_model_subset(e)]


def _selected_models_sort_key(entry: dict):
    """
    Ordering for selected-model bar plots:
    - Standard (no_norm_fwd, thetaf=0.5, no_lateral) first
    - then the remaining selected series in a fixed, paper-friendly order
    - Backprop last
    """
    mode = entry.get("mode")
    thetaf = entry.get("thetaf")
    norm_name = entry.get("norm_name")
    if mode == "backprop":
        return (2, float("inf"), 0)

    # Explicit ordering for selected-model plots.
    # (norm_name, mode, thetaf) -> rank
    key = (norm_name, mode, None if thetaf is None else float(thetaf))
    ranks = {
        ("no_norm_fwd", "no_lateral", 0.5): 0,   # Standard 0.5 first
        ("norm_fwd", "no_lateral", 0.5): 1,      # Norm no-lateral 0.5
        ("norm_fwd", "with_lateral", 0.5): 2,    # Norm + lateral 0.5
        # User-requested ordering: 1-2-3-5-4-6, i.e. Standard 0.99 before nonlinear mapping.
        ("no_norm_fwd", "no_lateral", 0.99): 3,  # Standard 0.99
        ("norm_fwd", "with_lateral", 1.0): 4,    # Nonlinear mapping
    }
    if key in ranks:
        return (1, ranks[key], 0)

    # Fallback: keep stable but after the known selected items.
    tf = float(thetaf) if thetaf is not None else 0.0
    mode_order = {"no_lateral": 0, "with_lateral": 1}.get(mode, 99)
    return (1, 100 + tf, mode_order)


def plot_selected_best_test_accuracies_bar_from_data(
    data: dict,
    results_dir: str,
    width: int = None,
    save_base_dir: str | None = None,
):
    """Extra bar plot (selected models only) from aggregated series dict."""
    entries = _filter_entries_selected_models(list(data.values()))
    def _selected_label_override(entry: dict) -> str | None:
        if entry.get("norm_name") == "no_norm_fwd" and entry.get("mode") == "no_lateral" and np.isclose(entry.get("thetaf"), 0.5):
            return r"Standard ($\theta_f = 0.5$)"
        if entry.get("norm_name") == "no_norm_fwd" and entry.get("mode") == "no_lateral" and np.isclose(entry.get("thetaf"), 0.99):
            return r"Standard ($\theta_f = 0.99$)"
        return None
    _plot_best_test_accuracies_bar_entries(
        entries,
        results_dir,
        width=width,
        save_filename="best_test_accuracies_bar_selected_models.pdf",
        label_override=lambda e: _selected_label_override(e)
        or _series_style(e["mode"], e["thetaf"], e.get("norm_name"))["label"],
        show_xticklabels=False,
        save_base_dir=save_base_dir,
        acc_plot_ymin=SELECTED_ACC_PLOT_YMIN,
        sort_key=_selected_models_sort_key,
        fig_width=max(
            SELECTED_BAR_PLOT_MIN_WIDTH,
            SELECTED_BAR_PLOT_WIDTH_PER_LABEL * max(len(entries), 1),
        ),
    )
    _plot_selected_models_legend(
        entries,
        results_dir,
        width=width,
        save_filename="selected_models_legend.pdf",
        label_override=lambda e: _selected_label_override(e)
        or _series_style(e["mode"], e["thetaf"], e.get("norm_name"))["label"],
        save_base_dir=save_base_dir,
    )
    _plot_connections_legend(
        results_dir,
        width=width,
        save_filename="connections_legend.pdf",
        save_base_dir=save_base_dir,
    )


def plot_selected_best_rankme_bar_from_data(
    data: dict,
    results_dir: str,
    width: int = None,
    save_base_dir: str | None = None,
):
    """Extra bar plot (selected models only) from aggregated series dict."""
    entries = _filter_entries_selected_models(list(data.values()))
    def _selected_label_override(entry: dict) -> str | None:
        if entry.get("norm_name") == "no_norm_fwd" and entry.get("mode") == "no_lateral" and np.isclose(entry.get("thetaf"), 0.5):
            return r"Standard ($\theta_f = 0.5$)"
        if entry.get("norm_name") == "no_norm_fwd" and entry.get("mode") == "no_lateral" and np.isclose(entry.get("thetaf"), 0.99):
            return r"Standard ($\theta_f = 0.99$)"
        return None
    _plot_best_rankme_bar_entries(
        entries,
        results_dir,
        width=width,
        save_filename="best_rankme_bar_selected_models.pdf",
        label_override=lambda e: _selected_label_override(e)
        or _series_style(e["mode"], e["thetaf"], e.get("norm_name"))["label"],
        show_xticklabels=False,
        save_base_dir=save_base_dir,
        sort_key=_selected_models_sort_key,
        fig_width=max(
            SELECTED_BAR_PLOT_MIN_WIDTH,
            SELECTED_BAR_PLOT_WIDTH_PER_LABEL * max(len(entries), 1),
        ),
    )
    _plot_selected_models_legend(
        entries,
        results_dir,
        width=width,
        save_filename="selected_models_legend.pdf",
        label_override=lambda e: _selected_label_override(e)
        or _series_style(e["mode"], e["thetaf"], e.get("norm_name"))["label"],
        save_base_dir=save_base_dir,
    )


def plot_selected_best_rankme_init_bar_from_data(
    data: dict,
    results_dir: str,
    width: int = None,
    save_base_dir: str | None = None,
):
    """
    Same as `plot_selected_best_rankme_bar_from_data`, but uses RankMe computed
    from the *initialisation* of PDM activities (forward/no-inference), i.e. `rankme_fwd.npy`.
    """
    entries = _filter_entries_selected_models(list(data.values()))
    def _selected_label_override(entry: dict) -> str | None:
        if entry.get("norm_name") == "no_norm_fwd" and entry.get("mode") == "no_lateral" and np.isclose(entry.get("thetaf"), 0.5):
            return r"Standard ($\theta_f = 0.5$)"
        if entry.get("norm_name") == "no_norm_fwd" and entry.get("mode") == "no_lateral" and np.isclose(entry.get("thetaf"), 0.99):
            return r"Standard ($\theta_f = 0.99$)"
        return None
    _plot_best_rankme_bar_entries(
        entries,
        results_dir,
        width=width,
        save_filename="best_rankme_init_bar_selected_models.pdf",
        rankme_key="rankmes_fwd",
        ylabel="RankMe (init activities)",
        label_override=lambda e: _selected_label_override(e)
        or _series_style(e["mode"], e["thetaf"], e.get("norm_name"))["label"],
        show_xticklabels=False,
        save_base_dir=save_base_dir,
        sort_key=_selected_models_sort_key,
        fig_width=max(
            SELECTED_BAR_PLOT_MIN_WIDTH,
            SELECTED_BAR_PLOT_WIDTH_PER_LABEL * max(len(entries), 1),
        ),
    )
    _plot_connections_legend(
        results_dir,
        width=width,
        save_filename="connections_legend.pdf",
        save_base_dir=save_base_dir,
    )


def plot_best_val_accuracies_bar(
    results_dir: str,
    print_every: int,
    width: int = None,
    thetafs: list = None,
    save_base_dir: str | None = None,
):
    """
    Bar plot of best test accuracies for each plotted series.

    The "best" learning rate per series matches the script's existing convention:
    choose the lr that minimizes mean best test loss, then report the corresponding
    best test accuracy (max over time) averaged over seeds.
    """
    lr_data, best_lr, _, _, _, _ = _compute_best_lrs_from_lrs_iris(
        results_dir, print_every, width=width, thetafs=thetafs
    )
    if not best_lr:
        print("No test accuracy data across learning rates. Run train_iris.py with multiple lrs first.")
        return

    entries = [lr_data[series_key][lr_val] for series_key, lr_val in best_lr.items()]
    _plot_best_test_accuracies_bar_entries(
        entries,
        results_dir,
        width=width,
        save_filename="best_test_accuracies_bar.pdf",
        save_base_dir=save_base_dir,
    )


def plot_best_rankme_bar(
    results_dir: str,
    print_every: int,
    width: int = None,
    thetafs: list = None,
    save_base_dir: str | None = None,
):
    """
    Bar plot of end-of-training RankMe for each plotted series.

    Uses the same best learning rate per series as the test-loss convention and the
    accuracy bar plot: lr that minimizes mean best test loss, then final RankMe per
    seed (end of training), mean ± stderr across seeds.
    """
    lr_data, best_lr, _, _, _, _ = _compute_best_lrs_from_lrs_iris(
        results_dir, print_every, width=width, thetafs=thetafs
    )
    if not best_lr:
        print("No data across learning rates. Run train_iris.py with multiple lrs first.")
        return

    entries = [lr_data[series_key][lr_val] for series_key, lr_val in best_lr.items()]
    _plot_best_rankme_bar_entries(
        entries,
        results_dir,
        width=width,
        save_filename="best_rankme_bar.pdf",
        save_base_dir=save_base_dir,
    )


def plot_selected_best_val_accuracies_bar(
    results_dir: str,
    print_every: int,
    width: int = None,
    thetafs: list = None,
    save_base_dir: str | None = None,
):
    """Extra bar plot (selected models only) using the script's best-lr convention."""
    lr_data, best_lr, _, _, _, _ = _compute_best_lrs_from_lrs_iris(
        results_dir, print_every, width=width, thetafs=thetafs
    )
    if not best_lr:
        print("No test accuracy data across learning rates. Run train_iris.py with multiple lrs first.")
        return

    entries = [lr_data[series_key][lr_val] for series_key, lr_val in best_lr.items()]
    entries = _filter_entries_selected_models(entries)
    def _selected_label_override(entry: dict) -> str | None:
        if entry.get("norm_name") == "no_norm_fwd" and entry.get("mode") == "no_lateral" and np.isclose(entry.get("thetaf"), 0.5):
            return r"Standard ($\theta_f = 0.5$)"
        if entry.get("norm_name") == "no_norm_fwd" and entry.get("mode") == "no_lateral" and np.isclose(entry.get("thetaf"), 0.99):
            return r"Standard ($\theta_f = 0.99$)"
        return None
    _plot_best_test_accuracies_bar_entries(
        entries,
        results_dir,
        width=width,
        save_filename="best_test_accuracies_bar_selected_models.pdf",
        label_override=lambda e: _selected_label_override(e)
        or _series_style(e["mode"], e["thetaf"], e.get("norm_name"))["label"],
        show_xticklabels=False,
        save_base_dir=save_base_dir,
        acc_plot_ymin=SELECTED_ACC_PLOT_YMIN,
        sort_key=_selected_models_sort_key,
        fig_width=max(
            SELECTED_BAR_PLOT_MIN_WIDTH,
            SELECTED_BAR_PLOT_WIDTH_PER_LABEL * max(len(entries), 1),
        ),
    )
    _plot_selected_models_legend(
        entries,
        results_dir,
        width=width,
        save_filename="selected_models_legend.pdf",
        label_override=lambda e: _selected_label_override(e)
        or _series_style(e["mode"], e["thetaf"], e.get("norm_name"))["label"],
        save_base_dir=save_base_dir,
    )
    _plot_connections_legend(
        results_dir,
        width=width,
        save_filename="connections_legend.pdf",
        save_base_dir=save_base_dir,
    )


def plot_selected_best_rankme_bar(
    results_dir: str,
    print_every: int,
    width: int = None,
    thetafs: list = None,
    save_base_dir: str | None = None,
):
    """Extra bar plot (selected models only) using the script's best-lr convention."""
    lr_data, best_lr, _, _, _, _ = _compute_best_lrs_from_lrs_iris(
        results_dir, print_every, width=width, thetafs=thetafs
    )
    if not best_lr:
        print("No data across learning rates. Run train_iris.py with multiple lrs first.")
        return

    entries = [lr_data[series_key][lr_val] for series_key, lr_val in best_lr.items()]
    entries = _filter_entries_selected_models(entries)
    def _selected_label_override(entry: dict) -> str | None:
        if entry.get("norm_name") == "no_norm_fwd" and entry.get("mode") == "no_lateral" and np.isclose(entry.get("thetaf"), 0.5):
            return r"Standard ($\theta_f = 0.5$)"
        if entry.get("norm_name") == "no_norm_fwd" and entry.get("mode") == "no_lateral" and np.isclose(entry.get("thetaf"), 0.99):
            return r"Standard ($\theta_f = 0.99$)"
        return None
    _plot_best_rankme_bar_entries(
        entries,
        results_dir,
        width=width,
        save_filename="best_rankme_bar_selected_models.pdf",
        label_override=lambda e: _selected_label_override(e)
        or _series_style(e["mode"], e["thetaf"], e.get("norm_name"))["label"],
        show_xticklabels=False,
        save_base_dir=save_base_dir,
        sort_key=_selected_models_sort_key,
        fig_width=max(
            SELECTED_BAR_PLOT_MIN_WIDTH,
            SELECTED_BAR_PLOT_WIDTH_PER_LABEL * max(len(entries), 1),
        ),
    )
    _plot_selected_models_legend(
        entries,
        results_dir,
        width=width,
        save_filename="selected_models_legend.pdf",
        label_override=lambda e: _selected_label_override(e)
        or _series_style(e["mode"], e["thetaf"], e.get("norm_name"))["label"],
        save_base_dir=save_base_dir,
    )


def plot_selected_best_rankme_init_bar(
    results_dir: str,
    print_every: int,
    width: int = None,
    thetafs: list = None,
    save_base_dir: str | None = None,
):
    """Same as `plot_selected_best_rankme_bar`, but for `rankme_fwd.npy`."""
    lr_data, best_lr, _, _, _, _ = _compute_best_lrs_from_lrs_iris(
        results_dir, print_every, width=width, thetafs=thetafs
    )
    if not best_lr:
        print("No losses found across learning rates. Run train_iris.py with multiple lrs first.")
        return

    entries = [lr_data[series_key][lr_val] for series_key, lr_val in best_lr.items()]
    entries = _filter_entries_selected_models(entries)
    def _selected_label_override(entry: dict) -> str | None:
        if entry.get("norm_name") == "no_norm_fwd" and entry.get("mode") == "no_lateral" and np.isclose(entry.get("thetaf"), 0.5):
            return r"Standard ($\theta_f = 0.5$)"
        if entry.get("norm_name") == "no_norm_fwd" and entry.get("mode") == "no_lateral" and np.isclose(entry.get("thetaf"), 0.99):
            return r"Standard ($\theta_f = 0.99$)"
        return None
    _plot_best_rankme_bar_entries(
        entries,
        results_dir,
        width=width,
        save_filename="best_rankme_init_bar_selected_models.pdf",
        rankme_key="rankmes_fwd",
        ylabel="RankMe (init activities)",
        label_override=lambda e: _selected_label_override(e)
        or _series_style(e["mode"], e["thetaf"], e.get("norm_name"))["label"],
        show_xticklabels=False,
        save_base_dir=save_base_dir,
        sort_key=_selected_models_sort_key,
        fig_width=max(
            SELECTED_BAR_PLOT_MIN_WIDTH,
            SELECTED_BAR_PLOT_WIDTH_PER_LABEL * max(len(entries), 1),
        ),
    )
    _plot_connections_legend(
        results_dir,
        width=width,
        save_filename="connections_legend.pdf",
        save_base_dir=save_base_dir,
    )


def _plot_selected_models_legend(
    entries: list,
    results_dir: str,
    width: int = None,
    save_filename: str = "selected_models_legend.pdf",
    label_override=None,
    save_base_dir: str | None = None,
):
    """Create a legend-only plot for the selected-models bars (colors + labels)."""
    if not entries:
        return
    from matplotlib.lines import Line2D

    # Preserve the same ordering as the bar plots.
    entries = sorted(entries, key=_bar_series_sort_key)

    handles = []
    labels = []
    series_colors = {}
    for entry in entries:
        style = _series_style(entry["mode"], entry["thetaf"], entry.get("norm_name"), series_colors)
        label = (
            label_override(entry) if callable(label_override) else style["label"]
        )
        if label is None:
            label = style["label"]
        # Standalone legend: expand abbreviation for readability.
        label = label.replace("Norm.", "Normalised")
        labels.append(label)
        handles.append(Line2D([0], [0], color=style["color"], lw=6))

    # Compact legend-only figure.
    sns.set_theme(context="talk")
    fig, ax = plt.subplots(figsize=(7.0, 4.5))
    ax.axis("off")
    ax.legend(
        handles,
        labels,
        loc="center",
        frameon=False,
        fontsize=20,
        handlelength=2.5,
    )
    plt.tight_layout()

    save_base = save_base_dir if save_base_dir is not None else _summary_save_base(results_dir, width=width)
    os.makedirs(save_base, exist_ok=True)
    save_path = os.path.join(save_base, save_filename)
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved {save_path}")


def _plot_connections_legend(
    results_dir: str,
    width: int = None,
    save_filename: str = "connections_legend.pdf",
    save_base_dir: str | None = None,
):
    """Legend-only plot describing connection types (shared across selected plots)."""
    from matplotlib.lines import Line2D

    sns.set_theme(context="talk")
    fig, ax = plt.subplots(figsize=(7.0, 2.8))
    ax.axis("off")

    handles = [
        Line2D([0], [0], color="black", lw=6),
        Line2D([0], [0], color="#2ca02c", lw=6),  # match toy nonlinear green
    ]
    labels = ["Fwd. & bwd. connections", "Lateral connections"]
    ax.legend(
        handles,
        labels,
        loc="center",
        frameon=False,
        fontsize=20,
        handlelength=2.5,
    )
    plt.tight_layout()

    save_base = save_base_dir if save_base_dir is not None else _summary_save_base(results_dir, width=width)
    os.makedirs(save_base, exist_ok=True)
    save_path = os.path.join(save_base, save_filename)
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved {save_path}")


def plot_best_val_loss_vs_lr(
    results_dir: str,
    print_every: int,
    log_scale_y: bool = False,
    width: int = None,
    thetafs: list = None,
    save_base_dir: str | None = None,
):
    """Plot best (lowest) test loss vs learning rate for each plotted series."""
    sns.set_theme(context="talk")
    lr_data, _, _, best_val_means, _, _ = _compute_best_lrs_from_lrs_iris(
        results_dir, print_every, width=width, thetafs=thetafs
    )
    if not any(best_val_means.values()):
        print("No test-loss data across learning rates. Run train_iris.py with multiple lrs first.")
        return

    fig, ax = plt.subplots(figsize=(METRIC_PLOT_WIDTH, METRIC_PLOT_HEIGHT))
    series_colors = {}
    for series_key, values in best_val_means.items():
        if not values:
            continue
        example_entry = next(iter(lr_data[series_key].values()))
        style = _series_style(
            example_entry["mode"], example_entry["thetaf"], example_entry.get("norm_name"), series_colors
        )
        lrs = sorted(values.keys())
        vals = [values[lr] for lr in lrs]
        ax.plot(
            lrs,
            vals,
            "-o",
            color=style["color"],
            linestyle=style["linestyle"],
            label=style["label"],
            linewidth=2,
        )
    ax.set_xscale("log")
    if log_scale_y:
        all_vals = []
        for values in best_val_means.values():
            if values:
                all_vals.extend(values.values())
        y = np.asarray(all_vals) if all_vals else np.array([])
        if y.size > 0 and np.nanmin(y) > 0:
            ax.set_yscale("log")
    ax.set_xlabel("Learning rate", fontsize=AXIS_LABEL_FONTSIZE, labelpad=10)
    ax.set_ylabel("Best test loss", fontsize=AXIS_LABEL_FONTSIZE, labelpad=10)
    ax.legend(
        loc="upper left",
        bbox_to_anchor=(1.02, 1.0),
        borderaxespad=0.0,
        fontsize=LEGEND_FONTSIZE,
    )
    ax.tick_params(axis="both", labelsize=TICK_LABEL_FONTSIZE)
    ax.grid(True, alpha=0.3)
    plt.tight_layout(rect=(0, 0, 0.82, 1))
    save_base = save_base_dir if save_base_dir is not None else _summary_save_base(results_dir, width=width)
    os.makedirs(save_base, exist_ok=True)
    save_path = os.path.join(save_base, "best_test_loss_vs_lr.pdf")
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved {save_path}")


def plot_best_val_acc_vs_lr(
    results_dir: str,
    print_every: int,
    width: int = None,
    thetafs: list = None,
    save_base_dir: str | None = None,
):
    """Plot best (highest) test accuracy vs learning rate for each plotted series."""
    sns.set_theme(context="talk")
    lr_data, _, _, _, best_val_acc_means, _ = _compute_best_lrs_from_lrs_iris(
        results_dir, print_every, width=width, thetafs=thetafs
    )
    if not any(best_val_acc_means.values()):
        print("No test-accuracy data across learning rates. Run train_iris.py with multiple lrs first.")
        return

    fig, ax = plt.subplots(figsize=(METRIC_PLOT_WIDTH, METRIC_PLOT_HEIGHT))
    series_colors = {}
    for series_key, values in best_val_acc_means.items():
        if not values:
            continue
        example_entry = next(iter(lr_data[series_key].values()))
        style = _series_style(
            example_entry["mode"], example_entry["thetaf"], example_entry.get("norm_name"), series_colors
        )
        lrs = sorted(values.keys())
        vals = [values[lr] for lr in lrs]
        ax.plot(
            lrs,
            vals,
            "-o",
            color=style["color"],
            linestyle=style["linestyle"],
            label=style["label"],
            linewidth=2,
        )
    ax.set_xscale("log")
    ax.set_ylabel("Best test acc.", fontsize=AXIS_LABEL_FONTSIZE, labelpad=10)
    ax.set_xlabel("Learning rate", fontsize=AXIS_LABEL_FONTSIZE, labelpad=10)
    ax.legend(
        loc="upper left",
        bbox_to_anchor=(1.02, 1.0),
        borderaxespad=0.0,
        fontsize=LEGEND_FONTSIZE,
    )
    ax.tick_params(axis="both", labelsize=TICK_LABEL_FONTSIZE)
    ax.set_ylim(ACC_PLOT_YMIN, 1.0)
    ax.grid(True, alpha=0.3)
    plt.tight_layout(rect=(0, 0, 0.82, 1))
    save_base = save_base_dir if save_base_dir is not None else _summary_save_base(results_dir, width=width)
    os.makedirs(save_base, exist_ok=True)
    save_path = os.path.join(save_base, "best_test_acc_vs_lr.pdf")
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved {save_path}")


def plot_rankme_best_lr_over_lrs(
    results_dir: str,
    print_every: int,
    max_steps=None,
    width: int = None,
    thetafs: list = None,
    save_base_dir: str | None = None,
):
    """Plot RankMe over training at the lr with highest RankMe for each series."""
    sns.set_theme(context="talk")
    lr_data, _, best_rankme_lr, _, _, _ = _compute_best_lrs_from_lrs_iris(
        results_dir, print_every, width=width, thetafs=thetafs
    )
    if not best_rankme_lr:
        print("No RankMe data across learning rates. Run train_iris.py with multiple lrs first.")
        return

    fig, ax = plt.subplots(figsize=(METRIC_PLOT_WIDTH, METRIC_PLOT_HEIGHT))
    series_colors = {}
    for series_key, lr_val in best_rankme_lr.items():
        entry = lr_data[series_key][lr_val]
        rankmes = entry.get("rankmes")
        if rankmes is None or rankmes.size == 0:
            continue
        n_train_iters = _true_n_train_iters(entry, print_every)
        # Shade mean ± standard error across seeds for consistency with other metrics.
        center, stderr = _mean_stderr(rankmes)
        low, high = center - stderr, center + stderr
        steps = test_steps_from_length(n_train_iters, print_every, len(center))
        if max_steps is not None:
            mask = steps <= max_steps
            if mask.any():
                steps, center, low, high = steps[mask], center[mask], low[mask], high[mask]
        style = _series_style(entry["mode"], entry["thetaf"], entry.get("norm_name"), series_colors)
        ax.plot(
            steps,
            center,
            color=style["color"],
            linestyle=style["linestyle"],
            label=style["label"],
            linewidth=2,
        )
        if rankmes.ndim > 1 and rankmes.shape[0] > 1:
            ax.fill_between(steps, low, high, color=style["color"], alpha=0.18)
    ax.set_xlabel("Training step", fontsize=AXIS_LABEL_FONTSIZE, labelpad=10)
    ax.set_ylabel("RankMe", fontsize=AXIS_LABEL_FONTSIZE, labelpad=10)
    ax.legend(
        loc="upper left",
        bbox_to_anchor=(1.02, 1.0),
        borderaxespad=0.0,
        fontsize=LEGEND_FONTSIZE,
    )
    ax.tick_params(axis="both", labelsize=TICK_LABEL_FONTSIZE)
    ax.set_xlim(left=0)
    ax.grid(True, alpha=0.3)
    plt.tight_layout(rect=(0, 0, 0.8, 1))
    save_base = save_base_dir if save_base_dir is not None else _summary_save_base(results_dir, width=width)
    os.makedirs(save_base, exist_ok=True)
    save_path = os.path.join(save_base, "best_rankme.pdf")
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved {save_path}")


def main():
    parser = argparse.ArgumentParser(description="Plot Iris lateral PDM results aggregated over seeds.")
    parser.add_argument(
        "--results_dir",
        type=str,
        default="results/iris",
        help=(
            "Results directory. Supports both layouts:\n"
            "  - old: <results_dir>/(no_lateral|with_lateral)/...\n"
            "  - new: <results_dir>/(norm_fwd|no_norm_fwd)/(no_lateral|with_lateral)/..."
        ),
    )
    parser.add_argument(
        "--norm_name",
        type=str,
        default=None,
        choices=["norm_fwd", "no_norm_fwd"],
        help=(
            "When results_dir contains norm_fwd/ and/or no_norm_fwd/, choose which one to plot. "
            "If omitted, the script plots both if both are present (otherwise plots the one that exists)."
        ),
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=None,
        help="Learning rate to plot (matches lr_<lr> subdirectories). "
        "If omitted, all lr_* subdirectories under results_dir/no_lateral/ are processed.",
    )
    parser.add_argument(
        "--print_every",
        type=int,
        default=100,
        help="Print/eval interval used during training (for val loss x-axis).",
    )
    parser.add_argument(
        "--log_loss_scale",
        action="store_true",
        default=True,
        help="Use log scale for loss y-axis (default: log).",
    )
    parser.add_argument(
        "--max_steps",
        type=int,
        default=None,
        help="Maximum training step to show in loss plots (crop curves).",
    )
    parser.add_argument(
        "--thetaf",
        type=float,
        default=None,
        help="Specific thetaf to plot for PDM runs. If omitted, discover all thetaf_* dirs.",
    )
    parser.add_argument(
        "--skip_separability_runs",
        action="store_true",
        default=False,
        help="Skip per-run margin/activity/weight-space PDFs generation (enabled by default).",
    )
    parser.add_argument(
        "--w2f_data_pc_autoscale",
        action="store_true",
        default=False,
        help="Autoscale axes for w2f_data_pc_projections.pdf based on plotted points (default: fixed limits).",
    )
    args = parser.parse_args()
    print(f"[plot_iris] skip_separability_runs={args.skip_separability_runs}")
    print(f"[plot_iris] results_dir={args.results_dir} norm_name={args.norm_name}")

    # Decide which result trees to process.
    if args.norm_name is None:
        # Default: if both norm_fwd and no_norm_fwd exist, plot both.
        norm_dirs = _discover_norm_dirs(args.results_dir, requested=None)
        if not norm_dirs:
            # Fall back to legacy/flat layout.
            norm_dirs = [(args.results_dir, None)]
    else:
        resolved = _resolve_results_dir(args.results_dir, norm_name=args.norm_name)
        _, embedded = _norm_root_and_name(resolved)
        norm_dirs = [(resolved, embedded)]

    # If we have both norms available and the user didn't force one, overlay them in the
    # same figures and write to a dedicated comparison folder.
    have_two_norms = (
        args.norm_name is None
        and len(norm_dirs) >= 2
        and any(n == "norm_fwd" for _, n in norm_dirs)
        and any(n == "no_norm_fwd" for _, n in norm_dirs)
    )

    if have_two_norms:
        root, embedded = _norm_root_and_name(norm_dirs[0][0])
        root_results_dir = root if embedded is not None else args.results_dir

        # Choose a representative results_dir for width/thetaf discovery.
        rep_results_dir, _ = norm_dirs[0]

        widths = _discover_widths_from_results(rep_results_dir)
        if not widths:
            no_lateral_dir = os.path.join(rep_results_dir, "no_lateral")
            if not os.path.isdir(no_lateral_dir):
                print(
                    f"No width_* or lr_* subdirectories found under {no_lateral_dir}. Run train_iris.py first."
                )
                return
            widths = [None]

        for width in widths:
            thetafs = (
                [args.thetaf]
                if args.thetaf is not None
                else _discover_thetafs_from_results(rep_results_dir, width=width)
            )
            if not thetafs:
                width_label = f"width_{width}" if width is not None else "flat"
                print(f"No thetaf_* or flat lr_* directories for {width_label}, skipping.")
                continue

            save_base_dir = _combined_save_base(root_results_dir, width=width)
            os.makedirs(save_base_dir, exist_ok=True)

            # Decide which layout we are in by checking any norm directory.
            gridsearch_layout = any(
                _width_has_best_lr_training_runs(rd, width) for rd, _ in norm_dirs
            )

            if args.lr is None and gridsearch_layout:
                width_label = f"width_{width}" if width is not None else "flat"
                thetaf_label = ", ".join(
                    f"{thetaf:g}" if thetaf is not None else "legacy" for thetaf in thetafs
                )
                print(
                    f"\nProcessing Iris results for {width_label}, thetafs=[{thetaf_label}] "
                    f"(gridsearch-best curves, combined norms)"
                )

                combined = {}
                source_results_dir_by_series_key = {}
                for rd, nn in norm_dirs:
                    data = load_losses_best_selected(rd, width=width, thetafs=thetafs, norm_name=nn)
                    for series_key, entry in data.items():
                        if entry["mode"] == "backprop":
                            # Backprop is shared; avoid duplicating it across norms.
                            combined[_series_key("backprop", None)] = {
                                **entry,
                                "norm_name": None,
                            }
                        else:
                            sk = _series_key_with_norm(nn, entry["mode"], entry["thetaf"])
                            combined[sk] = entry
                            source_results_dir_by_series_key[sk] = rd

                if not combined:
                    print(f"  No gridsearch best-selected data found for {width_label}; skipping.")
                    continue

                lr_label = "gridsearch_best"
                plot_val_losses(
                    combined,
                    results_dir=rep_results_dir,
                    lr=lr_label,
                    print_every=args.print_every,
                    log_scale=args.log_loss_scale,
                    max_steps=args.max_steps,
                    width=width,
                    thetafs=thetafs,
                    save_base_dir=save_base_dir,
                )
                plot_train_losses(
                    combined,
                    results_dir=rep_results_dir,
                    lr=lr_label,
                    log_scale=args.log_loss_scale,
                    max_steps=args.max_steps,
                    width=width,
                    thetafs=thetafs,
                    save_base_dir=save_base_dir,
                )
                plot_val_accuracies(
                    combined,
                    results_dir=rep_results_dir,
                    lr=lr_label,
                    print_every=args.print_every,
                    max_steps=args.max_steps,
                    width=width,
                    thetafs=thetafs,
                    save_base_dir=save_base_dir,
                )
                plot_rankme(
                    combined,
                    results_dir=rep_results_dir,
                    lr=lr_label,
                    print_every=args.print_every,
                    max_steps=args.max_steps,
                    width=width,
                    thetafs=thetafs,
                    save_base_dir=save_base_dir,
                )
                plot_best_test_accuracies_bar_from_data(
                    combined, results_dir=rep_results_dir, width=width, save_base_dir=save_base_dir
                )
                plot_best_rankme_bar_from_data(
                    combined, results_dir=rep_results_dir, width=width, save_base_dir=save_base_dir
                )
                plot_selected_best_test_accuracies_bar_from_data(
                    combined, results_dir=rep_results_dir, width=width, save_base_dir=save_base_dir
                )
                plot_selected_best_rankme_bar_from_data(
                    combined, results_dir=rep_results_dir, width=width, save_base_dir=save_base_dir
                )
                plot_selected_best_rankme_init_bar_from_data(
                    combined, results_dir=rep_results_dir, width=width, save_base_dir=save_base_dir
                )
                plot_lateral_weight_distributions(
                    combined,
                    results_dir=rep_results_dir,
                    lr=lr_label,
                    width=width,
                    thetafs=thetafs,
                    save_base_dir=save_base_dir,
                    source_results_dir_by_series_key=source_results_dir_by_series_key,
                )
                continue

            if args.lr is not None:
                lrs = [args.lr]
            else:
                # Union LRs across norms.
                all_lrs = set()
                for rd, _ in norm_dirs:
                    all_lrs.update(_discover_all_lrs_from_results(rd, width=width, thetafs=thetafs))
                lrs = sorted(all_lrs)
                if not lrs:
                    continue

            width_label = f"width_{width}" if width is not None else "flat"
            thetaf_label = ", ".join(
                f"{thetaf:g}" if thetaf is not None else "legacy" for thetaf in thetafs
            )
            print(
                f"\nProcessing Iris results for {width_label}, thetafs=[{thetaf_label}] (combined norms)"
            )

            for lr in lrs:
                print(f"  lr={lr}")
                combined = {}
                source_results_dir_by_series_key = {}
                for rd, nn in norm_dirs:
                    data = load_losses_for_lr(
                        rd,
                        lr,
                        args.print_every,
                        width=width,
                        thetafs=thetafs,
                        norm_name=nn,
                    )
                    for series_key, entry in data.items():
                        if entry["mode"] == "backprop":
                            # Backprop is shared; avoid duplicating it across norms.
                            combined[_series_key("backprop", None)] = {
                                **entry,
                                "norm_name": None,
                            }
                        else:
                            sk = _series_key_with_norm(nn, entry["mode"], entry["thetaf"])
                            combined[sk] = entry
                            source_results_dir_by_series_key[sk] = rd

                if not combined:
                    print(f"    No data for lr={lr}. Skipping.")
                    continue

                plot_val_losses(
                    combined,
                    results_dir=rep_results_dir,
                    lr=lr,
                    print_every=args.print_every,
                    log_scale=args.log_loss_scale,
                    max_steps=args.max_steps,
                    width=width,
                    thetafs=thetafs,
                    save_base_dir=save_base_dir,
                )
                plot_train_losses(
                    combined,
                    results_dir=rep_results_dir,
                    lr=lr,
                    log_scale=args.log_loss_scale,
                    max_steps=args.max_steps,
                    width=width,
                    thetafs=thetafs,
                    save_base_dir=save_base_dir,
                )
                plot_val_accuracies(
                    combined,
                    results_dir=rep_results_dir,
                    lr=lr,
                    print_every=args.print_every,
                    max_steps=args.max_steps,
                    width=width,
                    thetafs=thetafs,
                    save_base_dir=save_base_dir,
                )
                plot_rankme(
                    combined,
                    results_dir=rep_results_dir,
                    lr=lr,
                    print_every=args.print_every,
                    max_steps=args.max_steps,
                    width=width,
                    thetafs=thetafs,
                    save_base_dir=save_base_dir,
                )
                plot_lateral_weight_distributions(
                    combined,
                    results_dir=rep_results_dir,
                    lr=lr,
                    width=width,
                    thetafs=thetafs,
                    save_base_dir=save_base_dir,
                    source_results_dir_by_series_key=source_results_dir_by_series_key,
                )

            if args.lr is None and len(lrs) > 0:
                print(f"\nPlotting best-* and vs-lr summary for {width_label} (combined norms)...")
                # NOTE: For combined norms, the best-* summaries are not computed here because
                # the underlying best-lr search is per results_dir tree. Use gridsearch-best
                # plots (default when present) or run with --norm_name to compute per-norm best-lr.
                # We still provide the per-lr overlays and bar plots above.
        # Done with combined-norm plots; continue into per-run separability generation below.

    for results_dir, norm_name in norm_dirs:
        norm_label = f"{norm_name}/" if norm_name is not None else ""

        # Discover widths (no_lateral/width_*). If none, try flat no_lateral/lr_* for backward compat.
        widths = _discover_widths_from_results(results_dir)
        if not widths:
            no_lateral_dir = os.path.join(results_dir, "no_lateral")
            if not os.path.isdir(no_lateral_dir):
                print(
                    f"No width_* or lr_* subdirectories found under {no_lateral_dir}. Run train_iris.py first."
                )
                continue
            # Flat layout: single "width" represented as None
            widths = [None]

        for width in widths:
            thetafs = (
                [args.thetaf]
                if args.thetaf is not None
                else _discover_thetafs_from_results(results_dir, width=width)
            )
            if not thetafs:
                width_label = f"width_{width}" if width is not None else "flat"
                print(f"No thetaf_* or flat lr_* directories for {norm_label}{width_label}, skipping.")
                continue

            # Grid-search training layout: seed_*/best_lr_*/<run>/train_losses.npy.
            # Do not gate on "no lr_* under backprop" — stale plot dirs can leave lr_* behind.
            gridsearch_layout = _width_has_best_lr_training_runs(results_dir, width)

            if args.lr is None and gridsearch_layout:
                width_label = f"width_{width}" if width is not None else "flat"
                thetaf_label = ", ".join(
                    f"{thetaf:g}" if thetaf is not None else "legacy" for thetaf in thetafs
                )
                print(
                    f"\nProcessing Iris results in {results_dir} for {width_label}, "
                    f"thetafs=[{thetaf_label}] (gridsearch-best curves)"
                )
                data = load_losses_best_selected(
                    results_dir, width=width, thetafs=thetafs, norm_name=norm_name
                )
                if not data:
                    print(f"  No gridsearch best-selected data found for {width_label}; skipping.")
                    continue

                lr_label = "gridsearch_best"
                plot_val_losses(
                    data,
                    results_dir=results_dir,
                    lr=lr_label,
                    print_every=args.print_every,
                    log_scale=args.log_loss_scale,
                    max_steps=args.max_steps,
                    width=width,
                    thetafs=thetafs,
                )
                plot_train_losses(
                    data,
                    results_dir=results_dir,
                    lr=lr_label,
                    log_scale=args.log_loss_scale,
                    max_steps=args.max_steps,
                    width=width,
                    thetafs=thetafs,
                )
                plot_val_accuracies(
                    data,
                    results_dir=results_dir,
                    lr=lr_label,
                    print_every=args.print_every,
                    max_steps=args.max_steps,
                    width=width,
                    thetafs=thetafs,
                )
                plot_rankme(
                    data,
                    results_dir=results_dir,
                    lr=lr_label,
                    print_every=args.print_every,
                    max_steps=args.max_steps,
                    width=width,
                    thetafs=thetafs,
                )
                plot_best_test_accuracies_bar_from_data(data, results_dir=results_dir, width=width)
                plot_best_rankme_bar_from_data(data, results_dir=results_dir, width=width)
                plot_selected_best_test_accuracies_bar_from_data(
                    data, results_dir=results_dir, width=width
                )
                plot_selected_best_rankme_bar_from_data(data, results_dir=results_dir, width=width)
                plot_selected_best_rankme_init_bar_from_data(
                    data, results_dir=results_dir, width=width
                )
                plot_lateral_weight_distributions(
                    data,
                    results_dir=results_dir,
                    lr=lr_label,
                    width=width,
                    thetafs=thetafs,
                )
                continue

            if args.lr is not None:
                lrs = [args.lr]
            else:
                lrs = _discover_all_lrs_from_results(results_dir, width=width, thetafs=thetafs)
                if not lrs:
                    continue

            width_label = f"width_{width}" if width is not None else "flat"
            thetaf_label = ", ".join(
                f"{thetaf:g}" if thetaf is not None else "legacy" for thetaf in thetafs
            )
            print(
                f"\nProcessing Iris results in {results_dir} for {width_label}, thetafs=[{thetaf_label}]"
            )

            for lr in lrs:
                print(f"  lr={lr}")
                data = load_losses_for_lr(
                    results_dir,
                    lr,
                    args.print_every,
                    width=width,
                    thetafs=thetafs,
                    norm_name=norm_name,
                )

                if not data:
                    print(f"    No data for lr={lr}. Skipping.")
                    continue

                plot_val_losses(
                    data,
                    results_dir=results_dir,
                    lr=lr,
                    print_every=args.print_every,
                    log_scale=args.log_loss_scale,
                    max_steps=args.max_steps,
                    width=width,
                    thetafs=thetafs,
                )
                plot_train_losses(
                    data,
                    results_dir=results_dir,
                    lr=lr,
                    log_scale=args.log_loss_scale,
                    max_steps=args.max_steps,
                    width=width,
                    thetafs=thetafs,
                )
                plot_val_accuracies(
                    data,
                    results_dir=results_dir,
                    lr=lr,
                    print_every=args.print_every,
                    max_steps=args.max_steps,
                    width=width,
                    thetafs=thetafs,
                )
                plot_rankme(
                    data,
                    results_dir=results_dir,
                    lr=lr,
                    print_every=args.print_every,
                    max_steps=args.max_steps,
                    width=width,
                    thetafs=thetafs,
                )
                plot_lateral_weight_distributions(
                    data,
                    results_dir=results_dir,
                    lr=lr,
                    width=width,
                    thetafs=thetafs,
                )

            if args.lr is None and len(lrs) > 0:
                print(f"\nPlotting best-* and vs-lr summary for {width_label}...")
                plot_train_loss_best_lr_over_lrs(
                    results_dir,
                    print_every=args.print_every,
                    log_scale=args.log_loss_scale,
                    max_steps=args.max_steps,
                    width=width,
                    thetafs=thetafs,
                )
                plot_val_loss_best_lr_over_lrs(
                    results_dir,
                    print_every=args.print_every,
                    log_scale=args.log_loss_scale,
                    max_steps=args.max_steps,
                    width=width,
                    thetafs=thetafs,
                )
                plot_val_acc_best_lr_over_lrs(
                    results_dir,
                    print_every=args.print_every,
                    max_steps=args.max_steps,
                    width=width,
                    thetafs=thetafs,
                )
                plot_best_val_accuracies_bar(
                    results_dir,
                    print_every=args.print_every,
                    width=width,
                    thetafs=thetafs,
                )
                plot_best_rankme_bar(
                    results_dir,
                    print_every=args.print_every,
                    width=width,
                    thetafs=thetafs,
                )
                plot_selected_best_val_accuracies_bar(
                    results_dir,
                    print_every=args.print_every,
                    width=width,
                    thetafs=thetafs,
                )
                plot_selected_best_rankme_bar(
                    results_dir,
                    print_every=args.print_every,
                    width=width,
                    thetafs=thetafs,
                )
                plot_best_val_loss_vs_lr(
                    results_dir,
                    print_every=args.print_every,
                    log_scale_y=args.log_loss_scale,
                    width=width,
                    thetafs=thetafs,
                )
                plot_best_val_acc_vs_lr(
                    results_dir,
                    print_every=args.print_every,
                    width=width,
                    thetafs=thetafs,
                )
                plot_rankme_best_lr_over_lrs(
                    results_dir,
                    print_every=args.print_every,
                    max_steps=args.max_steps,
                    width=width,
                    thetafs=thetafs,
                )

    if not args.skip_separability_runs:
        # Create per-run PDFs anywhere separability_metrics.npz exists.
        for resolved_dir, embedded_norm in norm_dirs:
            print(f"Generating per-run separability PDFs under {resolved_dir} ...")
            plot_separability_all_runs(
                resolved_dir,
                norm_name=embedded_norm,
                w2f_data_pc_autoscale=bool(args.w2f_data_pc_autoscale),
            )
        # Backprop runs are stored in a sibling directory (typically <root>/backprop),
        # not under norm_fwd/ or no_norm_fwd/. When we plot normed trees, explicitly
        # scan backprop as well so BP gets per-run PCA/LDA PDFs.
        if any(embedded is not None for _, embedded in norm_dirs):
            backprop_root = os.path.join(args.results_dir, "backprop")
            if os.path.isdir(backprop_root):
                print(f"Generating per-run separability PDFs under {backprop_root} ...")
                plot_separability_all_runs(backprop_root, norm_name=None)


if __name__ == "__main__":
    main()

