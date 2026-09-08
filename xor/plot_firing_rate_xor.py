#!/usr/bin/env python3
"""Plot firing-rate XOR curves from results/firing_rate.csv."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

SCRIPT_DIR = Path(__file__).resolve().parent
RESULTS_PATH = SCRIPT_DIR / "results" / "firing_rate.csv"
PLOTS_PATH = SCRIPT_DIR / "plots" / "firing_rate_xor.pdf"

PLOT_FIG_W_IN = 8.0
PLOT_FIG_H_IN = 5.0
AXIS_LABEL_FONTSIZE = 30
TICK_LABEL_FONTSIZE = 25
LEGEND_FONTSIZE = 18
LINE_WIDTH = 1.75
SAVE_DPI = 150
SAVE_PAD_IN = 0.10

DISPLAY_NAMES = {
    "predictive_dendrites": "Predictive dendrites",
    "hebbian": "Hebbian",
    "bounded_hebbian": "Bounded Hebbian",
}
HUE_ORDER = ["Predictive dendrites", "Hebbian", "Bounded Hebbian"]
LEGEND_ORDER = ["Hebbian", "Bounded Hebbian", "Predictive dendrites"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot firing-rate XOR MSE curves from a long-form CSV."
    )
    parser.add_argument("--results-path", type=Path, default=RESULTS_PATH)
    parser.add_argument("--output", type=Path, default=PLOTS_PATH)
    parser.add_argument(
        "--lr",
        type=float,
        default=None,
        help="Learning rate to plot. Defaults to the only lr in the CSV, or 0.1.",
    )
    parser.add_argument(
        "--max-y",
        type=float,
        default=10**12,
        help="Max y-axis value (for log scale).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    df = pd.read_csv(args.results_path)
    if df.empty:
        raise ValueError(f"No rows in {args.results_path}")

    lr = args.lr
    if lr is None:
        unique_lrs = sorted(df["lr"].unique())
        lr = 0.1 if 0.1 in unique_lrs else float(unique_lrs[0])

    df = df[(df["lr"] - lr).abs() < 1e-12].copy()
    if df.empty:
        raise ValueError(f"No rows for lr={lr} in {args.results_path}")

    plot_df = df.rename(columns={"step": "Training epoch", "mse": "Mean squared error"})
    plot_df["Model"] = plot_df["model"].map(DISPLAY_NAMES)
    missing = plot_df["Model"].isna()
    if missing.any():
        unknown = sorted(plot_df.loc[missing, "model"].unique())
        raise ValueError(f"Unknown model names in CSV: {unknown}")

    sns.set_theme(context="talk")
    g = sns.relplot(
        data=plot_df,
        kind="line",
        x="Training epoch",
        y="Mean squared error",
        hue="Model",
        hue_order=HUE_ORDER,
        linewidth=LINE_WIDTH,
        height=PLOT_FIG_H_IN,
        aspect=PLOT_FIG_W_IN / PLOT_FIG_H_IN,
    )
    if g._legend is not None:
        handles = g._legend.legend_handles
        labels = [t.get_text() for t in g._legend.texts]
        g._legend.remove()
        by_label = dict(zip(labels, handles))
        ordered_handles = [by_label[name] for name in LEGEND_ORDER if name in by_label]
        ordered_labels = [name for name in LEGEND_ORDER if name in by_label]
        for ax in g.axes.flat:
            ax.legend(
                ordered_handles,
                ordered_labels,
                title=None,
                loc="upper right",
                fontsize=LEGEND_FONTSIZE,
            )
    g.set(yscale="log")
    if args.max_y is not None:
        for ax in g.axes.flat:
            ax.set_ylim(top=float(args.max_y))

    ax0 = g.axes.flat[0]
    ax0.set_xlabel("Training epoch", fontsize=AXIS_LABEL_FONTSIZE, labelpad=10)
    ax0.set_ylabel("Mean squared error", fontsize=AXIS_LABEL_FONTSIZE, labelpad=10)
    ax0.tick_params(axis="both", labelsize=TICK_LABEL_FONTSIZE)
    ax0.grid(True, alpha=0.3)
    g.figure.set_size_inches(PLOT_FIG_W_IN, PLOT_FIG_H_IN)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    g.savefig(args.output, dpi=SAVE_DPI, bbox_inches="tight", pad_inches=SAVE_PAD_IN)
    plt.close("all")

    summary = (
        df.groupby("model", as_index=False)["mse"]
        .min()
        .rename(columns={"mse": "min_mse"})
        .sort_values("min_mse")
        .reset_index(drop=True)
    )
    summary["model"] = summary["model"].map(DISPLAY_NAMES)
    print(f"Loaded results from {args.results_path} (lr={lr})")
    print(f"Saved plot to {args.output}")
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
