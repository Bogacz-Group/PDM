#!/usr/bin/env python3
"""Plot Supplementary Fig. 3c from the long-form trajectory CSV."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
import pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parent
RESULTS_PATH = SCRIPT_DIR / "results" / "supplementary_figure3c.csv"
PDF_PATH = SCRIPT_DIR / "plots" / "supplementary_figure3c.pdf"
SVG_PATH = SCRIPT_DIR / "plots" / "supplementary_figure3c.svg"

REQUIRED_COLUMNS = {
    "seed",
    "initial_wf1",
    "initial_wf2",
    "dataset_iteration",
    "wf1",
    "wf2",
}


def add_feature_regions(ax: plt.Axes) -> None:
    """Overlay the two firing-rate XOR feature regions from panel 3b."""

    # Active only for (0, 1): wf2 > 0 and wf1 + wf2 <= 0.
    blue = [(-1.0, 0.0), (-1.0, 1.0), (0.0, 0.0), (-1.0, 0.0)]
    # Active only for (1, 0): wf1 > 0 and wf1 + wf2 <= 0.
    green = [(0.0, 0.0), (1.0, -1.0), (0.0, -1.0), (0.0, 0.0)]
    for vertices, color in ((blue, "#00a6ca"), (green, "#009e73")):
        xs, ys = zip(*vertices)
        ax.plot(xs, ys, color=color, linestyle="--", linewidth=2.0, zorder=4)


def plot_trajectories(
    results_path: Path = RESULTS_PATH,
    *,
    pdf_path: Path = PDF_PATH,
    svg_path: Path = SVG_PATH,
    show_feature_regions: bool = True,
) -> None:
    """Read results, render one panel, and save PDF and SVG outputs."""

    frame = pd.read_csv(results_path)
    missing = REQUIRED_COLUMNS.difference(frame.columns)
    if missing:
        raise ValueError(f"Missing columns in {results_path}: {sorted(missing)}")
    if frame.empty:
        raise ValueError(f"No rows in {results_path}")

    fig, ax = plt.subplots(figsize=(5.2, 4.6), constrained_layout=True)
    norm = Normalize(
        vmin=float(frame["dataset_iteration"].min()),
        vmax=float(frame["dataset_iteration"].max()),
    )
    groups = frame.groupby(["seed", "initial_wf1", "initial_wf2"], sort=False)
    for _, trajectory in groups:
        ax.scatter(
            trajectory["wf1"],
            trajectory["wf2"],
            c=trajectory["dataset_iteration"],
            cmap="magma_r",
            norm=norm,
            s=8,
            alpha=0.48,
            linewidths=0,
            rasterized=False,
            zorder=2,
        )

    # A separate mappable keeps the colorbar independent of group ordering.
    mappable = plt.cm.ScalarMappable(norm=norm, cmap="magma_r")
    colorbar = fig.colorbar(
        mappable,
        ax=ax,
        orientation="horizontal",
        location="top",
        fraction=0.08,
        pad=0.04,
    )
    colorbar.set_label("Training iteration")

    ax.axhline(0.0, color="black", linewidth=1.0, zorder=3)
    ax.axvline(0.0, color="black", linewidth=1.0, zorder=3)
    if show_feature_regions:
        add_feature_regions(ax)
    ax.set(
        xlabel=r"Weight from input 1, $w^{2,f}_{1,1}$",
        ylabel=r"Weight from input 2, $w^{2,f}_{1,2}$",
        xlim=(-1.06, 1.06),
        ylim=(-1.06, 1.06),
    )
    ax.set_aspect("equal", adjustable="box")
    ax.grid(True, alpha=0.24)

    for output_path in (Path(pdf_path), Path(svg_path)):
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot Supplementary Fig. 3c.")
    parser.add_argument("--results-path", type=Path, default=RESULTS_PATH)
    parser.add_argument("--pdf-output", type=Path, default=PDF_PATH)
    parser.add_argument("--svg-output", type=Path, default=SVG_PATH)
    parser.add_argument("--no-feature-regions", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    plot_trajectories(
        args.results_path,
        pdf_path=args.pdf_output,
        svg_path=args.svg_output,
        show_feature_regions=not args.no_feature_regions,
    )
    print(f"Saved {args.pdf_output}")
    print(f"Saved {args.svg_output}")


if __name__ == "__main__":
    main()
