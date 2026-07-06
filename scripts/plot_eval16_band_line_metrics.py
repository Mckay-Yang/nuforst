#!/usr/bin/env python3
"""Plot per-band full-scene errors as bars with method-wise trend lines."""

from __future__ import annotations

from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


BANDS = ["B2", "B3", "B4", "B8", "B11", "B12"]
METHOD_ORDER = ["nufrost", "zhu2015", "hants"]
METHOD_LABELS = {
    "nufrost": "NUFROST",
    "zhu2015": "Zhu2015",
    "hants": "HANTS",
}

METHOD_COLORS = {
    "nufrost": "#3378BC",
    "zhu2015": "#D8A642",
    "hants": "#D45F5F",
}

METHOD_MARKERS = {
    "nufrost": "o",
    "zhu2015": "s",
    "hants": "^",
}


def setup_style() -> None:
    mpl.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans", "sans-serif"],
            "svg.fonttype": "none",
            "pdf.fonttype": 42,
            "font.size": 5.9,
            "axes.labelsize": 6.3,
            "xtick.labelsize": 5.9,
            "ytick.labelsize": 5.9,
            "legend.fontsize": 5.8,
            "axes.spines.right": False,
            "axes.spines.top": False,
            "axes.linewidth": 0.72,
            "xtick.major.width": 0.65,
            "ytick.major.width": 0.65,
            "legend.frameon": False,
        }
    )


def save_figure(fig: plt.Figure, out_base: Path) -> None:
    fig.savefig(out_base.with_suffix(".png"), dpi=600, bbox_inches="tight")


def plot_metric(source: pd.DataFrame, metric: str, ylabel: str, out_base: Path) -> None:
    fig, ax = plt.subplots(figsize=(3.9, 1.95), constrained_layout=True)
    x = np.arange(len(BANDS), dtype=float)
    bar_width = 0.22
    offsets = {
        "nufrost": -bar_width,
        "zhu2015": 0.0,
        "hants": bar_width,
    }

    for method in METHOD_ORDER:
        rows = (
            source[source["method"] == method]
            .set_index("band")
            .reindex(BANDS)
            .reset_index()
        )
        values = rows[metric].to_numpy(dtype=float)
        xpos = x + offsets[method]
        ax.bar(
            xpos,
            values,
            width=bar_width * 0.88,
            color=METHOD_COLORS[method],
            alpha=0.78,
            linewidth=0,
            label=METHOD_LABELS[method],
            zorder=2,
        )
        ax.plot(
            xpos,
            values,
            color=METHOD_COLORS[method],
            marker=METHOD_MARKERS[method],
            markersize=1.9,
            markeredgewidth=0.0,
            lw=0.62,
            alpha=0.30,
            zorder=3,
        )

    ax.set_xticks(x, BANDS)
    ax.set_xlim(-0.55, len(BANDS) - 0.45)
    ax.set_ylim(bottom=0)
    ax.set_xlabel("Sentinel-2 band")
    ax.set_ylabel(ylabel)
    ax.tick_params(axis="x", length=0, pad=3)
    ax.tick_params(axis="y", length=2.5)
    ax.legend(
        loc="lower center",
        bbox_to_anchor=(0.5, 1.02),
        ncol=3,
        handlelength=1.1,
        handletextpad=0.35,
        columnspacing=0.9,
        borderaxespad=0.0,
    )
    save_figure(fig, out_base)
    plt.close(fig)


def main() -> None:
    workspace = Path(__file__).resolve().parents[2]
    figure_dir = workspace / "assets" / "figures"
    source_path = figure_dir / "nufrost_eval16_band_error_source.csv"
    source = pd.read_csv(source_path)

    setup_style()
    plot_metric(
        source,
        "rmse_reflectance",
        "RMSE (reflectance)",
        figure_dir / "nufrost_eval16_band_rmse",
    )
    plot_metric(
        source,
        "mae_reflectance",
        "MAE (reflectance)",
        figure_dir / "nufrost_eval16_band_mae",
    )


if __name__ == "__main__":
    main()
