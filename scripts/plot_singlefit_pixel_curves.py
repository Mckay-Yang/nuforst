#!/usr/bin/env python3
"""Plot single-fit dense reconstruction curves for representative pixels."""

from __future__ import annotations

from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import pandas as pd


WORKSPACE = Path(__file__).resolve().parents[2]
FIG_DIR = WORKSPACE / "assets" / "figures"
CURVE_DIR = FIG_DIR / "typical_pixel_curves_singlefit"
CASE_CSV = FIG_DIR / "nufrost_typical_pixel_cases.csv"

BANDS = ["B4", "B8", "B11"]
METHODS = [
    ("nufrost", "NUFROST", "#FF4E48", "-"),
    ("zhu2015", "Zhu2015", "#3378BC", "--"),
    ("hants", "HANTS", "#329731", ":"),
]
CASE_LABELS = {
    "stable_vegetation": "Stable vegetation",
    "snow_ice_high_reflectance": "Snow/ice",
    "water_or_wet_surface": "Water/wet surface",
    "bare_dark_surface": "Bare/dark surface",
    "sparse_or_contaminated_difficult": "Sparse/difficult",
}


mpl.rcParams.update(
    {
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans", "sans-serif"],
        "svg.fonttype": "none",
        "pdf.fonttype": 42,
        "font.size": 7,
        "axes.spines.right": False,
        "axes.spines.top": False,
        "axes.linewidth": 0.8,
        "xtick.major.width": 0.7,
        "ytick.major.width": 0.7,
        "legend.frameon": False,
    }
)


def save_pub(fig: plt.Figure, stem: Path) -> None:
    fig.savefig(f"{stem}.png", dpi=300, bbox_inches="tight")
    fig.savefig(f"{stem}.svg", bbox_inches="tight")
    fig.savefig(f"{stem}.pdf", bbox_inches="tight")
    fig.savefig(f"{stem}.tiff", dpi=600, bbox_inches="tight")


def load_curve(case: str) -> pd.DataFrame:
    csv_path = CURVE_DIR / f"{case}.csv"
    df = pd.read_csv(csv_path)
    ts = pd.to_datetime(df["time_epoch"], unit="s", utc=True)
    year_start = pd.to_datetime(ts.dt.year.astype(str) + "-01-01", utc=True)
    next_year = pd.to_datetime((ts.dt.year + 1).astype(str) + "-01-01", utc=True)
    df["time_year_abs"] = ts.dt.year + (ts - year_start) / (next_year - year_start)
    return df


def draw_band_panel(ax: plt.Axes, df: pd.DataFrame, band: str, show_ylabel: bool) -> None:
    band_df = df[df["band"] == band].copy()
    curve = band_df[band_df["kind"] == "curve"].sort_values("time_year_abs")
    obs = band_df[band_df["kind"] == "observed"]
    target = band_df[band_df["kind"] == "target"]

    ax.scatter(
        obs["time_year_abs"],
        obs["observed"] / 10000.0,
        s=7,
        color="#6B6B6B",
        alpha=0.55,
        linewidths=0,
        zorder=2,
    )
    ax.scatter(
        target["time_year_abs"],
        target["observed"] / 10000.0,
        s=14,
        marker="x",
        color="#111111",
        linewidths=0.85,
        zorder=5,
    )

    for col, label, color, linestyle in METHODS:
        ax.plot(
            curve["time_year_abs"],
            curve[col] / 10000.0,
            color=color,
            lw=1.35 if col == "nufrost" else 1.15,
            linestyle=linestyle,
            label=label,
            zorder=4 if col == "nufrost" else 3,
        )

    finite_vals = pd.concat(
        [
            obs["observed"] / 10000.0,
            target["observed"] / 10000.0,
            *(curve[col] / 10000.0 for col, _, _, _ in METHODS),
        ],
        ignore_index=True,
    ).replace([float("inf"), float("-inf")], pd.NA).dropna()
    if not finite_vals.empty:
        lo = finite_vals.quantile(0.02)
        hi = finite_vals.quantile(0.98)
        pad = max((hi - lo) * 0.18, 0.025)
        ax.set_ylim(max(0.0, lo - pad), min(1.05, hi + pad))

    ax.grid(axis="y", color="#D9D9D9", lw=0.45, alpha=0.65)
    ax.tick_params(labelsize=6.2, length=2.4)
    if show_ylabel:
        ax.set_ylabel("Reflectance", labelpad=3)
    else:
        ax.set_ylabel("")


def plot_case(case: str) -> None:
    df = load_curve(case)

    fig, axes = plt.subplots(1, len(BANDS), figsize=(6.7, 1.85), sharex=True)
    if len(BANDS) == 1:
        axes = [axes]

    for ax, band in zip(axes, BANDS):
        draw_band_panel(ax, df, band, ax is axes[0])
        ax.set_title(band, pad=4, fontsize=8)
        ax.set_xlabel("Year", labelpad=2)

    handles, labels = axes[-1].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="upper center",
        ncol=3,
        bbox_to_anchor=(0.5, 1.08),
        handlelength=2.4,
        columnspacing=1.4,
    )
    fig.subplots_adjust(left=0.07, right=0.995, top=0.78, bottom=0.24, wspace=0.25)

    stem = FIG_DIR / f"nufrost_typical_pixel_curve_{case}_singlefit_compact"
    save_pub(fig, stem)
    plt.close(fig)


def plot_horizontal_composite(cases: list[str]) -> None:
    n_rows = len(BANDS)
    n_cols = len(cases)
    fig, axes = plt.subplots(
        n_rows,
        n_cols,
        figsize=(13.2, 4.7),
        sharex=False,
        constrained_layout=False,
    )

    for col, case in enumerate(cases):
        df = load_curve(case)
        for row, band in enumerate(BANDS):
            ax = axes[row, col]
            draw_band_panel(ax, df, band, show_ylabel=(col == 0))
            if row == 0:
                ax.set_title(CASE_LABELS.get(case, case), pad=5, fontsize=8)
            if row == n_rows - 1:
                ax.set_xlabel("Year", labelpad=2)
            else:
                ax.set_xlabel("")
                ax.tick_params(labelbottom=False)

    handles, labels = axes[0, -1].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="upper center",
        ncol=3,
        bbox_to_anchor=(0.5, 1.015),
        handlelength=2.6,
        columnspacing=1.5,
        fontsize=8,
    )
    fig.subplots_adjust(left=0.045, right=0.995, top=0.89, bottom=0.12, wspace=0.28, hspace=0.25)
    stem = FIG_DIR / "nufrost_typical_pixel_curves_singlefit_horizontal"
    save_pub(fig, stem)
    plt.close(fig)


def plot_case_rows_band_columns(cases: list[str]) -> None:
    n_rows = len(cases)
    n_cols = len(BANDS)
    fig, axes = plt.subplots(
        n_rows,
        n_cols,
        figsize=(9.6, 8.4),
        sharex=False,
        constrained_layout=False,
    )

    for row, case in enumerate(cases):
        df = load_curve(case)
        for col, band in enumerate(BANDS):
            ax = axes[row, col]
            draw_band_panel(ax, df, band, show_ylabel=(col == 0))
            if row == 0:
                ax.set_title(band, pad=5, fontsize=8)
            if col == 0:
                ax.text(
                    -0.24,
                    0.5,
                    CASE_LABELS.get(case, case),
                    transform=ax.transAxes,
                    rotation=90,
                    ha="center",
                    va="center",
                    fontsize=8,
                )
            if row == n_rows - 1:
                ax.set_xlabel("Year", labelpad=2)
            else:
                ax.set_xlabel("")
                ax.tick_params(labelbottom=False)

    handles, labels = axes[0, -1].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="upper center",
        ncol=3,
        bbox_to_anchor=(0.5, 1.01),
        handlelength=2.6,
        columnspacing=1.5,
        fontsize=8,
    )
    fig.subplots_adjust(left=0.11, right=0.995, top=0.94, bottom=0.075, wspace=0.25, hspace=0.32)
    stem = FIG_DIR / "nufrost_typical_pixel_curves_singlefit_wide"
    save_pub(fig, stem)
    plt.close(fig)


def main() -> None:
    cases = pd.read_csv(CASE_CSV)["case"].tolist()
    for case in cases:
        plot_case(case)
    plot_horizontal_composite(cases)
    plot_case_rows_band_columns(cases)


if __name__ == "__main__":
    main()
