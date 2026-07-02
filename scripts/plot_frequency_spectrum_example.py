#!/usr/bin/env python3
"""Plot the NUFROST vector-frequency spectrum example.

The source pixel is the stable-vegetation example already used by the typical
pixel curve figures. Each band is robust-standardized independently before the
non-uniform Fourier power is computed. The joint spectrum is the sum of the
six band powers.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


BANDS = ["B2", "B3", "B4", "B8", "B11", "B12"]

# Low-saturation colors: visible, NIR, and SWIR remain distinguishable without
# competing with the joint power curve.
BAND_COLORS = {
    "B2": "#6F8FB8",
    "B3": "#78A77B",
    "B4": "#C27D78",
    "B8": "#8D8AB9",
    "B11": "#B9965E",
    "B12": "#9D9586",
}


def robust_z(values: np.ndarray) -> np.ndarray:
    center = np.nanmedian(values)
    mad = np.nanmedian(np.abs(values - center))
    scale = 1.4826 * mad
    if not np.isfinite(scale) or scale < 1e-6:
        scale = np.nanstd(values)
    if not np.isfinite(scale) or scale < 1e-6:
        scale = 1.0
    return (values - center) / scale


def nonuniform_power(times: np.ndarray, values: np.ndarray, freqs: np.ndarray) -> np.ndarray:
    phase = np.exp(-2j * np.pi * freqs[:, None] * times[None, :])
    coeff = phase @ values
    return (np.abs(coeff) ** 2) / max(len(times), 1)


def save_figure(fig: plt.Figure, out_base: Path) -> None:
    fig.savefig(out_base.with_suffix(".png"), dpi=600, bbox_inches="tight")
    fig.savefig(out_base.with_suffix(".svg"), bbox_inches="tight")
    fig.savefig(out_base.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(out_base.with_suffix(".tiff"), dpi=600, bbox_inches="tight")


def main() -> None:
    workspace = Path(__file__).resolve().parents[2]
    figure_dir = workspace / "assets" / "figures"
    source_path = figure_dir / "nufrost_typical_pixel_curves_source.csv"
    spectrum_source_path = figure_dir / "nufrost_frequency_spectrum_example_source.csv"
    out_base = figure_dir / "nufrost_frequency_spectrum_example"

    data = pd.read_csv(source_path)
    stable = data[data["case"] == "stable_vegetation"].copy()
    stable = stable[stable["band"].isin(BANDS)]
    stable = stable[np.isfinite(stable["observed_reflectance"])]

    # Multiple acquisitions can share the same date after the source export.
    # Averaging keeps one spectral vector per unique time step for the example.
    pivot = (
        stable.groupby(["time_year", "band"], as_index=False)["observed_reflectance"]
        .mean()
        .pivot(index="time_year", columns="band", values="observed_reflectance")
        .sort_index()
    )
    pivot = pivot.dropna(subset=BANDS, how="any")
    times = pivot.index.to_numpy(dtype=float)

    freqs = np.linspace(0.05, 6.0, 480)
    band_power: dict[str, np.ndarray] = {}
    for band in BANDS:
        z = robust_z(pivot[band].to_numpy(dtype=float))
        band_power[band] = nonuniform_power(times, z, freqs)

    joint_power = np.sum(np.vstack([band_power[band] for band in BANDS]), axis=0)

    # Plot normalized power so band contributions and joint spectrum are visible
    # in the same panel. The raw source values are still written to CSV.
    joint_norm = joint_power / np.nanmax(joint_power)
    band_norm = {
        band: values / np.nanmax(joint_power)
        for band, values in band_power.items()
    }

    spectrum = pd.DataFrame({"frequency_cycles_per_year": freqs, "joint_power": joint_power})
    for band in BANDS:
        spectrum[f"{band}_power"] = band_power[band]
        spectrum[f"{band}_power_normalized_to_joint_max"] = band_norm[band]
    spectrum["joint_power_normalized"] = joint_norm
    spectrum.to_csv(spectrum_source_path, index=False)

    mpl.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans", "sans-serif"],
            "svg.fonttype": "none",
            "pdf.fonttype": 42,
            "font.size": 6.2,
            "axes.spines.right": False,
            "axes.spines.top": False,
            "axes.linewidth": 0.72,
            "axes.labelsize": 6.7,
            "xtick.labelsize": 6.1,
            "ytick.labelsize": 6.1,
            "legend.fontsize": 5.9,
            "legend.frameon": False,
            "xtick.major.width": 0.65,
            "ytick.major.width": 0.65,
        }
    )

    fig, ax = plt.subplots(figsize=(5.15, 2.28), constrained_layout=True)

    ax.fill_between(
        freqs,
        0,
        joint_norm,
        color="#252A32",
        alpha=0.035,
        lw=0,
        zorder=1,
    )

    for band in BANDS:
        ax.plot(
            freqs,
            band_norm[band],
            color=BAND_COLORS[band],
            lw=0.55,
            alpha=0.36,
            label=band,
            zorder=3,
        )

    ax.plot(
        freqs,
        joint_norm,
        color="#C44E52",
        lw=1.38,
        alpha=0.96,
        label="Joint power",
        zorder=6,
    )

    reference_lines = [
        (1.0, "Annual"),
        (2.0, "Semiannual"),
    ]
    for x, label in reference_lines:
        ax.axvline(x, color="#A8A8A8", lw=0.62, ls=(0, (3, 2)), alpha=0.62, zorder=2)
        ax.text(
            x + 0.045,
            1.025,
            label,
            ha="left",
            va="center",
            color="#7A7A7A",
            fontsize=5.6,
        )

    ax.set_xlim(0, 6)
    ax.set_ylim(0, 1.10)
    ax.set_xlabel("Frequency (cycles yr$^{-1}$)")
    ax.set_ylabel("Normalized power")
    ax.grid(axis="y", color="#E2E2E2", lw=0.42, alpha=0.88)
    ax.grid(axis="x", color="#EFEFEF", lw=0.32, alpha=0.45)
    ax.legend(
        loc="lower center",
        bbox_to_anchor=(0.5, 1.03),
        ncol=7,
        handlelength=1.10,
        columnspacing=0.85,
        handletextpad=0.28,
        borderaxespad=0.0,
    )

    save_figure(fig, out_base)
    plt.close(fig)


if __name__ == "__main__":
    main()
