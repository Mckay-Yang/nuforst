#!/usr/bin/env python3
"""Plot full-scene per-band error heatmaps by region.

The script reads the CSV produced by `scripts/evaluate_reconstructions.py`
and plots method x band x region heatmaps for DIFF RMSE and MAE.

By default, RMSE/MAE are divided by the Sentinel-2 reflectance scale factor
10000 so the heatmaps are in normalized reflectance units.
"""

from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path

import matplotlib.colors as mcolors
import matplotlib.pyplot as plt


METHODS = ["nufrost", "hants", "zhu2015"]
BANDS = ["B2", "B3", "B4", "B8", "B11", "B12"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot full-scene band RMSE/MAE heatmaps by region."
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("data/assets/records/reconstruction_evaluation/evaluation_summary.csv"),
        help="Evaluation summary CSV path.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/assets/figures"),
        help="Directory for output PNG and CSV files.",
    )
    parser.add_argument(
        "--reflectance-scale",
        type=float,
        default=10000.0,
        help="Divide RMSE/MAE by this scale factor; Sentinel-2 SR defaults to 10000.",
    )
    parser.add_argument(
        "--metric",
        choices=["rmse", "mae", "both"],
        default="both",
        help="Metric to plot.",
    )
    parser.add_argument(
        "--dpi",
        type=int,
        default=220,
        help="Output PNG DPI.",
    )
    return parser.parse_args()


def load_diff_rows(path: Path) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    with path.open(newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if not (
                row["section"] == "continuous"
                and row["product"] == "DIFF"
                and row["class_kind"] == "all"
            ):
                continue
            method = row["method"]
            band = row["band_or_index"]
            if method not in METHODS or band not in BANDS:
                continue
            rows.append(
                {
                    "scene": row["scene"],
                    "method": method,
                    "band": band,
                    "n": row["n"],
                    "rmse": parse_float(row["rmse"]),
                    "mae": parse_float(row["mae"]),
                    "bias": parse_float(row["bias"]),
                }
            )
    return rows


def attach_scaled_metrics(
    rows: list[dict[str, object]], reflectance_scale: float
) -> None:
    if not math.isfinite(reflectance_scale) or reflectance_scale <= 0:
        raise ValueError(f"reflectance scale must be positive, got {reflectance_scale}")
    for row in rows:
        row["normalizer"] = reflectance_scale
        row["normalizer_kind"] = "reflectance_scale"
        row["rmse_norm"] = float(row["rmse"]) / reflectance_scale
        row["mae_norm"] = float(row["mae"]) / reflectance_scale


def parse_float(value: str) -> float:
    return float(value) if value else math.nan


def write_metric_csv(rows: list[dict[str, object]], output_csv: Path) -> None:
    scenes = sorted({str(r["scene"]) for r in rows})
    lookup = {
        (str(r["method"]), str(r["band"]), str(r["scene"])): r
        for r in rows
    }
    with output_csv.open("w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "scene",
                "method",
                "band",
                "n",
                "rmse",
                "mae",
                "bias",
                "normalizer_kind",
                "normalizer",
                "rmse_norm",
                "mae_norm",
            ],
        )
        writer.writeheader()
        for scene in scenes:
            for method in METHODS:
                for band in BANDS:
                    row = lookup.get((method, band, scene))
                    if row:
                        out = dict(row)
                        for key in ("rmse_norm", "mae_norm"):
                            value = out.get(key)
                            if isinstance(value, float) and math.isfinite(value):
                                out[key] = f"{value:.4f}"
                        writer.writerow(out)


def plot_metric(
    rows: list[dict[str, object]],
    metric: str,
    output_dir: Path,
    dpi: int,
    normalized: bool,
) -> None:
    scenes = sorted({str(r["scene"]) for r in rows})
    lookup = {
        (str(r["method"]), str(r["band"]), str(r["scene"])): r
        for r in rows
    }
    value_key = f"{metric}_norm" if normalized else metric
    values = [
        float(r[value_key])
        for r in rows
        if isinstance(r[value_key], float) and math.isfinite(float(r[value_key]))
    ]
    if not values:
        raise ValueError(f"No finite {value_key} values found")

    values_sorted = sorted(values)
    p95 = values_sorted[int(0.95 * (len(values_sorted) - 1))]
    cap = max(p95, 0.001)

    fig, axes = plt.subplots(len(METHODS), 1, figsize=(18, 9.5), constrained_layout=True)
    if len(METHODS) == 1:
        axes = [axes]

    cmap_name = "YlOrRd" if metric == "rmse" else "YlGnBu"
    cmap = plt.get_cmap(cmap_name).copy()
    cmap.set_bad("#E5E7EB")
    norm = mcolors.Normalize(vmin=0, vmax=cap)

    for ax, method in zip(axes, METHODS):
        matrix: list[list[float]] = []
        for band in BANDS:
            row_vals = []
            for scene in scenes:
                rec = lookup.get((method, band, scene))
                row_vals.append(float(rec[value_key]) if rec else math.nan)
            matrix.append(row_vals)

        im = ax.imshow(matrix, aspect="auto", cmap=cmap, norm=norm)
        ax.set_title(method.upper(), loc="left", fontsize=13, fontweight="bold")
        ax.set_yticks(range(len(BANDS)))
        ax.set_yticklabels(BANDS, fontsize=10)
        ax.set_xticks(range(len(scenes)))
        ax.set_xticklabels(scenes, rotation=45, ha="right", fontsize=8)
        ax.tick_params(axis="both", length=0)

        for y, _band in enumerate(BANDS):
            for x, _scene in enumerate(scenes):
                val = matrix[y][x]
                if not math.isfinite(val):
                    continue
                color = "white" if val > cap * 0.62 else "#111827"
                label = f"{val:.4f}" if normalized else f"{val:.0f}"
                ax.text(x, y, label, ha="center", va="center", fontsize=7, color=color)

        for spine in ax.spines.values():
            spine.set_visible(False)

    metric_upper = metric.upper()
    title_prefix = "Normalized " if normalized else ""
    unit_label = (
        f"{metric_upper} / reflectance scale"
        if normalized
        else metric_upper
    )
    fig.suptitle(
        f"Full-scene {title_prefix}{metric_upper} by Region and Sentinel-2 Band",
        fontsize=17,
        fontweight="bold",
    )
    fig.supxlabel("Region (lon_lat)", fontsize=11)
    fig.supylabel("Band", fontsize=11)
    cbar = fig.colorbar(im, ax=axes, shrink=0.9, pad=0.01)
    cbar.set_label(
        f"{unit_label} (color capped at p95={cap:.2f}; labels show actual values)",
        fontsize=10,
    )

    output_png = output_dir / f"full_scene_band_{metric}_by_region.png"
    fig.savefig(output_png, dpi=dpi)
    plt.close(fig)
    print(output_png)


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    rows = load_diff_rows(args.input)
    if not rows:
        raise SystemExit(f"No DIFF band rows found in {args.input}")
    normalized = True
    attach_scaled_metrics(rows, args.reflectance_scale)

    metrics = ["rmse", "mae"] if args.metric == "both" else [args.metric]
    for metric in metrics:
        output_csv = args.output_dir / f"full_scene_band_{metric}_by_region.csv"
        write_metric_csv(rows, output_csv)
        print(output_csv)
        plot_metric(rows, metric, args.output_dir, args.dpi, normalized)


if __name__ == "__main__":
    main()
