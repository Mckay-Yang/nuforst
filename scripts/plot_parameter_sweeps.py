#!/usr/bin/env python3
"""Generate SVG summaries for NUFROST parameter sweep CSV files.

This script intentionally uses only the Python standard library so the figures
can be regenerated in minimal environments.
"""

from __future__ import annotations

import argparse
import csv
import html
import math
from collections import defaultdict
from pathlib import Path


PALETTE = [
    "#3378BC",
    "#FF4E48",
    "#329731",
    "#8A5FBF",
    "#E08A1E",
    "#1B9E9E",
    "#6B7280",
]

FONT = "PingFang SC, Microsoft YaHei, Noto Sans CJK SC, Arial, sans-serif"


def to_float(value: str | None) -> float:
    if value is None or value == "":
        return math.nan
    try:
        return float(value)
    except ValueError:
        return math.nan


def fmt(value: float, ndigits: int = 3) -> str:
    if not math.isfinite(value):
        return "-"
    return f"{value:.{ndigits}f}"


def esc(text: object) -> str:
    return html.escape(str(text), quote=True)


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    return [
        row
        for row in rows
        if row.get("status") == "ok" and math.isfinite(to_float(row.get("rmse")))
    ]


def svg_header(width: int, height: int) -> list[str]:
    return [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        "<defs>",
        "<style>",
        f"text {{ font-family: {FONT}; fill: #111827; }}",
        ".title { font-size: 24px; font-weight: 700; }",
        ".subtitle { font-size: 13px; fill: #4B5563; }",
        ".axis { stroke: #9CA3AF; stroke-width: 1; }",
        ".grid { stroke: #E5E7EB; stroke-width: 1; }",
        ".tick { font-size: 11px; fill: #6B7280; }",
        ".label { font-size: 12px; fill: #374151; }",
        ".small { font-size: 10px; fill: #6B7280; }",
        "</style>",
        "</defs>",
        '<rect x="0" y="0" width="100%" height="100%" fill="#FFFFFF"/>',
    ]


def write_svg(path: Path, content: list[str]) -> None:
    path.write_text("\n".join(content) + "\n", encoding="utf-8")


def scale(value: float, d0: float, d1: float, r0: float, r1: float) -> float:
    if d1 == d0:
        return (r0 + r1) / 2.0
    return r0 + (value - d0) * (r1 - r0) / (d1 - d0)


def nice_ticks(vmin: float, vmax: float, count: int = 5) -> list[float]:
    if not math.isfinite(vmin) or not math.isfinite(vmax) or vmin == vmax:
        return [vmin]
    span = vmax - vmin
    raw = span / max(count - 1, 1)
    mag = 10 ** math.floor(math.log10(raw))
    norm = raw / mag
    if norm <= 1:
        step = mag
    elif norm <= 2:
        step = 2 * mag
    elif norm <= 5:
        step = 5 * mag
    else:
        step = 10 * mag
    start = math.floor(vmin / step) * step
    ticks = []
    value = start
    while value <= vmax + step * 0.5:
        if value >= vmin - step * 0.5:
            ticks.append(value)
        value += step
    return ticks[: count + 2]


def phase_label(phase_file: str) -> str:
    label = phase_file.replace("_20k_results.csv", "").replace("_results.csv", "")
    label = label.replace("phase", "p")
    return label


def draw_scatter(rows: list[dict[str, str]], output: Path) -> None:
    width, height = 1200, 760
    left, right, top, bottom = 88, 280, 92, 90
    plot_w = width - left - right
    plot_h = height - top - bottom

    rmses = [to_float(r["rmse"]) for r in rows]
    maes = [to_float(r["mae"]) for r in rows]
    xmin, xmax = min(rmses), max(rmses)
    ymin, ymax = min(maes), max(maes)
    pad_x = (xmax - xmin) * 0.04
    pad_y = (ymax - ymin) * 0.06
    xmin -= pad_x
    xmax += pad_x
    ymin -= pad_y
    ymax += pad_y

    phases = sorted({r["phase_file"] for r in rows})
    color = {p: PALETTE[i % len(PALETTE)] for i, p in enumerate(phases)}

    parts = svg_header(width, height)
    parts.append('<text x="48" y="46" class="title">NUFROST参数扫描：总体RMSE/MAE分布</text>')
    parts.append(
        f'<text x="48" y="72" class="subtitle">样本数：{len(rows)}；每个点为一组参数配置，颜色表示实验批次。</text>'
    )

    for tick in nice_ticks(xmin, xmax):
        x = scale(tick, xmin, xmax, left, left + plot_w)
        parts.append(f'<line x1="{x:.1f}" y1="{top}" x2="{x:.1f}" y2="{top + plot_h}" class="grid"/>')
        parts.append(f'<text x="{x:.1f}" y="{top + plot_h + 24}" text-anchor="middle" class="tick">{fmt(tick, 0)}</text>')
    for tick in nice_ticks(ymin, ymax):
        y = scale(tick, ymin, ymax, top + plot_h, top)
        parts.append(f'<line x1="{left}" y1="{y:.1f}" x2="{left + plot_w}" y2="{y:.1f}" class="grid"/>')
        parts.append(f'<text x="{left - 14}" y="{y + 4:.1f}" text-anchor="end" class="tick">{fmt(tick, 0)}</text>')

    parts.append(f'<line x1="{left}" y1="{top + plot_h}" x2="{left + plot_w}" y2="{top + plot_h}" class="axis"/>')
    parts.append(f'<line x1="{left}" y1="{top}" x2="{left}" y2="{top + plot_h}" class="axis"/>')
    parts.append(f'<text x="{left + plot_w / 2:.1f}" y="{height - 34}" text-anchor="middle" class="label">RMSE</text>')
    parts.append(f'<text transform="translate(28 {top + plot_h / 2:.1f}) rotate(-90)" text-anchor="middle" class="label">MAE</text>')

    best = min(rows, key=lambda r: to_float(r["rmse"]))
    for r in rows:
        x = scale(to_float(r["rmse"]), xmin, xmax, left, left + plot_w)
        y = scale(to_float(r["mae"]), ymin, ymax, top + plot_h, top)
        radius = 4.8 if r is best else 3.8
        stroke = "#111827" if r is best else "none"
        parts.append(
            f'<circle cx="{x:.2f}" cy="{y:.2f}" r="{radius}" fill="{color[r["phase_file"]]}" fill-opacity="0.72" stroke="{stroke}" stroke-width="1.4">'
            f'<title>{esc(r["name"])}\\nRMSE={fmt(to_float(r["rmse"]))}, MAE={fmt(to_float(r["mae"]))}</title></circle>'
        )

    lx = width - right + 28
    parts.append(f'<text x="{lx}" y="{top}" class="label" font-weight="700">实验批次</text>')
    for i, p in enumerate(phases[:16]):
        y = top + 24 + i * 26
        parts.append(f'<rect x="{lx}" y="{y - 10}" width="13" height="13" fill="{color[p]}"/>')
        parts.append(f'<text x="{lx + 22}" y="{y + 1}" class="small">{esc(phase_label(p))}</text>')
    if len(phases) > 16:
        parts.append(f'<text x="{lx}" y="{top + 24 + 16 * 26}" class="small">另有{len(phases) - 16}个批次</text>')

    bx = scale(to_float(best["rmse"]), xmin, xmax, left, left + plot_w)
    by = scale(to_float(best["mae"]), ymin, ymax, top + plot_h, top)
    parts.append(f'<text x="{bx + 12:.1f}" y="{by - 12:.1f}" class="label" font-weight="700">best RMSE</text>')
    parts.append("</svg>")
    write_svg(output, parts)


def draw_top_bars(rows: list[dict[str, str]], output: Path) -> None:
    top_rows = sorted(rows, key=lambda r: to_float(r["rmse"]))[:12]
    width, height = 1250, 780
    left, top = 390, 108
    bar_w, row_h = 660, 42
    rmse_vals = [to_float(r["rmse"]) for r in top_rows]
    mae_vals = [to_float(r["mae"]) for r in top_rows]
    xmin = min(min(rmse_vals), min(mae_vals)) * 0.96
    xmax = max(max(rmse_vals), max(mae_vals)) * 1.02

    parts = svg_header(width, height)
    parts.append('<text x="48" y="46" class="title">最优参数配置Top12</text>')
    parts.append('<text x="48" y="72" class="subtitle">按全波段RMSE升序排列；红色为RMSE，蓝色为MAE。</text>')

    for tick in nice_ticks(xmin, xmax, 7):
        x = scale(tick, xmin, xmax, left, left + bar_w)
        parts.append(f'<line x1="{x:.1f}" y1="{top - 18}" x2="{x:.1f}" y2="{top + row_h * len(top_rows)}" class="grid"/>')
        parts.append(f'<text x="{x:.1f}" y="{top - 28}" text-anchor="middle" class="tick">{fmt(tick, 0)}</text>')

    for i, r in enumerate(top_rows):
        y = top + i * row_h
        name = r["name"]
        short = name if len(name) <= 44 else name[:41] + "..."
        parts.append(f'<text x="{left - 14}" y="{y + 16}" text-anchor="end" class="small">{esc(short)}</text>')
        x0 = scale(xmin, xmin, xmax, left, left + bar_w)
        x_rmse = scale(to_float(r["rmse"]), xmin, xmax, left, left + bar_w)
        x_mae = scale(to_float(r["mae"]), xmin, xmax, left, left + bar_w)
        parts.append(f'<rect x="{x0:.1f}" y="{y + 4}" width="{x_rmse - x0:.1f}" height="14" fill="#FF4E48" rx="2"/>')
        parts.append(f'<rect x="{x0:.1f}" y="{y + 22}" width="{x_mae - x0:.1f}" height="14" fill="#3378BC" rx="2"/>')
        parts.append(f'<text x="{x_rmse + 8:.1f}" y="{y + 16}" class="small">{fmt(to_float(r["rmse"]), 2)}</text>')
        parts.append(f'<text x="{x_mae + 8:.1f}" y="{y + 34}" class="small">{fmt(to_float(r["mae"]), 2)}</text>')

    legend_y = top + row_h * len(top_rows) + 34
    parts.append(f'<rect x="{left}" y="{legend_y}" width="16" height="12" fill="#FF4E48"/>')
    parts.append(f'<text x="{left + 24}" y="{legend_y + 11}" class="label">RMSE</text>')
    parts.append(f'<rect x="{left + 96}" y="{legend_y}" width="16" height="12" fill="#3378BC"/>')
    parts.append(f'<text x="{left + 120}" y="{legend_y + 11}" class="label">MAE</text>')
    parts.append("</svg>")
    write_svg(output, parts)


def draw_heatmap(rows: list[dict[str, str]], output: Path) -> None:
    cells: dict[tuple[float, float], float] = {}
    for r in rows:
        fw = to_float(r.get("freq_weight"))
        sh = to_float(r.get("multiband_shrinkage"))
        rmse = to_float(r.get("rmse"))
        if not all(math.isfinite(x) for x in (fw, sh, rmse)):
            continue
        key = (fw, sh)
        cells[key] = min(cells.get(key, math.inf), rmse)

    xs = sorted({k[1] for k in cells})
    ys = sorted({k[0] for k in cells})
    if not xs or not ys:
        return

    width, height = 1180, 820
    left, top = 92, 108
    cell_w = min(76, max(24, int((width - 250) / len(xs))))
    cell_h = min(38, max(18, int((height - 220) / len(ys))))
    plot_w, plot_h = cell_w * len(xs), cell_h * len(ys)
    vals = list(cells.values())
    vmin, vmax = min(vals), max(vals)

    def color_for(v: float) -> str:
        t = (v - vmin) / (vmax - vmin) if vmax > vmin else 0.0
        # Blue to pale to red.
        if t < 0.5:
            q = t / 0.5
            r = int(51 + (242 - 51) * q)
            g = int(120 + (242 - 120) * q)
            b = int(188 + (242 - 188) * q)
        else:
            q = (t - 0.5) / 0.5
            r = int(242 + (255 - 242) * q)
            g = int(242 + (78 - 242) * q)
            b = int(242 + (72 - 242) * q)
        return f"#{r:02X}{g:02X}{b:02X}"

    parts = svg_header(width, height)
    parts.append('<text x="48" y="46" class="title">参数敏感性：freq_weight × multiband_shrinkage</text>')
    parts.append('<text x="48" y="72" class="subtitle">每个格子取该参数组合下最小全波段RMSE；蓝色更优，红色更差。</text>')

    for j, fw in enumerate(ys):
        y = top + j * cell_h
        parts.append(f'<text x="{left - 12}" y="{y + cell_h * 0.62:.1f}" text-anchor="end" class="tick">{fmt(fw, 3).rstrip("0").rstrip(".")}</text>')
    for i, sh in enumerate(xs):
        x = left + i * cell_w
        parts.append(f'<text transform="translate({x + cell_w * 0.55:.1f} {top + plot_h + 16}) rotate(45)" class="tick">{fmt(sh, 3).rstrip("0").rstrip(".")}</text>')

    for j, fw in enumerate(ys):
        for i, sh in enumerate(xs):
            x = left + i * cell_w
            y = top + j * cell_h
            v = cells.get((fw, sh))
            fill = "#F3F4F6" if v is None else color_for(v)
            parts.append(f'<rect x="{x}" y="{y}" width="{cell_w}" height="{cell_h}" fill="{fill}" stroke="#FFFFFF" stroke-width="1"/>')
            if v is not None and cell_w >= 46 and cell_h >= 28:
                parts.append(f'<text x="{x + cell_w / 2:.1f}" y="{y + cell_h / 2 + 4:.1f}" text-anchor="middle" class="small">{fmt(v, 1)}</text>')

    parts.append(f'<text x="{left + plot_w / 2:.1f}" y="{height - 40}" text-anchor="middle" class="label">multiband_shrinkage</text>')
    parts.append(f'<text transform="translate(28 {top + plot_h / 2:.1f}) rotate(-90)" text-anchor="middle" class="label">freq_weight</text>')

    lx = left + plot_w + 56
    ly = top
    parts.append(f'<text x="{lx}" y="{ly - 12}" class="label">RMSE</text>')
    for i in range(0, 120):
        t = i / 119
        v = vmin + t * (vmax - vmin)
        parts.append(f'<rect x="{lx}" y="{ly + i * 3}" width="22" height="3" fill="{color_for(v)}"/>')
    parts.append(f'<text x="{lx + 32}" y="{ly + 4}" class="tick">{fmt(vmin, 1)}</text>')
    parts.append(f'<text x="{lx + 32}" y="{ly + 120 * 3}" class="tick">{fmt(vmax, 1)}</text>')
    parts.append("</svg>")
    write_svg(output, parts)


def draw_index_profile(rows: list[dict[str, str]], output: Path) -> None:
    best_rows = sorted(rows, key=lambda r: to_float(r["rmse"]))[:6]
    metrics = ["ndvi", "ndwi", "ndmi", "ndsi", "nbr", "evi"]
    width, height = 1250, 760
    left, top = 92, 108
    plot_w, plot_h = 930, 500
    all_vals = []
    for r in best_rows:
        for m in metrics:
            all_vals.append(to_float(r.get(f"{m}_rmse")))
    ymin, ymax = 0.0, max(v for v in all_vals if math.isfinite(v)) * 1.12

    parts = svg_header(width, height)
    parts.append('<text x="48" y="46" class="title">最优配置的指数RMSE对比</text>')
    parts.append('<text x="48" y="72" class="subtitle">选取全波段RMSE最优的6组配置；指数包括NDVI、NDWI、NDMI、NDSI、NBR和EVI。</text>')

    for tick in nice_ticks(ymin, ymax, 6):
        y = scale(tick, ymin, ymax, top + plot_h, top)
        parts.append(f'<line x1="{left}" y1="{y:.1f}" x2="{left + plot_w}" y2="{y:.1f}" class="grid"/>')
        parts.append(f'<text x="{left - 14}" y="{y + 4:.1f}" text-anchor="end" class="tick">{fmt(tick, 2)}</text>')

    group_w = plot_w / len(metrics)
    bar_w = min(18, group_w / (len(best_rows) + 2))
    for mi, metric in enumerate(metrics):
        gx = left + mi * group_w
        parts.append(f'<text x="{gx + group_w / 2:.1f}" y="{top + plot_h + 28}" text-anchor="middle" class="label">{metric.upper()}</text>')
        for ri, row in enumerate(best_rows):
            v = to_float(row.get(f"{metric}_rmse"))
            x = gx + group_w * 0.18 + ri * (bar_w + 4)
            y = scale(v, ymin, ymax, top + plot_h, top)
            h = top + plot_h - y
            parts.append(
                f'<rect x="{x:.1f}" y="{y:.1f}" width="{bar_w:.1f}" height="{h:.1f}" fill="{PALETTE[ri % len(PALETTE)]}" rx="2">'
                f'<title>{esc(row["name"])}\\n{metric.upper()} RMSE={fmt(v, 4)}</title></rect>'
            )

    parts.append(f'<line x1="{left}" y1="{top + plot_h}" x2="{left + plot_w}" y2="{top + plot_h}" class="axis"/>')
    parts.append(f'<line x1="{left}" y1="{top}" x2="{left}" y2="{top + plot_h}" class="axis"/>')
    parts.append(f'<text transform="translate(30 {top + plot_h / 2:.1f}) rotate(-90)" text-anchor="middle" class="label">指数RMSE</text>')

    lx = left + plot_w + 38
    parts.append(f'<text x="{lx}" y="{top}" class="label" font-weight="700">配置</text>')
    for i, row in enumerate(best_rows):
        y = top + 24 + i * 52
        parts.append(f'<rect x="{lx}" y="{y - 11}" width="14" height="14" fill="{PALETTE[i % len(PALETTE)]}"/>')
        name = row["name"]
        short = name if len(name) <= 30 else name[:27] + "..."
        parts.append(f'<text x="{lx + 22}" y="{y}" class="small">{esc(short)}</text>')
        parts.append(f'<text x="{lx + 22}" y="{y + 16}" class="small">RMSE={fmt(to_float(row["rmse"]), 2)}, MAE={fmt(to_float(row["mae"]), 2)}</text>')
    parts.append("</svg>")
    write_svg(output, parts)


def draw_phase_summary(rows: list[dict[str, str]], output: Path) -> None:
    grouped: dict[str, list[dict[str, str]]] = defaultdict(list)
    for r in rows:
        grouped[r["phase_file"]].append(r)
    summary = []
    for phase, items in grouped.items():
        best = min(items, key=lambda r: to_float(r["rmse"]))
        summary.append((to_float(best["rmse"]), to_float(best["mae"]), phase, best["name"], len(items)))
    summary.sort()

    width, height = 1250, 760
    left, top = 360, 100
    plot_w, row_h = 700, 42
    shown = summary[:14]
    xmin = min(x[0] for x in shown) * 0.98
    xmax = max(x[0] for x in shown) * 1.01
    parts = svg_header(width, height)
    parts.append('<text x="48" y="46" class="title">各实验批次最优RMSE</text>')
    parts.append('<text x="48" y="72" class="subtitle">每个批次只保留RMSE最优配置，用于观察不同调参阶段的收益。</text>')

    for tick in nice_ticks(xmin, xmax, 7):
        x = scale(tick, xmin, xmax, left, left + plot_w)
        parts.append(f'<line x1="{x:.1f}" y1="{top - 14}" x2="{x:.1f}" y2="{top + row_h * len(shown)}" class="grid"/>')
        parts.append(f'<text x="{x:.1f}" y="{top - 24}" text-anchor="middle" class="tick">{fmt(tick, 1)}</text>')

    for i, (rmse, mae, phase, name, count) in enumerate(shown):
        y = top + i * row_h
        label = phase_label(phase)
        label = label if len(label) <= 36 else label[:33] + "..."
        x = scale(rmse, xmin, xmax, left, left + plot_w)
        parts.append(f'<text x="{left - 14}" y="{y + 22}" text-anchor="end" class="small">{esc(label)}</text>')
        parts.append(f'<rect x="{left}" y="{y + 7}" width="{x - left:.1f}" height="22" fill="#329731" rx="3"/>')
        parts.append(f'<text x="{x + 8:.1f}" y="{y + 23}" class="small">RMSE={fmt(rmse, 2)} MAE={fmt(mae, 2)} n={count}</text>')
    parts.append("</svg>")
    write_svg(output, parts)


def draw_single_parameter_rmse(rows: list[dict[str, str]], param: str, output: Path) -> bool:
    points = []
    for r in rows:
        x = to_float(r.get(param))
        y = to_float(r.get("rmse"))
        if math.isfinite(x) and math.isfinite(y):
            points.append((x, y, r))
    if len({p[0] for p in points}) < 2:
        return False

    grouped: dict[float, list[float]] = defaultdict(list)
    for x, y, _ in points:
        grouped[x].append(y)
    xs_unique = sorted(grouped)
    mins = [(x, min(grouped[x])) for x in xs_unique]
    meds = []
    for x in xs_unique:
        vals = sorted(grouped[x])
        n = len(vals)
        median = vals[n // 2] if n % 2 else (vals[n // 2 - 1] + vals[n // 2]) / 2
        meds.append((x, median))

    width, height = 980, 620
    left, right, top, bottom = 86, 44, 86, 86
    plot_w = width - left - right
    plot_h = height - top - bottom
    xvals = [p[0] for p in points]
    yvals = [p[1] for p in points]
    xmin, xmax = min(xvals), max(xvals)
    ymin, ymax = min(yvals), max(yvals)
    xpad = (xmax - xmin) * 0.08 if xmax > xmin else 1
    ypad = (ymax - ymin) * 0.08 if ymax > ymin else 1
    xmin -= xpad
    xmax += xpad
    ymin -= ypad
    ymax += ypad

    best = min(points, key=lambda p: p[1])

    parts = svg_header(width, height)
    parts.append(f'<text x="42" y="42" class="title">{esc(param)}与RMSE关系</text>')
    parts.append(
        f'<text x="42" y="68" class="subtitle">灰点为全部配置；红线为同一参数值下最小RMSE；蓝线为中位RMSE。</text>'
    )

    for tick in nice_ticks(xmin, xmax, 7):
        x = scale(tick, xmin, xmax, left, left + plot_w)
        parts.append(f'<line x1="{x:.1f}" y1="{top}" x2="{x:.1f}" y2="{top + plot_h}" class="grid"/>')
        parts.append(f'<text x="{x:.1f}" y="{top + plot_h + 24}" text-anchor="middle" class="tick">{fmt(tick, 3).rstrip("0").rstrip(".")}</text>')
    for tick in nice_ticks(ymin, ymax, 6):
        y = scale(tick, ymin, ymax, top + plot_h, top)
        parts.append(f'<line x1="{left}" y1="{y:.1f}" x2="{left + plot_w}" y2="{y:.1f}" class="grid"/>')
        parts.append(f'<text x="{left - 12}" y="{y + 4:.1f}" text-anchor="end" class="tick">{fmt(tick, 1)}</text>')

    parts.append(f'<line x1="{left}" y1="{top + plot_h}" x2="{left + plot_w}" y2="{top + plot_h}" class="axis"/>')
    parts.append(f'<line x1="{left}" y1="{top}" x2="{left}" y2="{top + plot_h}" class="axis"/>')
    parts.append(f'<text x="{left + plot_w / 2:.1f}" y="{height - 28}" text-anchor="middle" class="label">{esc(param)}</text>')
    parts.append(f'<text transform="translate(28 {top + plot_h / 2:.1f}) rotate(-90)" text-anchor="middle" class="label">RMSE</text>')

    # Draw raw points first, lightly.
    for x0, y0, r in points:
        x = scale(x0, xmin, xmax, left, left + plot_w)
        y = scale(y0, ymin, ymax, top + plot_h, top)
        parts.append(
            f'<circle cx="{x:.2f}" cy="{y:.2f}" r="3.2" fill="#6B7280" fill-opacity="0.28">'
            f'<title>{esc(r["name"])}\\n{esc(param)}={fmt(x0, 6)}, RMSE={fmt(y0, 6)}</title></circle>'
        )

    def polyline(series: list[tuple[float, float]], color: str, width_: float) -> None:
        coords = []
        for x0, y0 in series:
            x = scale(x0, xmin, xmax, left, left + plot_w)
            y = scale(y0, ymin, ymax, top + plot_h, top)
            coords.append(f"{x:.2f},{y:.2f}")
        parts.append(f'<polyline points="{" ".join(coords)}" fill="none" stroke="{color}" stroke-width="{width_}" stroke-linejoin="round" stroke-linecap="round"/>')
        for x0, y0 in series:
            x = scale(x0, xmin, xmax, left, left + plot_w)
            y = scale(y0, ymin, ymax, top + plot_h, top)
            parts.append(f'<circle cx="{x:.2f}" cy="{y:.2f}" r="4.2" fill="{color}" stroke="#FFFFFF" stroke-width="1"/>')

    polyline(meds, "#3378BC", 2.4)
    polyline(mins, "#FF4E48", 3.0)

    bx = scale(best[0], xmin, xmax, left, left + plot_w)
    by = scale(best[1], ymin, ymax, top + plot_h, top)
    parts.append(f'<circle cx="{bx:.2f}" cy="{by:.2f}" r="7" fill="#329731" stroke="#111827" stroke-width="1.2"/>')
    parts.append(f'<text x="{bx + 10:.1f}" y="{by - 10:.1f}" class="label" font-weight="700">全局最优</text>')

    lx, ly = left + plot_w - 210, top + 18
    parts.append(f'<rect x="{lx - 14}" y="{ly - 22}" width="224" height="86" fill="#FFFFFF" fill-opacity="0.88" stroke="#E5E7EB" rx="6"/>')
    parts.append(f'<circle cx="{lx}" cy="{ly}" r="4" fill="#6B7280" fill-opacity="0.45"/><text x="{lx + 14}" y="{ly + 4}" class="small">全部配置</text>')
    parts.append(f'<line x1="{lx - 4}" y1="{ly + 24}" x2="{lx + 8}" y2="{ly + 24}" stroke="#FF4E48" stroke-width="3"/><text x="{lx + 14}" y="{ly + 28}" class="small">同值最小RMSE</text>')
    parts.append(f'<line x1="{lx - 4}" y1="{ly + 48}" x2="{lx + 8}" y2="{ly + 48}" stroke="#3378BC" stroke-width="3"/><text x="{lx + 14}" y="{ly + 52}" class="small">同值中位RMSE</text>')

    parts.append("</svg>")
    write_svg(output, parts)
    return True


def draw_single_parameter_set(rows: list[dict[str, str]], output_dir: Path) -> list[str]:
    params = [
        "freq_weight",
        "multiband_shrinkage",
        "lambda_high",
        "ridge",
        "huber_delta",
        "huber_iters",
        "outlier_reject_iters",
        "outlier_reject_sigma",
        "outlier_reject_max_fraction",
        "preferred_top_k",
        "spectral_top_k",
        "num_peaks",
        "power_cum",
    ]
    written = []
    single_dir = output_dir / "single_parameter_rmse"
    single_dir.mkdir(parents=True, exist_ok=True)
    for param in params:
        out = single_dir / f"rmse_vs_{param}.svg"
        if draw_single_parameter_rmse(rows, param, out):
            written.append(str(out.relative_to(output_dir)))
    return written


def metric_min_series(rows: list[dict[str, str]], param: str, metric: str) -> list[tuple[float, float]]:
    grouped: dict[float, list[float]] = defaultdict(list)
    for row in rows:
        x = to_float(row.get(param))
        y = to_float(row.get(metric))
        if math.isfinite(x) and math.isfinite(y):
            grouped[x].append(y)
    return [(x, min(grouped[x])) for x in sorted(grouped)]


def draw_combined_parameter_metric(rows: list[dict[str, str]], metric: str, output: Path) -> None:
    params = [
        "freq_weight",
        "multiband_shrinkage",
        "lambda_high",
        "ridge",
        "huber_delta",
        "huber_iters",
        "outlier_reject_iters",
        "outlier_reject_sigma",
        "outlier_reject_max_fraction",
        "preferred_top_k",
        "spectral_top_k",
        "num_peaks",
        "power_cum",
    ]
    panels = []
    for param in params:
        series = metric_min_series(rows, param, metric)
        if len(series) >= 2:
            panels.append((param, series))

    cols = 4
    panel_w, panel_h = 360, 230
    margin_x, margin_y = 34, 26
    legend_h = 0
    rows_n = math.ceil(len(panels) / cols)
    width = margin_x * 2 + cols * panel_w
    height = margin_y * 2 + legend_h + rows_n * panel_h
    parts = svg_header(width, height)

    line_color = "#FF4E48" if metric == "rmse" else "#3378BC"

    for idx, (param, series) in enumerate(panels):
        col = idx % cols
        row = idx // cols
        ox = margin_x + col * panel_w
        oy = margin_y + legend_h + row * panel_h
        left, right, top, bottom = 54, 20, 28, 45
        x0, y0 = ox + left, oy + top
        plot_w, plot_h = panel_w - left - right, panel_h - top - bottom
        x_vals = [x for x, _ in series]
        y_vals = [y for _, y in series]
        xmin, xmax = min(x_vals), max(x_vals)
        ymin, ymax = min(y_vals), max(y_vals)
        xpad = (xmax - xmin) * 0.08 if xmax > xmin else 1.0
        ypad = (ymax - ymin) * 0.10 if ymax > ymin else 1.0
        xmin -= xpad
        xmax += xpad
        ymin -= ypad
        ymax += ypad

        parts.append(f'<text x="{ox + panel_w / 2:.1f}" y="{oy + 16}" text-anchor="middle" class="label" font-weight="700">{esc(param)}</text>')

        for tick in nice_ticks(xmin, xmax, 4):
            x = scale(tick, xmin, xmax, x0, x0 + plot_w)
            parts.append(f'<line x1="{x:.1f}" y1="{y0}" x2="{x:.1f}" y2="{y0 + plot_h}" class="grid"/>')
            parts.append(f'<text x="{x:.1f}" y="{y0 + plot_h + 18}" text-anchor="middle" class="tick">{fmt(tick, 2).rstrip("0").rstrip(".")}</text>')
        for tick in nice_ticks(ymin, ymax, 4):
            y = scale(tick, ymin, ymax, y0 + plot_h, y0)
            parts.append(f'<line x1="{x0}" y1="{y:.1f}" x2="{x0 + plot_w}" y2="{y:.1f}" class="grid"/>')
            parts.append(f'<text x="{x0 - 8}" y="{y + 3:.1f}" text-anchor="end" class="tick">{fmt(tick, 0)}</text>')

        parts.append(f'<line x1="{x0}" y1="{y0 + plot_h}" x2="{x0 + plot_w}" y2="{y0 + plot_h}" class="axis"/>')
        parts.append(f'<line x1="{x0}" y1="{y0}" x2="{x0}" y2="{y0 + plot_h}" class="axis"/>')

        def draw_line(series_: list[tuple[float, float]], color: str) -> None:
            coords = []
            for x_raw, y_raw in series_:
                x = scale(x_raw, xmin, xmax, x0, x0 + plot_w)
                y = scale(y_raw, ymin, ymax, y0 + plot_h, y0)
                coords.append(f"{x:.2f},{y:.2f}")
            parts.append(f'<polyline points="{" ".join(coords)}" fill="none" stroke="{color}" stroke-width="2.4" stroke-linecap="round" stroke-linejoin="round"/>')
            for x_raw, y_raw in series_:
                x = scale(x_raw, xmin, xmax, x0, x0 + plot_w)
                y = scale(y_raw, ymin, ymax, y0 + plot_h, y0)
                parts.append(f'<circle cx="{x:.2f}" cy="{y:.2f}" r="2.7" fill="{color}" stroke="#FFFFFF" stroke-width="0.8"/>')

        draw_line(series, line_color)

    parts.append("</svg>")
    write_svg(output, parts)


def write_markdown_summary(rows: list[dict[str, str]], output_dir: Path, csv_path: Path) -> None:
    best_rmse = min(rows, key=lambda r: to_float(r["rmse"]))
    best_mae = min(rows, key=lambda r: to_float(r["mae"]))
    lines = [
        "# NUFROST参数扫描图表说明",
        "",
        f"- 源文件：`{csv_path}`",
        f"- 有效配置数量：{len(rows)}",
        f"- 最低全波段RMSE：{fmt(to_float(best_rmse['rmse']), 6)}，配置：`{best_rmse['name']}`",
        f"- 最低全波段MAE：{fmt(to_float(best_mae['mae']), 6)}，配置：`{best_mae['name']}`",
        "",
        "## 图件",
        "",
        "- `parameter_sweep_rmse_mae_scatter.svg`：总体RMSE/MAE散点分布。",
        "- `parameter_sweep_top_rmse_mae.svg`：全波段RMSE最优Top12配置。",
        "- `parameter_sweep_freq_shrinkage_heatmap.svg`：`freq_weight × multiband_shrinkage`参数敏感性。",
        "- `parameter_sweep_indices_best_configs.svg`：最优配置在NDVI、NDWI、NDMI、NDSI、NBR和EVI上的RMSE对比。",
        "- `parameter_sweep_phase_best.svg`：各实验批次的最优RMSE。",
        "- `single_parameter_rmse/`：单个参数与RMSE的关系图，每张图叠加全部配置点、同值最小RMSE和同值中位RMSE。",
        "- `parameter_sweep_single_parameter_rmse_grid.svg`：单参数与RMSE关系的小多图。",
        "- `parameter_sweep_single_parameter_mae_grid.svg`：单参数与MAE关系的小多图。",
        "",
        "## 最优配置Top10",
        "",
        "| rank | phase | name | RMSE | MAE | freq_weight | multiband_shrinkage | lambda_high |",
        "| ---: | --- | --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for i, row in enumerate(sorted(rows, key=lambda r: to_float(r["rmse"]))[:10], 1):
        lines.append(
            "| "
            + " | ".join(
                [
                    str(i),
                    phase_label(row["phase_file"]),
                    f"`{row['name']}`",
                    fmt(to_float(row["rmse"]), 6),
                    fmt(to_float(row["mae"]), 6),
                    fmt(to_float(row.get("freq_weight")), 6),
                    fmt(to_float(row.get("multiband_shrinkage")), 6),
                    fmt(to_float(row.get("lambda_high")), 6),
                ]
            )
            + " |"
        )
    (output_dir / "parameter_sweep_figures_summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--csv",
        type=Path,
        default=Path("nufrost/docs/experiments/nufrost_parameter_sweeps_2026-06-24_summary.csv"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("master-thesis-file/assets/figures/parameter_sweeps"),
    )
    args = parser.parse_args()

    rows = read_rows(args.csv)
    if not rows:
        raise SystemExit(f"No valid rows found in {args.csv}")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    draw_scatter(rows, args.output_dir / "parameter_sweep_rmse_mae_scatter.svg")
    draw_top_bars(rows, args.output_dir / "parameter_sweep_top_rmse_mae.svg")
    draw_heatmap(rows, args.output_dir / "parameter_sweep_freq_shrinkage_heatmap.svg")
    draw_index_profile(rows, args.output_dir / "parameter_sweep_indices_best_configs.svg")
    draw_phase_summary(rows, args.output_dir / "parameter_sweep_phase_best.svg")
    single_plots = draw_single_parameter_set(rows, args.output_dir)
    draw_combined_parameter_metric(
        rows,
        "rmse",
        args.output_dir / "parameter_sweep_single_parameter_rmse_grid.svg",
    )
    draw_combined_parameter_metric(
        rows,
        "mae",
        args.output_dir / "parameter_sweep_single_parameter_mae_grid.svg",
    )
    write_markdown_summary(rows, args.output_dir, args.csv)

    print(f"Wrote figures to {args.output_dir}")
    print(f"Wrote {len(single_plots)} single-parameter RMSE plots")
    best = min(rows, key=lambda r: to_float(r["rmse"]))
    print(
        "Best RMSE:",
        best["name"],
        fmt(to_float(best["rmse"]), 6),
        "MAE",
        fmt(to_float(best["mae"]), 6),
    )


if __name__ == "__main__":
    main()
