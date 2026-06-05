#!/usr/bin/env python3
"""
Python parity fixture generator for Rust rewrite.

Generates deterministic synthetic time-series fixtures and a small real-window
fixture, runs NUFROST/HANTS/Zhu2015 on each, and saves inputs, configs, and
expected predictions under tests/fixtures/rust_parity/.

Usage:
    conda activate geo-science
    python scripts/generate_parity_fixtures.py [--output-dir OUTPUT_DIR]

Reproducibility:
    Running this script twice into separate output dirs MUST produce identical
    checksums.  np.random.seed(42) is used throughout.
"""

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import rasterio

# Ensure src/ is importable from the worktree root.
HERE = Path(__file__).resolve().parent
WORKTREE = HERE.parent
sys.path.insert(0, str(WORKTREE))

# ── deterministic RNG ────────────────────────────────────────────────────
SEED = 42
FIXED_TIMESTAMP = "2026-01-01T00:00:00Z"
np.random.seed(SEED)
RNG = np.random.RandomState(SEED)

# ── algorithm per-pixel entry points ─────────────────────────────────────
from src.nufrost import predict_single_pixel
from src.hants import hants_pixel
from src.zhu2015 import fit_predict_pixel

# ── default configs (mirror config/*.json) ───────────────────────────────
NUFROST_DEFAULTS: Dict[str, Any] = {
    "modes": 4096,
    "eps": 1e-12,
    "num_peaks": 10,
    "power_cum": 0.7,
    "ignore_dc_hz": 1e-10,
    "refine_peaks": True,
    "include_trend": True,
    "ridge_lam": 0.005,          # config key "ridge"
    "freq_weight": 2.0,
    "huber_iters": 3,
    "huber_delta": 0.05,
    "min_obs": 12,
}

HANTS_DEFAULTS: Dict[str, Any] = {
    "nof": 3,
    "sf": "high",
    "fet": 500.0,
    "dod": 5,
    "valid_min": None,
    "valid_max": None,
    "period": 365.25,
}

ZHU2015_DEFAULTS: Dict[str, Any] = {
    "lasso_alpha": 0.1,
}

# ── tolerance guidance (used in README) ──────────────────────────────────
UNIT_TOLERANCES = {
    "nufrost":  {"atol": 1e-6, "rtol": 1e-5,  "note": "floating-point NUFFT"},
    "hants":    {"atol": 1e-6, "rtol": 1e-5,  "note": "lstsq linear algebra"},
    "zhu2015":  {"atol": 1e-6, "rtol": 5e-4,  "note": "LASSO solver differences"},
}

# ═══════════════════════════════════════════════════════════════════════════
#  Synthetic time-series generators
# ═══════════════════════════════════════════════════════════════════════════

def gen_simple_harmonic(n: int = 50, period: float = 365.25, noise_std: float = 0.05) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Clean harmonic: mean + annual + semi-annual, no gaps.

    Returns (t_days, y, valid_mask).
    """
    t = np.linspace(0, 2 * period, n, dtype=np.float64)
    w = 2 * np.pi / period
    y = (
        0.3
        + 0.5 * np.cos(w * t + 0.2)
        + 0.2 * np.cos(2 * w * t - 0.5)
        + noise_std * RNG.randn(n)
    )
    mask = np.ones(n, dtype=bool)
    return t, y.astype(np.float64), mask


def gen_gaps_outliers(n: int = 80, period: float = 365.25) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Harmonic with ~20 % gaps and ~5 % outliers.

    Returns (t_days, y, valid_mask).
    """
    t = np.sort(RNG.uniform(0, 3 * period, n)).astype(np.float64)
    w = 2 * np.pi / period
    y_clean = (
        0.3
        + 0.6 * np.cos(w * t + 0.3)
        + 0.15 * np.cos(2 * w * t)
        + 0.04 * RNG.randn(n)
    )
    y = y_clean.copy()
    mask = np.ones(n, dtype=bool)

    # 20 % random gaps
    gap_idx = RNG.choice(n, size=int(0.2 * n), replace=False)
    y[gap_idx] = np.nan
    mask[gap_idx] = False

    # 5 % outliers among valid points
    valid_pos = np.where(mask)[0]
    n_outl = max(1, int(0.05 * len(valid_pos)))
    outl_idx = RNG.choice(valid_pos, size=n_outl, replace=False)
    y[outl_idx] += RNG.choice([-1, 1], size=n_outl) * 3.0 * np.nanstd(y_clean)
    return t, y.astype(np.float64), mask


def gen_step_break(n: int = 100, period: float = 365.25) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Two-segment series with a structural break at t ≈ 2*period.

    Returns (t_days, y, valid_mask).
    """
    t = np.sort(RNG.uniform(0, 4 * period, n)).astype(np.float64)
    break_t = 2 * period
    w = 2 * np.pi / period
    y = np.empty(n, dtype=np.float64)

    seg1 = t <= break_t
    y[seg1] = (
        0.3
        + 0.5 * np.cos(w * t[seg1] - 0.3)
        + 0.2 * np.cos(2 * w * t[seg1] + 0.4)
        + 0.05 * RNG.randn(np.sum(seg1))
    )

    seg2 = t > break_t
    y[seg2] = (
        0.5                                         # mean shift
        + 0.35 * np.cos(w * t[seg2] + 1.2)          # phase/amplitude change
        + 0.08 * np.cos(2 * w * t[seg2])
        + 0.05 * RNG.randn(np.sum(seg2))
    )

    mask = np.ones(n, dtype=bool)
    # ~5 % gaps
    gap_idx = RNG.choice(n, size=int(0.05 * n), replace=False)
    y[gap_idx] = np.nan
    mask[gap_idx] = False

    return t, y.astype(np.float64), mask


# ═══════════════════════════════════════════════════════════════════════════
#  Per-algorithm prediction helpers
# ═══════════════════════════════════════════════════════════════════════════

def run_nufrost_pixel(t_days: np.ndarray, y: np.ndarray, mask: np.ndarray,
                      target_t_day: float, config: Optional[Dict[str, Any]] = None) -> float:
    """Run NUFROST predict_single_pixel on one time series.

    NUFROST works with any consistent time unit; we pass days.
    """
    cfg = dict(NUFROST_DEFAULTS)
    if config:
        cfg.update(config)

    y_masked = y.copy()
    y_masked[~mask] = np.nan

    t_sec = t_days * 86400.0
    target_sec = target_t_day * 86400.0

    pred, _ = predict_single_pixel(
        t_sec, y_masked, target_sec,
        nufft_modes=cfg["modes"],
        eps=cfg["eps"],
        num_peaks=cfg["num_peaks"],
        power_cum=cfg["power_cum"],
        ignore_dc_hz=cfg["ignore_dc_hz"],
        refine_peaks=cfg["refine_peaks"],
        include_trend=cfg["include_trend"],
        ridge_lam=cfg["ridge_lam"],
        freq_weight=cfg["freq_weight"],
        huber_iters=cfg["huber_iters"],
        huber_delta=cfg["huber_delta"],
        min_obs=cfg["min_obs"],
    )
    return float(pred)


def run_hants_pixel(t_days: np.ndarray, y: np.ndarray, mask: np.ndarray,
                    target_t_day: float, config: Optional[Dict[str, Any]] = None) -> float:
    """Run HANTS hants_pixel on one time series."""
    cfg = dict(HANTS_DEFAULTS)
    if config:
        cfg.update(config)

    y_masked = y.copy()
    y_masked[~mask] = np.nan

    pred = hants_pixel(
        t_days, y_masked, target_t_day,
        nof=cfg["nof"],
        sf=cfg["sf"],
        valid_min=cfg["valid_min"],
        valid_max=cfg["valid_max"],
        fet=cfg["fet"],
        dod=cfg["dod"],
        period=cfg["period"],
    )
    return float(pred)


def run_zhu2015_pixel(t_days: np.ndarray, y: np.ndarray, mask: np.ndarray,
                      target_t_day: float, config: Optional[Dict[str, Any]] = None) -> Tuple[float, int]:
    """Run Zhu2015 fit_predict_pixel on one time series.

    Returns (prediction, qa).  qa is the model order used (0-3) as a
    simple confidence indicator; the previous QA digit has been removed
    from the Python implementation.
    """
    cfg = dict(ZHU2015_DEFAULTS)
    if config:
        cfg.update(config)

    from src.zhu2015 import _select_model_order

    y_masked = y.copy()
    y_masked[~mask] = np.nan

    pred = fit_predict_pixel(
        t_days, y_masked, target_t_day,
        lasso_alpha=cfg["lasso_alpha"],
    )

    # Derive QA band from the model order that would have been selected.
    valid_mask = np.isfinite(y_masked)
    n_valid = int(np.sum(valid_mask))
    qa = _select_model_order(n_valid)

    return float(pred), qa


# ═══════════════════════════════════════════════════════════════════════════
#  Fixture save / hash utilities
# ═══════════════════════════════════════════════════════════════════════════

def _sha256_hex(path: Path) -> str:
    """Return SHA-256 hex digest of a file."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 16), b""):
            h.update(chunk)
    return h.hexdigest()


def _save_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(data, f, indent=2, sort_keys=True, default=str)


def _save_npz(path: Path, **arrays) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **arrays)


def _save_npy(path: Path, arr: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.save(path, arr)


def dir_checksums(root: Path) -> Dict[str, str]:
    """Compute SHA-256 for every regular file under *root*."""
    chk: Dict[str, str] = {}
    for f in sorted(root.rglob("*")):
        if f.is_file():
            chk[str(f.relative_to(root))] = _sha256_hex(f)
    return chk


# ═══════════════════════════════════════════════════════════════════════════
#  Synthetic fixture generation
# ═══════════════════════════════════════════════════════════════════════════

def build_synthetic_fixture(
    name: str,
    description: str,
    t_days: np.ndarray,
    y: np.ndarray,
    mask: np.ndarray,
    target_t_day: float,
    nufrost_cfg: Optional[Dict[str, Any]] = None,
    hants_cfg: Optional[Dict[str, Any]] = None,
    zhu2015_cfg: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Run all three algorithms on one synthetic series and return metadata."""

    print(f"  Running NUFROST on {name} ...")
    nufrost_pred = run_nufrost_pixel(t_days, y, mask, target_t_day, nufrost_cfg)

    print(f"  Running HANTS on {name} ...")
    hants_pred = run_hants_pixel(t_days, y, mask, target_t_day, hants_cfg)

    print(f"  Running Zhu2015 on {name} ...")
    zhu_pred, zhu_qa = run_zhu2015_pixel(t_days, y, mask, target_t_day, zhu2015_cfg)

    return {
        "name": name,
        "description": description,
        "nufrost_prediction": float(nufrost_pred),
        "hants_prediction": float(hants_pred),
        "zhu2015_prediction": float(zhu_pred),
        "zhu2015_qa": int(zhu_qa),
    }


def save_synthetic_fixture(output_dir: Path, fixture: Dict[str, Any],
                           t_days: np.ndarray, y: np.ndarray, mask: np.ndarray,
                           target_t_day: float, config: Dict[str, Any]) -> None:
    """Save one synthetic fixture to disk."""
    d = output_dir / fixture["name"]
    d.mkdir(parents=True, exist_ok=True)

    _save_npz(
        d / "data.npz",
        timestamps_days=t_days,
        observations=y,
        valid_mask=mask,
        target_time_day=np.array(target_t_day),
        nufrost_prediction=np.array(fixture["nufrost_prediction"]),
        hants_prediction=np.array(fixture["hants_prediction"]),
        zhu2015_prediction=np.array(fixture["zhu2015_prediction"]),
        zhu2015_qa=np.array(fixture["zhu2015_qa"]),
    )

    _save_json(d / "config.json", {
        "name": fixture["name"],
        "description": fixture["description"],
        "n_observations": len(y),
        "n_valid": int(np.sum(mask)),
        "target_time_day": target_t_day,
        "config": config,
        "seed": SEED,
        "generator_version": "1.0.0",
    })


# ═══════════════════════════════════════════════════════════════════════════
#  Real-window fixture (tiny spatial window from test GeoTIFF)
# ═══════════════════════════════════════════════════════════════════════════

def build_real_window_fixture(
    tif_path: Path,
    output_dir: Path,
    band_limit: int = 200,
    window: Tuple[int, int, int, int] = (0, 2, 0, 4),
) -> Dict[str, Any]:
    """Extract a tiny spatial window, run all three algorithms per-pixel.

    Parameters
    ----------
    tif_path : Path
        Path to a single-band multitemporal GeoTIFF.
    band_limit : int
        Read at most this many bands (time steps).
    window : (row_start, row_end, col_start, col_end)
        Spatial window to extract.

    Returns metadata dict.
    """
    print(f"\n  Reading real window from {tif_path.name} ...")
    with rasterio.open(str(tif_path)) as src:
        n_bands = src.count
        use_bands = min(n_bands, band_limit)
        data = src.read(
            indexes=list(range(1, use_bands + 1)),
            window=rasterio.windows.Window.from_slices(
                (window[0], window[1]), (window[2], window[3]),
            ),
        )  # shape: (use_bands, rows, cols)
        n_cols = window[3] - window[2]
        n_rows = window[1] - window[0]
        descriptions = list(src.descriptions) if src.descriptions else []
        print(f"    Loaded shape: {data.shape}  band_limit={band_limit}  window={window}")

    timestamps_raw = []
    for b in range(use_bands):
        if b < len(descriptions) and descriptions[b]:
            timestamps_raw.append(descriptions[b].split("_")[0])
        else:
            timestamps_raw.append(f"band_{b+1}")

    # Convert timestamps to days from t0.
    from src.nufrost import timestamps_to_seconds
    t_sec = timestamps_to_seconds(np.array(timestamps_raw, dtype="U32"))
    t0_sec = float(np.min(t_sec))
    t_days = (t_sec - t0_sec) / 86400.0

    # Target: middle time step.
    target_idx = use_bands // 2
    target_t_day = t_days[target_idx]
    target_time_str = timestamps_raw[target_idx]

    # Run per-pixel reconstruction.
    nufrost_pred = np.full((n_rows, n_cols), np.nan, dtype=np.float32)
    hants_pred = np.full((n_rows, n_cols), np.nan, dtype=np.float32)
    zhu2015_pred = np.full((n_rows, n_cols), np.nan, dtype=np.float32)
    zhu2015_qa = np.full((n_rows, n_cols), -1, dtype=np.int32)

    print(f"    Running {n_rows}x{n_cols} = {n_rows*n_cols} pixels ...")
    for ri in range(n_rows):
        for ci in range(n_cols):
            y = data[:, ri, ci].astype(np.float64)
            mask = np.isfinite(y)

            pred_n = run_nufrost_pixel(t_days, y, mask, target_t_day)
            pred_h = run_hants_pixel(t_days, y, mask, target_t_day)
            pred_z, qa_z = run_zhu2015_pixel(t_days, y, mask, target_t_day)

            nufrost_pred[ri, ci] = pred_n
            hants_pred[ri, ci] = pred_h
            zhu2015_pred[ri, ci] = pred_z
            zhu2015_qa[ri, ci] = qa_z

    # Build matchable filenames.
    prefix = tif_path.stem

    _save_npy(output_dir / "inputs.npy", data.astype(np.float32))
    _save_npz(
        output_dir / "timestamps.npz",
        timestamps_raw=np.array(timestamps_raw, dtype="U32"),
        timestamps_days=t_days,
    )
    _save_npy(output_dir / f"{prefix}_nufrost_pred.npy", nufrost_pred)
    _save_npy(output_dir / f"{prefix}_hants_pred.npy", hants_pred)
    _save_npy(output_dir / f"{prefix}_zhu2015_pred.npy", zhu2015_pred)
    _save_npy(output_dir / f"{prefix}_zhu2015_qa.npy", zhu2015_qa.astype(np.int32))

    _save_json(output_dir / "info.json", {
        "source_tif": str(tif_path),
        "window": list(window),
        "band_limit": band_limit,
        "n_bands_used": use_bands,
        "n_rows": n_rows,
        "n_cols": n_cols,
        "target_time_str": target_time_str,
        "target_time_day": float(target_t_day),
        "target_time_idx": int(target_idx),
        "prediction_shape": [int(n_rows), int(n_cols)],
        "seed": SEED,
        "generator_version": "1.0.0",
    })

    _save_json(output_dir / "config.json", {
        "nufrost": NUFROST_DEFAULTS,
        "hants": HANTS_DEFAULTS,
        "zhu2015": ZHU2015_DEFAULTS,
    })

    return {
        "name": "small_window",
        "n_rows": n_rows,
        "n_cols": n_cols,
        "n_bands": use_bands,
        "target_time_str": target_time_str,
    }


# ═══════════════════════════════════════════════════════════════════════════
#  Manifest
# ═══════════════════════════════════════════════════════════════════════════

def write_manifest(output_dir: Path, results: List[Dict[str, Any]]) -> None:
    manifest: Dict[str, Any] = {
        "generated_at": FIXED_TIMESTAMP,
        "seed": SEED,
        "generator_version": "1.0.0",
        "tolerances": UNIT_TOLERANCES,
        "fixtures": [],
    }
    for r in results:
        entry: Dict[str, Any] = {"name": r["name"]}
        if "source_tif" in r:
            entry["type"] = "real_window"
            entry["n_rows"] = r.get("n_rows")
            entry["n_cols"] = r.get("n_cols")
            entry["n_bands"] = r.get("n_bands")
        else:
            entry["type"] = "synthetic"
            entry["description"] = r.get("description", "")
        # File listing
        fixture_dir = output_dir / ("real" if r.get("type") == "real_window" else "synthetic") / r["name"]
        if fixture_dir.exists():
            files = sorted(
                str(p.relative_to(output_dir))
                for p in fixture_dir.rglob("*") if p.is_file()
            )
            entry["files"] = files
        manifest["fixtures"].append(entry)

    _save_json(output_dir / "manifest.json", manifest)


# ═══════════════════════════════════════════════════════════════════════════
#  README generation
# ═══════════════════════════════════════════════════════════════════════════

README_CONTENT = """\
# Rust Parity Fixtures

Auto-generated by `scripts/generate_parity_fixtures.py`.  These fixtures
provide deterministic inputs and expected outputs for NUFROST, HANTS, and
Zhu2015 that the Rust rewrite must match within tolerance.

## Regeneration

```bash
conda activate geo-science
python scripts/generate_parity_fixtures.py
```

To verify reproducibility:
```bash
python scripts/generate_parity_fixtures.py --output-dir /tmp/fixtures_a
python scripts/generate_parity_fixtures.py --output-dir /tmp/fixtures_b
diff -rq /tmp/fixtures_a /tmp/fixtures_b   # should be identical
```

## Fixtures

### synthetic/simple_harmonic
- **Tests**: clean harmonic (mean + annual + semi-annual), no gaps, no outliers.
- **Purpose**: happy-path unit test for all three algorithms.

### synthetic/gaps_outliers
- **Tests**: harmonic with ~20 % gaps and ~5 % outliers.
- **Purpose**: validate gap handling and outlier robustness.

### synthetic/step_break
- **Tests**: two-segment series with a structural break at t ≈ 2 years.
- **Purpose**: break detection for NUFROST/Zhu2015; robustness for HANTS.

### real/small_window
- **Tests**: 4×4 spatial window from a real Sentinel-2 B2 GeoTIFF.
- **Purpose**: integration test with real noise, gaps, and band metadata.

## Expected Tolerances

### Unit tests (per-pixel scalar predictions)

| Algorithm | atol    | rtol  | Note                           |
|-----------|---------|-------|--------------------------------|
| NUFROST   | 1e-6    | 1e-5  | floating-point NUFFT           |
| HANTS     | 1e-6    | 1e-5  | lstsq linear algebra           |
| Zhu2015   | 1e-6    | 5e-4  | LASSO solver convergence       |

### Raster-level tests (2D prediction maps)

| Algorithm | RMSE max | MAE max | MaxAE max | Note                   |
|-----------|----------|---------|-----------|------------------------|
| NUFROST   | 1e-4     | 1e-4    | 1e-3      | pixelwise double-check |
| HANTS     | 1e-4     | 1e-4    | 1e-3      | pixelwise double-check |
| Zhu2015   | 5e-3     | 5e-3    | 1e-2      | LASSO solver variance  |

## File Format

Each synthetic fixture directory contains:
- `data.npz` — timestamps, observations, mask, target time, all predictions
- `config.json` — algorithm parameters used

The real window fixture directory contains:
- `inputs.npy` — 3D cube (bands × rows × cols)
- `timestamps.npz` — timestamps (raw strings + days-from-t0)
- `<prefix>_<algo>_pred.npy` — 2D prediction maps
- `<prefix>_zhu2015_qa.npy` — QA band for Zhu2015
- `config.json` — algorithm parameters
- `info.json` — window metadata

`manifest.json` at the root lists all fixtures, their file paths, and tolerance
reference values.

## Notes

- Zhu2015 QA band encodes the model order (0-3) selected per-pixel;
  the previous QA digit was removed from the Python implementation.
- All synthetic data generated with `np.random.seed(42)`.
- Timestamps in synthetic fixtures are in days (floating-point).
"""


# ═══════════════════════════════════════════════════════════════════════════
#  Main
# ═══════════════════════════════════════════════════════════════════════════

def main() -> None:
    parser = argparse.ArgumentParser(description="Generate Rust parity fixtures")
    parser.add_argument(
        "--output-dir", default=None,
        help="Output root (default: tests/fixtures/rust_parity/)",
    )
    parser.add_argument(
        "--real-tif", default=None,
        help="Path to real GeoTIFF for window fixture "
             "(default: tests/fixtures/input/COPERNICUS_S2_HARMONIZED_B2_lon100.112_lat25.654-0000001024-0000000000.tif)",
    )
    args = parser.parse_args()

    output_dir = Path(args.output_dir) if args.output_dir else WORKTREE / "tests" / "fixtures" / "rust_parity"
    output_dir.mkdir(parents=True, exist_ok=True)

    synthetic_dir = output_dir / "synthetic"
    real_dir = output_dir / "real"

    results: List[Dict[str, Any]] = []

    print("=" * 70)
    print("Generating synthetic fixtures ...")
    print("=" * 70)

    # ── Fixture 1: simple harmonic ──
    t1, y1, m1 = gen_simple_harmonic()
    target_t1 = t1[len(t1) // 2]  # middle of the series
    fix1 = build_synthetic_fixture(
        "simple_harmonic",
        "Clean harmonic (mean + annual + semi-annual), no gaps.",
        t1, y1, m1, target_t1,
    )
    save_synthetic_fixture(
        synthetic_dir, fix1, t1, y1, m1, target_t1,
        {"nufrost": NUFROST_DEFAULTS, "hants": HANTS_DEFAULTS, "zhu2015": ZHU2015_DEFAULTS},
    )
    results.append(fix1)
    print(f"    ✓ {fix1['name']}: NUFROST={fix1['nufrost_prediction']:.6f}  "
          f"HANTS={fix1['hants_prediction']:.6f}  Zhu2015={fix1['zhu2015_prediction']:.6f}")

    # ── Fixture 2: gaps + outliers ──
    t2, y2, m2 = gen_gaps_outliers()
    target_t2 = t2[len(t2) // 2]
    fix2 = build_synthetic_fixture(
        "gaps_outliers",
        "Harmonic with ~20% gaps and ~5% outliers.",
        t2, y2, m2, target_t2,
    )
    save_synthetic_fixture(
        synthetic_dir, fix2, t2, y2, m2, target_t2,
        {"nufrost": NUFROST_DEFAULTS, "hants": HANTS_DEFAULTS, "zhu2015": ZHU2015_DEFAULTS},
    )
    results.append(fix2)
    print(f"    ✓ {fix2['name']}: NUFROST={fix2['nufrost_prediction']:.6f}  "
          f"HANTS={fix2['hants_prediction']:.6f}  Zhu2015={fix2['zhu2015_prediction']:.6f}")

    # ── Fixture 3: step break ──
    t3, y3, m3 = gen_step_break()
    target_t3 = t3[-10]  # near end of series (after the break)
    fix3 = build_synthetic_fixture(
        "step_break",
        "Two-segment series with structural break at ~2 years.",
        t3, y3, m3, target_t3,
    )
    save_synthetic_fixture(
        synthetic_dir, fix3, t3, y3, m3, target_t3,
        {"nufrost": NUFROST_DEFAULTS, "hants": HANTS_DEFAULTS, "zhu2015": ZHU2015_DEFAULTS},
    )
    results.append(fix3)
    print(f"    ✓ {fix3['name']}: NUFROST={fix3['nufrost_prediction']:.6f}  "
          f"HANTS={fix3['hants_prediction']:.6f}  Zhu2015={fix3['zhu2015_prediction']:.6f}")

    # ── Real-window fixture ──
    print("\n" + "=" * 70)
    print("Generating real-window fixture ...")
    print("=" * 70)

    default_tif = (
        WORKTREE / "tests" / "fixtures" / "input"
        / "COPERNICUS_S2_HARMONIZED_B2_lon100.112_lat25.654-0000001024-0000000000.tif"
    )
    tif_path = Path(args.real_tif) if args.real_tif else default_tif

    if not tif_path.exists():
        print(f"  ⚠  Real TIF not found: {tif_path}  — skipping real fixture.")
    else:
        real_output = real_dir / "small_window"
        real_output.mkdir(parents=True, exist_ok=True)
        real_result = build_real_window_fixture(tif_path, real_output)
        results.append({**real_result, "type": "real_window"})
        nufrost_pred = np.load(str(real_output / f"{tif_path.stem}_nufrost_pred.npy"))
        hants_pred = np.load(str(real_output / f"{tif_path.stem}_hants_pred.npy"))
        zhu2015_pred = np.load(str(real_output / f"{tif_path.stem}_zhu2015_pred.npy"))
        print(f"    ✓ small_window: {real_result['n_rows']}x{real_result['n_cols']} "
              f"({real_result['n_bands']} bands)")
        print(f"      NUFROST  min/mean/max: {np.nanmin(nufrost_pred):.4f} / {np.nanmean(nufrost_pred):.4f} / {np.nanmax(nufrost_pred):.4f}")
        print(f"      HANTS    min/mean/max: {np.nanmin(hants_pred):.4f} / {np.nanmean(hants_pred):.4f} / {np.nanmax(hants_pred):.4f}")
        print(f"      Zhu2015  min/mean/max: {np.nanmin(zhu2015_pred):.4f} / {np.nanmean(zhu2015_pred):.4f} / {np.nanmax(zhu2015_pred):.4f}")

    # ── Manifest ──
    write_manifest(output_dir, results)
    print(f"\n  ✓ manifest.json written")

    # ── README ──
    readme_path = output_dir / "README.md"
    readme_path.write_text(README_CONTENT)
    print(f"  ✓ README.md written")

    # ── Checksums ──
    print(f"\n{'='*70}")
    print("Checksums:")
    chks = dir_checksums(output_dir)
    for rel, h in sorted(chks.items()):
        print(f"  {h[:16]}  {rel}")

    print(f"\n{'='*70}")
    print("Done. All fixtures written to", output_dir)


if __name__ == "__main__":
    main()
