---
name: nufrost-project
description: Use when working on the NUFROST codebase — understanding algorithm architecture, evaluation pipeline, configuration, or data flow
---

# NUFROST Project Reference

## Overview

NUFROST is a remote sensing time-series reconstruction framework combining NUFFT + Huber-Ridge regression. The repo contains three independent algorithms (NUFROST, Zhu2015, HANTS), a streaming evaluation pipeline, and Jupyter notebook workflows.

## Architecture

```
src/
  nufrost.py      # NUFROST: NUFFT → frequency selection → Huber-Ridge IRLS
  hants.py        # HANTS:  Harmonic analysis with iterative outlier rejection
  zhu2015.py      # Zhu2015: Lasso-based piecewise harmonic fitting
  evaluation.py   # Evaluation: sampling, metrics, streaming vs cube entrypoints
  data_loader.py  # RSCube (full NPZ) + TimeSeriesRasterSource (streaming VRT)
  model_params.py # Parameter caching (unused in current notebook flow)
config/
  config.yaml     # Canonical defaults (overrides Args dataclass when using build_args)
  settings.py     # Args dataclass + build_args() merge logic
notebooks/
  local_evals.ipynb    # Main local evaluation entrypoint
```

## Three Algorithms Are Independent

- **NUFROST** (nufrost.py): The proposed method. NUFFT spectrum → hybrid frequency selection (prior + data-driven) → parabolic refinement → IRLS with Huber loss + frequency-weighted Ridge.
- **Zhu2015** (zhu2015.py): Comparison baseline. Lasso regression on fixed harmonic basis (annual, semi-annual, tri-annual) with piecewise segmentation.
- **HANTS** (hants.py): Comparison baseline. Iterative harmonic fitting with outlier rejection (FET threshold).

Do NOT treat Zhu2015 or HANTS as sub-components of NUFROST.

## Data Flow

```
GeoTIFF tiles → find_image_chunks() → VRT (cached in data/cache/local/vrts/)
                                           ↓
                        ┌─── RSCube.load() → NPZ (data/cache/local/npz/)  [full-cube path]
                        └─── TimeSeriesRasterSource  [streaming path, no NPZ]
                                           ↓
                        open_evaluation_source() → {source, t_days, timestamps}
                                           ↓
                        scan_pixel_stats() → cached {valid_counts, missing_ratios, native_gap_days}
                                           ↓
                        sample / evaluate → CSV in data/output/
```

**Notebook uses streaming path only.** NPZ path exists for backward compat and `reconstruct_nufrost()` CLI.

## Configuration Precedence

Python overrides > CLI flags > YAML defaults > Args dataclass fallbacks.

**Known inconsistency:** `config.yaml` and `Args` dataclass differ on:
- `ridge`: YAML=0.005, Args=1e-2, `fit_nufrost_pixel_params` signature=1e-2
- `ignore_dc_hz`: YAML=1e-10, Args=1e-9
- `num_peaks`: YAML=10, Args=8

When using `build_args({})`, YAML wins. When calling `fit_nufrost_pixel_params` directly without passing `ridge_lam`, the function default (1e-2) wins.

## Evaluation Pipeline (notebooks/local_evals.ipynb)

Four experiment stages per spatial/band chunk:
1. **Ablation**: NUFROST variants (full, w/o preferred freqs, w/o parabolic, w/o Huber, w/o ridge, w/o trend) + baselines
2. **Sparse sweep**: Vary number of random points (1000–20000)
3. **Gap sweep**: Vary continuous gap length (derived from gap index targets)
4. **Repeatability**: Repeat random + gap with different seeds

**Checkpoint mechanism:** Results written incrementally to CSV. On restart, `load_done_keys()` skips completed (Image, Scenario, Variant) / (Image, NumPoints) / (Image, GapLength) tuples. No checkpoint for intermediate computation (random-point pool, gap candidates).

## Key Parameters for Paper

| Symbol | Parameter | Default | Location |
|--------|-----------|---------|----------|
| $\lambda$ | `ridge` / `ridge_lam` | 0.005 (YAML) | Eq.(7) |
| $\gamma$ | `freq_weight` | 2.0 | Eq.(8) |
| $\delta$ | `huber_delta` | 1.5 | Eq.(5) |
| — | `huber_iters` | 3 | IRLS iterations |
| $\eta$ | `power_cum` | 0.7 | Cumulative energy threshold |
| $\epsilon_{tol}$ | `spectral_merge_tol` | 0.15 | Frequency snapping tolerance |
| $\nu_{min}$ | `ignore_dc_hz` | 1e-10 (YAML) | Min frequency threshold |
| — | `preferred_periods_days` | "365.25,182.625,91.3125,30.4375" | Annual/semi/seasonal/monthly |
| — | `min_obs` | 12 | Min valid observations per pixel |

## Output Shapes

- NUFROST / HANTS: single-band GeoTIFF
- Zhu2015: 2-band GeoTIFF (prediction + QA)

## Testing

- Only `tests/test_data_loader.py` exists; hardcodes local absolute paths and contains `breakpoint()`
- Do NOT assume `pytest tests/` is a clean smoke test on another machine
- No lint, formatter, typecheck, or CI config exists
