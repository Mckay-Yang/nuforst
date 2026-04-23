---
name: streaming-raster-evaluation
description: Use when evaluating remote sensing time-series reconstruction algorithms on large raster cubes, especially when per-pixel I/O causes hours-long stalls or when sampling random points and gap candidates from streaming raster sources
---

# Streaming Raster Time-Series Evaluation

## Overview

Evaluating reconstruction algorithms on full remote sensing imagery requires scanning all pixels for valid observation counts, missing ratios, and gap statistics. Doing this via per-pixel `read_pixel_series` calls is catastrophically slow; batch window reads with vectorized statistics is the only viable approach.

## When to Use

- Evaluating algorithms on raster cubes too large for memory (e.g., 14202 × 1002 × 1147)
- Sampling random points or gap candidates from streaming raster sources
- Computing per-pixel statistics (valid counts, missing ratios, native gap lengths)
- Any workflow that iterates over pixels and calls `read_pixel_series` in a loop

## Anti-Pattern: Per-Pixel I/O

```python
for r, c in candidates:
    y_ts = source.read_pixel_series(int(r), int(c))  # Each call = one rasterio window read
    valid_mask = np.isfinite(y_ts) & np.isfinite(t_days)
    missing_ratio = 1.0 - np.sum(valid_mask) / len(t_days)
```

For 10,000 candidates × 14,202 time steps, this is ~10,000 random I/O operations. A single such scan can take hours.

## Core Pattern: Batch Window Scan

Compute ALL per-pixel statistics in a single pass over spatial windows:

```python
def scan_pixel_stats(source, t_days, block_shape=(256, 256)):
    h, w = source.metadata()["height"], source.metadata()["width"]
    t_valid_mask = np.isfinite(t_days)
    n_valid_t = int(np.sum(t_valid_mask))

    valid_counts = np.zeros((h, w), dtype=np.int32)
    missing_ratios = np.ones((h, w), dtype=np.float32)
    native_gap_days = np.full((h, w), np.inf, dtype=np.float32)

    for row_slice, col_slice in source.iter_windows(block_shape=block_shape):
        arr = source.read_window(row_slice, col_slice)          # ONE read per window
        valid_mask = np.isfinite(arr) & t_valid_mask[:, None, None]
        vc = np.sum(valid_mask, axis=0).astype(np.int32)
        valid_counts[row_slice, col_slice] = vc

        has_obs = vc > 0
        mr = np.ones_like(vc, dtype=np.float32)
        mr[has_obs] = 1.0 - vc[has_obs].astype(np.float32) / n_valid_t
        missing_ratios[row_slice, col_slice] = mr

        for dr, dc in np.argwhere(vc >= 2):                     # Only pixels with 2+ obs
            ts_valid = np.sort(t_days[valid_mask[:, dr, dc]])
            if len(ts_valid) > 1:
                native_gap_days[dr + row_slice.start, dc + col_slice.start] = float(np.max(np.diff(ts_valid)))

    return {"valid_counts": valid_counts, "missing_ratios": missing_ratios, "native_gap_days": native_gap_days}
```

**Key insight:** `read_window` reads a 256×256×T block in ONE I/O operation, vs 65,536 individual `read_pixel_series` calls for the same spatial extent.

## Caching Strategy

Compute stats ONCE per spatial chunk, reuse across all evaluation stages:

```python
_stats_cache = {}

def get_pixel_stats(source, t_days, loc_id, min_obs):
    if loc_id not in _stats_cache:
        _stats_cache[loc_id] = scan_pixel_stats(source, t_days)
    return _stats_cache[loc_id]
```

Pass `precomputed_stats` dict to downstream functions (`sample_random_points_from_source`, `scan_gap_candidates_from_source`) to avoid redundant scans.

## Common Mistakes

| Mistake | Symptom | Fix |
|---------|---------|-----|
| `int(weights.sum())` as sample cap | Only 1 sample from 20,000 requested | Use `min(num_points, len(valid_pixels))` — normalized weights always sum to 1.0 |
| Calling `read_pixel_series` in a loop | Hours of wall time, no log output | Use `read_window` + vectorized operations |
| Re-scanning per evaluation stage | Same stats computed 4× per chunk | Cache stats dict per loc_id |
| Full-candidate enumeration | `max_candidates = height * width` | Cap at 50,000; random subsample is sufficient |
| No progress logging | User thinks kernel is dead | Log window progress (every 10%) |

## Performance Reference

For a 1002×1147×14202 Sentinel-2 cube:

| Method | Time | I/O ops |
|--------|------|---------|
| Per-pixel `read_pixel_series` (10k candidates) | ~1 hour | ~10,000 |
| Batch window scan (256×256 blocks) | ~8 min | ~20 |
| Cached stats reuse | 0 (lookup) | 0 |

## Quick Reference

- **Block shape:** `(256, 256)` balances memory and I/O efficiency
- **Thresholds:** `valid_counts >= max(min_obs + 3, 15)` for gap candidates; `>= max(min_obs + 1, 3)` for random points
- **Missing ratio:** `1.0 - valid_count / n_valid_timesteps` (denominator = count of finite t_days, NOT total t_days)
- **Native gap days:** `max(diff(sorted_valid_timestamps))` per pixel — this is the Python-loop bottleneck; only compute for pixels with ≥2 valid obs
