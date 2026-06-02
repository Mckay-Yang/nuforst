# Learnings — Rust Full-Scene Parity

## Task 2: Sentinel-2 Location Discovery (2026-06-02)

### Implemented
- `full_scene.rs` module in `rust/nufrost-gdal/` with:
  - `location_token(lon, lat)` — mirrors Python `_location_token`
  - `location_output_token(lon, lat)` — 6 decimal places for VRT naming
  - `sentinel_band_sort_key(name)` — extracts `(u32, String)` from band name
  - `sorted_band_names(&BTreeMap)` — sorts keys by sentinel convention
  - `discover_sentinel_band_stacks(data_dir, lon, lat)` — globs `*{token}*.tif`, groups by band

### Decisions
- Used `std::fs::read_dir` instead of a glob crate — avoids extra dependency
- BTreeMap with String keys (lexicographic ordering); `sorted_band_names()` provides
  Python-compatible numeric ordering when needed
- Multi-chunk bands kept as `Vec<PathBuf>` — VRT construction deferred to separate task
- Added `regex` crate to workspace dependencies (was not previously present)

### Test Results
- 19 tests pass (7 new full_scene tests + 12 existing)
- Real data test: lon=104.2595, lat=31.2170 → correctly discovers B2, B3, B4, B8, B11, B12
- Band ordering verified: B2 < B3 < B4 < B8 < B11 < B12 (matches Python)

### Python Parity Map
| Python | Rust |
|--------|------|
| `_location_token(lon, lat)` | `location_token(lon, lat)` |
| `_sentinel_band_sort_key(name)` | `sentinel_band_sort_key(name)` |
| `discover_location_band_stacks(...)` | `discover_sentinel_band_stacks(...)` |
| `sorted(stacks.keys(), key=...)` | `sorted_band_names(&stacks)` |

## Task 6: Shared Spectral Frequency Pool (2026-06-02)

### Key decisions
- `compute_spectrum_direct`, `select_peaks_adaptive`, `refine_parabolic` were made `pub` in nufrost-core to enable cross-crate frequency pool construction. These are now re-exported from `lib.rs`.
- The `build_shared_frequency_pool()` lives in `full_scene.rs` (nufrost-gdal crate) because it operates on band cubes and is only needed for full-scene reconstruction.
- `compute_spectrum_direct` does NOT center the time series before DFT (unlike Python). This is mathematically fine because centering only affects the DC component, which is filtered out by `ignore_dc_hz`. Power magnitudes at k>0 are invariant under translation.

### Patterns
- `shared_spectral` mode: caller provides pre-computed frequencies via `nufrost_pixel_with_shared()`. If called without shared freqs, falls back to per-pixel `"spectral"` — NOT `"hybrid"`.
- The `select_frequencies()` function's `shared_spectral → spectral` fallback is separate from `nufrost_fit_pixel`'s fallback. Both were updated.
- Config defaults now exactly match `config/nufrost.json`: `frequency_selection="shared_spectral"`, `spectral_top_k=8`.

### Gotchas
- The `s` macro import from ndarray was unused — removed to fix compilation warning.
- `next_even` was imported but unused (the function uses `nufrost_core::next_even` inline) — removed from imports.

## Task 7: Full-scene CLI Command (2026-06-02)

### Key decisions
- `read_all_bands()` was made `pub` in nufrost-gdal to let the CLI load band cubes without exposing the private `read_all_bands` helper.
- `extract_raw_band_descriptions()` added as a new public function returning `Vec<String>` of ISO timestamp substrings from GDAL band descriptions. Uses `find_timestamp_substring` which returns `Option<&str>` — need `.to_string()` conversion.
- `read_all_bands_window()` added to read top-left windows from all bands (matching Python's `RSCube._read_tif()` 512×512 window convention).
- Rayon `par_iter()` used for per-band parallel reconstruction within each method, keeping row-level rayon parallelism for per-pixel loops.

### Patterns
- Per-pixel reconstruction loop follows the same pattern as `reconstruct_single_band()` in nufrost-gdal but returns `Array2<f64>` instead of writing to GeoTIFF.
- Timestamp conversion: ISO strings → `parse_iso8601_to_epoch_seconds` → per-band relative days using earliest epoch of that band+target set. Target time converted same way.
- `nufrost_pixel_with_shared` parameter order: `(ts, obs, target, config, freqs)` — config BEFORE freqs.

### Gotchas
- `find_timestamp_substring` returns `Option<&str>`, not `Option<String>`. Must use `.unwrap_or("").to_string()`.
- `ndarray::concatenate` requires views as `&[ArrayViewD]` — use `.view()` on each chunk array.
- The `Path` import was removed from CLI because `std::path::Path` is used directly in function signatures.

