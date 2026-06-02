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
