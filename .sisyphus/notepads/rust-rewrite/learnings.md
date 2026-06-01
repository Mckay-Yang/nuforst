# Learnings — Task 2: Bootstrap Rust Workspace

## Date: 2026-06-01

## What was done
- Created `rust-toolchain.toml` pinning stable 1.85.1
- Created workspace root `Cargo.toml` with resolver v2 and 4 member crates
- Created four crates under `rust/`: `nufrost-core` (lib), `nufrost-gdal` (lib), `nufrost-cli` (bin), `nufrost-py` (cdylib)
- All crates have trivial placeholder tests that pass
- Created `RUST_README.md` with GDAL system dependency docs and build-time env vars
- All 5 tests pass (2 core, 1 cli, 1 gdal, 1 py)

## Decisions and rationale

### gdal crate version: 0.16 → 0.19
Initial pin was `gdal 0.16` but gdal-sys 0.9.1 has no pre-built bindings for GDAL 3.10.
Upgraded to `gdal 0.19` (gdal-sys 0.12.0) which also lacks pre-built 3.10 bindings
but compiled with `bindgen` feature that required Rust 1.88+ due to transitive dep `home 0.5.12`.
Removed `bindgen` feature — gdal-sys 0.12.0 with `GDAL_VERSION=3.10.3` env var resolves
pre-built bindings correctly (gdal-sys 0.12.0 supports GDAL 3.10).

Actually: re-checked after full build. gdal 0.19 / gdal-sys 0.12.0 DOES have pre-built
bindings for GDAL 3.10. The key env vars are:
- `GDAL_VERSION=3.10.3`
- `GDAL_INCLUDE_DIR=<conda_prefix>/include`
- `GDAL_LIB_DIR=<conda_prefix>/lib`
- `DYLD_LIBRARY_PATH=<conda_prefix>/lib` (for runtime linking on macOS)

### pyo3 testing workaround
`nufrost-py` uses `crate-type = ["cdylib"]` with pyo3 `extension-module` feature.
Cargo test binaries can't link against Python symbols without `libpython`.
Workaround: test module does NOT use pyo3 types — uses a simple `#[test] fn placeholder_no_py()`
that asserts plain Rust. The pyo3-dependent code (py_add, nufrost_py module init) remains
at the top level for cdylib builds.

## Task 3: Python Parity Fixtures (2026-06-01)

### What was done
- Created `scripts/generate_parity_fixtures.py` — deterministic fixture generator
- Generated 4 fixtures under `tests/fixtures/rust_parity/`:
  - `synthetic/simple_harmonic` — clean harmonic, no gaps
  - `synthetic/gaps_outliers` — ~20% gaps, ~5% outliers
  - `synthetic/step_break` — structural break at t≈2 years
  - `real/small_window` — 4-col, 2-row window from Sentinel-2 B2 GeoTIFF
- Each fixture runs all three algorithms (NUFROST, HANTS, Zhu2015) on identical inputs
- Two-run reproducibility verified: all 16 files produce identical SHA-256 checksums
- All fixture files < 7KB each (well under 1MB limit)

### Key findings
- NUFROST `predict_single_pixel` works with any consistent time unit (days or seconds);
  it calls `_to_seconds_since_start()` which just subtracts min(t), so the internal math
  is unitless as long as target_t uses the same unit.
- Zhu2015 QA band has been removed from the current Python implementation;
  we derive a simple QA value from `_select_model_order(n_valid)` as a placeholder.
- The test GeoTIFF has shape (2, 1024) — only 2 rows, so window must respect that.
- `manifest.json` timestamp (`generated_at`) must be fixed for reproducibility.
- `rasterio.DatasetReader.descriptions` is only accessible while the context manager
  is active; must capture inside the `with` block.

### Conventions
- Fixture NPZ arrays: `timestamps_days`, `observations`, `valid_mask`, `target_time_day`,
  `nufrost_prediction`, `hants_prediction`, `zhu2015_prediction`, `zhu2015_qa`
- Tolerance guidance: unit tests `atol=1e-6, rtol=1e-5` (Zhu2015 `rtol=5e-4`),
  raster tests `RMSE max=1e-4, MAE max=1e-4`
- `np.random.seed(42)` throughout; `RandomState(42)` for separate RNG

### pkg-config not available in conda geo-science env
The `gdal-sys` build.rs uses `pkg-config` crate which invokes the `pkg-config` binary.
This binary was NOT in the conda `geo-science` env. Setting `GDAL_VERSION`,
`GDAL_INCLUDE_DIR`, and `GDAL_LIB_DIR` env vars bypasses the need for pkg-config
linking metadata.
