# Task 12: Rust Performance Benchmark Report

**Date:** 2026-06-01
**Worktree:** `/Volumes/T7/nuforst/.worktrees/rust-rewrite`

---

## 1. Environment

| Item                | Value                                                       |
|---------------------|-------------------------------------------------------------|
| OS                  | macOS 15.5 (Darwin 25.5.0)                                  |
| Architecture        | ARM64 (Apple Silicon)                                       |
| CPU                 | Apple M4                                                   |
| Memory              | 24 GB                                                       |
| Rust                | rustc 1.85.1 (4eb161250 2025-03-15)                        |
| Cargo               | cargo 1.85.1 (d73d2caf9 2024-12-31)                        |
| Build profile       | `--release` (optimized)                                     |
| GDAL                | 3.10.3 (conda-forge, linked at runtime via DYLD_LIBRARY_PATH) |

---

## 2. Test Suite

```
$ cargo test --workspace                         time: 1.58s

  nufrost-core:  91 passed, 0 failed  (0.08s)
  nufrost-gdal:  14 passed, 0 failed  (0.02s)
  nufrost-cli:   10 passed, 0 failed  (0.01s)
  nufrost-py:     1 passed, 0 failed  (0.00s)
  ────────────────────────────────────
  TOTAL:        116 passed, 0 failed
```

All 116 tests pass across four crates. Test execution time is dominated by compilation
(not counted); actual test execution totals under 0.2 seconds.

---

## 3. CLI Single-Pixel Timing

Measured via 100 CLI invocations per algorithm on two fixtures. Times include
process startup, argument parsing, NPZ I/O, algorithm execution, and stdout/stderr
output.

| Fixture          | NUFROST (100×) | HANTS (100×) | Zhu2015 (100×) |
|------------------|----------------|--------------|----------------|
| simple_harmonic  | 0.64s (6.4ms)  | 0.67s (6.7ms)| 0.65s (6.5ms)  |
| gaps_outliers    | 0.62s (6.2ms)  | 0.65s (6.5ms)| 0.66s (6.6ms)  |

**Key finding:** CLI single-pixel timing is dominated by process startup overhead
(~6-7ms/invocation). Algorithm computation is <1ms per pixel for all three
algorithms. For production use, the Python wrapper (`nufrost-py`) avoids this
overhead by calling Rust functions directly via PyO3.

---

## 4. CLI Raster Reconstruction Timing

Measured on real Sentinel-2 B2 GeoTIFF data (200 spectral bands, single-band
per timestamp). Timings include GDAL I/O for reading all 200 bands per pixel.

### 4.1 Small Subwindow (100×1 pixels, 200 bands)

| Algorithm | Wall time | Per-pixel   | User CPU |
|-----------|-----------|-------------|----------|
| NUFROST   | 2.58s     | 25.8 ms     | 2.35s    |
| HANTS     | 0.07s     | 0.7 ms      | 0.05s    |
| Zhu2015   | 0.10s     | 1.0 ms      | 0.08s    |

NUFROST per-pixel time reflects the O(N·M) direct DFT computation (N=200 obs,
M=4096 modes). HANTS and Zhu2015 are I/O-bound at this scale; kernel computation
is negligible.

### 4.2 Full Source Raster (1024×2 pixels, 200 bands)

| Algorithm | Wall time  | Per-pixel   | User CPU |
|-----------|------------|-------------|----------|
| HANTS     | 93.5s      | 45.6 ms     | 73.3s    |
| Zhu2015   | 90.9s      | 44.4 ms     | 74.4s    |
| NUFROST   | 98.6s      | 48.1 ms     | 104.4s   |

At this scale, all three algorithms are primarily I/O-bound (reading 200 GDAL
bands per pixel). NUFROST shows higher user CPU due to the NUFFT computation.

### 4.3 Per-Pixel Kernel Timing (no I/O)

Measured via intra-process Rust benchmark (100,000 iterations, synthetic 50-point
time series, no GDAL, no process boundary):

| Operation              | Time per op | Notes                              |
|------------------------|-------------|------------------------------------|
| `make_design_matrix`   | 0.65 µs     | 50×5 design matrix                 |
| `hants_pixel`          | 3.29 µs     | NOF=3, least-squares fit           |
| `zhu2015_fit_predict`  | 5.25 µs     | LASSO coordinate descent           |
| `nufrost_pixel`        | 1964 µs     | Direct DFT + frequency selection + IRLS |

NUFROST is ~600× slower per pixel than HANTS due to the direct DFT (O(50×4096) ≈
200K f64 operations). This is the expected cost of the NUFFT approach. The
spectrum computation could be accelerated via a FINUFFT FFI binding (noted as
future work in the source).

---

## 5. Build Timing

```
$ cargo build --release -p nufrost-cli     time: 0.40s (cached)
```

Full clean build from scratch (not measured) compiles ~150 dependencies including
gdal-sys bindings. Incremental release builds complete in under 1 second.

---

## 6. Deployment Documentation

### 6.1 System Dependencies

```bash
# GDAL and pkg-config (macOS via conda)
conda install -c conda-forge gdal pkg-config

# Verify GDAL installation
gdalinfo --version          # should show 3.10.x
pkg-config --libs gdal      # should print linker flags
```

### 6.2 Rust Toolchain

```bash
# Install stable Rust
rustup default stable

# The workspace has rust-toolchain.toml pinning 1.85.1
# Verify:
rustc --version             # should show 1.85.1
cargo --version
```

### 6.3 Build

```bash
# Set GDAL library path for macOS
export DYLD_LIBRARY_PATH=/opt/homebrew/Caskroom/miniforge/base/envs/geo-science/lib

# Build all crates in release mode
cd rust/
cargo build --release

# Binary at: target/release/nufrost-cli
```

### 6.4 Python Wrapper (nufrost-py)

```bash
# Install maturin
pip install maturin

# Build and install Python package
cd rust/nufrost-py/
maturin develop --release

# Verify
python -c "import nufrost_py; print(dir(nufrost_py))"
```

### 6.5 Runtime Environment

On macOS, the GDAL dynamic library path must be set at runtime:

```bash
# Option A: environment variable
export DYLD_LIBRARY_PATH=/opt/homebrew/Caskroom/miniforge/base/envs/geo-science/lib

# Option B: add rpath to binary (recommended for deployment)
install_name_tool -add_rpath \
  /opt/homebrew/Caskroom/miniforge/base/envs/geo-science/lib \
  target/release/nufrost-cli

# Option C: symlink GDAL libraries to a system path
# (not recommended — use A or B)
```

### 6.6 Verify Installation

```bash
# Run all tests
cargo test --workspace

# Expected: 116 passed, 0 failed

# Run CLI smoke test
nufrost-cli hants \
  --data tests/fixtures/rust_parity/synthetic/simple_harmonic/data.npz

# Expected: prediction value ~0.954
```

---

## 7. Notes

- **No Python speedup claim:** This report measures Rust performance only. Direct
  Python-vs-Rust comparison is not attempted because the two implementations run
  in different environments with different I/O paths.
- **NUFROST NUFFT:** The direct DFT implementation is O(N·M) where N is the number
  of observations and M is the number of spectral modes (default 4096). This is
  bottleneck for NUFROST's per-pixel time. A FINUFFT FFI binding would reduce this
  to O(M·log M).
- **I/O dominance:** In raster mode, GDAL band reading dominates wall time for
  HANTS and Zhu2015 at any meaningful scale (10+ pixels). NUFROST's compute cost
  becomes visible only at 100+ pixels.
- **CLI overhead:** The ~6.5ms process startup overhead makes the CLI unsuitable
  for per-pixel reconstruction. Use `nufrost-py` for direct Rust function calls
  from Python.
