# NUFROST

NUFROST is now a Rust-first research codebase for reconstructing optical satellite image time series and comparing three reconstruction algorithms:

- `NUFROST`: non-uniform FFT based frequency discovery with robust fitting.
- `HANTS`: harmonic analysis of time series with iterative outlier rejection.
- `Zhu2015`: harmonic/LASSO baseline inspired by Zhu et al. (2015).

The previous Python scripts and Jupyter notebooks have been removed from the active workflow. Future plotting notebooks/scripts should be rebuilt around Rust outputs.

## Rust Workspace

```text
crates/
  gdal/            # GeoTIFF/VRT I/O, timestamp parsing, full-scene helpers
  nufrost-core/    # NUFROST algorithm
  hants-core/      # HANTS baseline
  zhu2015-core/    # Zhu2015 baseline
  cli/             # command-line entrypoint
  nufrost-py/      # optional PyO3 bindings
```

Dependency direction is intentionally simple:

```text
gdal

nufrost-core -> gdal
hants-core   -> gdal
zhu2015-core -> gdal

cli -> gdal
cli -> nufrost-core
cli -> hants-core
cli -> zhu2015-core
```

`gdal` is an I/O crate only. It must not depend on any algorithm crate.

## Configuration

Algorithm defaults are kept as JSON data:

```text
config/
  config.json
  nufrost.json
  hants.json
  zhu2015.json
```

## Build And Test

The workspace requires Rust 1.85+ and a system GDAL runtime.

```bash
cargo check --workspace
```

On macOS/conda, tests that link GDAL may need:

```bash
export DYLD_LIBRARY_PATH=/opt/homebrew/Caskroom/miniforge/base/envs/geo-science/lib
```

Useful test commands:

```bash
cargo test -p gdal@0.1.0 --lib
cargo test -p nufrost-core --lib
cargo test -p hants-core --lib
cargo test -p zhu2015-core --lib
cargo test -p cli --bin cli
```

The `gdal@0.1.0` package spec is used because the workspace crate is named `gdal` and it also depends on the upstream `gdal` crate through the alias `gdal-rs`.

## CLI

The CLI binary crate is `cli`:

```bash
cargo run -p cli -- nufrost --input-geotiff input.tif --output pred.tif
cargo run -p cli -- hants --input-geotiff input.tif --output pred.tif
cargo run -p cli -- zhu2015 --input-geotiff input.tif --output pred.tif
```

Full-scene reconstruction:

```bash
cargo run -p cli -- full-scene \
  --source-name sentinel-2 \
  --lon 94.2605 \
  --lat 29.7733 \
  --data-root data \
  --output-root data/output
```

## Fixtures

Rust parity fixtures live under `tests/fixtures/rust_parity/`. They are retained as deterministic test data even though the Python fixture generator has been removed.
