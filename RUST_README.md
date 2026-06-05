# Rust Workspace — NUFROST Rewrite

This workspace bootstraps a Rust rewrite of the reconstruction pipeline.
It contains six crates under `crates/`, using the standard Cargo workspace
layout where each crate keeps its Rust sources under its own `src/` directory:

| Crate            | Type        | Purpose                                      |
|------------------|-------------|----------------------------------------------|
| `nufrost-core`   | lib         | NUFROST algorithm, NUFFT, shared time/types   |
| `hants`          | lib         | HANTS baseline algorithm                      |
| `zhu2015`        | lib         | Zhu2015 baseline algorithm                    |
| `nufrost-gdal`   | lib         | Raster I/O via GDAL (requires system libgdal) |
| `nufrost-cli`    | binary      | CLI entrypoint for reconstruction            |
| `nufrost-py`     | cdylib      | Python bindings via PyO3 + maturin           |

## System Dependencies

### GDAL

`nufrost-gdal` depends on the system GDAL library (`libgdal`).
You must have GDAL installed before building the workspace.

**macOS (Homebrew):**
```sh
brew install gdal
```

**Conda:**
```sh
conda install -c conda-forge gdal
```

**Verify:**
```sh
gdalinfo --version
```

The crate links against GDAL via the `gdal` Rust crate (FFI bindings).

**Runtime library path:** On macOS with conda, set `DYLD_LIBRARY_PATH` so the dynamic linker can find `libgdal`:

```sh
export DYLD_LIBRARY_PATH="$CONDA_PREFIX/lib:$DYLD_LIBRARY_PATH"
```

**Build-time environment variables** (needed when `pkg-config` is unavailable):

```sh
export GDAL_VERSION=3.10.3
export GDAL_INCLUDE_DIR="$CONDA_PREFIX/include"
export GDAL_LIB_DIR="$CONDA_PREFIX/lib"
```

### Rust Toolchain

Rust 1.85+ is required. The toolchain is pinned via `rust-toolchain.toml`.

## Quick Start

```sh
# Verify workspace integrity
cargo metadata --format-version 1

# Build all crates
cargo build --workspace

# Run all tests
cargo test --workspace
```
