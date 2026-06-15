# NUFROST Agent Notes

## Project-Specific Notes

- This is now a Rust-first research repo. The active implementation lives under `crates/`.
- Treat `NUFROST`, `Zhu2015`, and `HANTS` as three separate reconstruction algorithms. `Zhu2015` and `HANTS` are comparison baselines, not sub-parts of NUFROST.
- In prose and docs, write the algorithm name as `NUFROST`; keep Rust crate identifiers lowercase/kebab-case (`nufrost-core`, `hants-core`, `zhu2015-core`).

## Rust Layout

- `crates/gdal`: GeoTIFF/VRT I/O, timestamp parsing, full-scene helpers, generic raster traversal.
- `crates/nufrost-core`: NUFROST algorithm and NUFFT/fitting logic.
- `crates/hants-core`: HANTS baseline algorithm.
- `crates/zhu2015-core`: Zhu2015 baseline algorithm.
- `crates/nufrost-cli`: command-line entrypoint.

Dependency direction:

- `gdal` is independent and must not depend on algorithm crates.
- `nufrost-core`, `hants-core`, and `zhu2015-core` may depend on `gdal`.
- `nufrost-cli` depends on `gdal` and all three algorithm crates.

## Config And Data

- Runtime algorithm defaults are JSON files under `config/`.
- Rust parity fixtures live under `tests/fixtures/rust_parity/`.
- Real imagery and generated outputs live under `data/`.

## Output And Verification

- Use `cargo check --workspace` for compile verification.
- Use `cargo test -p gdal@0.1.0 --lib` for the local `gdal` crate because the workspace also depends on upstream `gdal`.
- GDAL-linked tests may need `DYLD_LIBRARY_PATH=/opt/homebrew/Caskroom/miniforge/base/envs/geo-science/lib` on this machine.
- Python scripts, notebooks, and pytest tests have been removed from the active workflow. Future plotting scripts should be rebuilt around Rust outputs.
