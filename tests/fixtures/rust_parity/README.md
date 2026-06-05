# Rust Parity Fixtures

These fixtures provide deterministic inputs and expected outputs for Rust tests covering NUFROST, HANTS, and Zhu2015.

The previous Python generator has been removed from the active workflow. Treat these files as checked-in reference data unless a new Rust fixture generator is added.

## Fixtures

### `synthetic/simple_harmonic`

- Clean harmonic signal with no gaps or outliers.
- Used as a happy-path unit test for all three algorithms.

### `synthetic/gaps_outliers`

- Harmonic signal with gaps and outliers.
- Used to validate gap handling and outlier robustness.

### `synthetic/step_break`

- Two-segment series with a structural break.
- Used to test break/step behavior and baseline robustness.

### `real/small_window`

- Small spatial window from real Sentinel-2 B2 data.
- Used for raster-level integration tests.

## File Format

Synthetic fixture directories contain:

- `data.npz`
- `data.json`
- `config.json`

The real window fixture directory contains:

- `inputs.npy`
- `timestamps.npz`
- `<prefix>_<algo>_pred.npy`
- `<prefix>_zhu2015_qa.npy`
- `config.json`
- `info.json`

`manifest.json` lists fixture paths and tolerance reference values.
