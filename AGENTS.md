# NUFROST Agent Notes

## Required Orientation

Before changing code or documentation, read the relevant project skill under
`.agents/skills/`:

- `.agents/skills/nufrost-project/SKILL.md` for structure, naming, and method
  descriptions.
- `.agents/skills/nufrost-rust-workflow/SKILL.md` for Rust/GDAL/cache commands.
- `.agents/skills/nufrost-experiments/SKILL.md` for parameter experiments and
  evaluation conventions.

These files are part of the repository and should be kept current when project
workflow changes.

## Project-Specific Notes

- This is a Rust-first research repo. The active implementation lives under
  `crates/`.
- Treat `NUFROST`, `Zhu2015`, and `HANTS` as three separate reconstruction
  algorithms. `Zhu2015` and `HANTS` are comparison baselines, not sub-parts of
  NUFROST.
- In prose and docs, write the algorithm name as `NUFROST`; keep Rust crate
  identifiers lowercase/kebab-case (`nufrost-core`, `hants-core`,
  `zhu2015-core`, `nufrost-cli`).
- The active NUFROST path is vector-valued: vector NUFFT frequency discovery,
  date-level vector Huber IRLS, multi-output ridge, optional joint outlier
  rejection, and optional multiband coefficient shrinkage.
- Do not describe current NUFROST as six independent single-band fits.

## Rust Layout

- `crates/gdal`: GeoTIFF/VRT I/O, timestamp parsing, full-scene helpers, scene
  cache, sample cache, and generic raster traversal.
- `crates/nufrost-core`: NUFROST algorithm, vector NUFFT, robust fitting, and
  prediction logic.
- `crates/hants-core`: HANTS baseline algorithm.
- `crates/zhu2015-core`: Zhu2015 baseline algorithm.
- `crates/nufrost-cli`: command-line entrypoint and orchestration.

Dependency direction:

- `gdal` is independent and must not depend on algorithm crates.
- `nufrost-core`, `hants-core`, and `zhu2015-core` may depend on `gdal`.
- `nufrost-cli` depends on `gdal` and all three algorithm crates.

Keep algorithm logic out of `nufrost-cli` unless it is purely command-line
orchestration.

## Config And Data

- Runtime algorithm defaults are JSON files under `config/`.
- Root `data/` is local external data and is not tracked by git. On this
  machine it is usually a symlink to `/Volumes/T7/nufrost-data`.
- Small committed test imagery lives under `tests/data/`.
- Test-created `tests/data/cache/` and `tests/data/output/` are ignored.
- Real imagery, scene caches, sample caches, figures, and full outputs belong
  under root `data/`.

## Output And Verification

- Use `cargo check --workspace` for compile verification.
- Use `cargo test -p gdal@0.1.0 --lib` for the local `gdal` crate because the
  workspace also depends on upstream `gdal`.
- GDAL-linked tests may need:

```sh
DYLD_LIBRARY_PATH=/opt/homebrew/Caskroom/miniforge/base/envs/geo-science/lib
```

- The committed full-scene smoke test is:

```sh
DYLD_LIBRARY_PATH=/opt/homebrew/Caskroom/miniforge/base/envs/geo-science/lib \
cargo test -p nufrost-cli full_scene_test_data_runs_end_to_end_with_auto_cache -- --nocapture
```

- Python scripts, notebooks, and pytest tests are not part of the active Rust
  workflow. Notebooks may still be used for Earth Engine data acquisition or
  exploratory plotting when explicitly requested.

## Git Expectations

- Preserve user changes. Do not revert notebook or data edits unless explicitly
  asked.
- After committing on `develop`, push `develop` to `origin` unless the user says
  not to.
- Do not commit generated data/output by default. Commit only source, config,
  docs, small fixtures, and intentional agent instructions.
