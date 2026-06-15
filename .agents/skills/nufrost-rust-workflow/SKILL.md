# NUFROST Rust Workflow Skill

Use this skill before modifying Rust code, running tests, changing cache logic,
or giving commands for this repository.

## Environment

The project requires Rust 1.85+ and an external GDAL runtime.

On this machine, GDAL-linked commands usually need:

```sh
export DYLD_LIBRARY_PATH=/opt/homebrew/Caskroom/miniforge/base/envs/geo-science/lib:$DYLD_LIBRARY_PATH
```

Root `data/` is a local symlink to external storage and is not tracked by git.
Committed test data belongs under `tests/data/`.

## Verification

Default compile check:

```sh
cargo check --workspace
```

Local project `gdal` crate test:

```sh
DYLD_LIBRARY_PATH=/opt/homebrew/Caskroom/miniforge/base/envs/geo-science/lib \
cargo test -p gdal@0.1.0 --lib
```

End-to-end smoke test using committed test data:

```sh
DYLD_LIBRARY_PATH=/opt/homebrew/Caskroom/miniforge/base/envs/geo-science/lib \
cargo test -p nufrost-cli full_scene_test_data_runs_end_to_end_with_auto_cache -- --nocapture
```

## Common Commands

Build release CLI:

```sh
cargo build --release -p nufrost-cli
```

Build scene cache:

```sh
DYLD_LIBRARY_PATH=/opt/homebrew/Caskroom/miniforge/base/envs/geo-science/lib \
target/release/nufrost-cli build-scene-cache \
  --source-name sentinel-2 \
  --lon <LON> \
  --lat <LAT> \
  --data-root data \
  --cache-root data/cache/scenes
```

Run one full scene:

```sh
DYLD_LIBRARY_PATH=/opt/homebrew/Caskroom/miniforge/base/envs/geo-science/lib \
target/release/nufrost-cli full-scene \
  --source-name sentinel-2 \
  --lon <LON> \
  --lat <LAT> \
  --methods nufrost \
  --data-root data \
  --output-root data/output \
  --n-jobs <N>
```

Build global sample cache:

```sh
DYLD_LIBRARY_PATH=/opt/homebrew/Caskroom/miniforge/base/envs/geo-science/lib \
target/release/nufrost-cli build-sample-cache \
  --source-name sentinel-2 \
  --scene-cache-root data/cache/scenes \
  --output data/cache/samples/sentinel-2_v1 \
  --n-samples 1000000 \
  --min-joint-valid 12 \
  --seed 20260608
```

Evaluate sample cache:

```sh
DYLD_LIBRARY_PATH=/opt/homebrew/Caskroom/miniforge/base/envs/geo-science/lib \
target/release/nufrost-cli eval-sample-cache \
  --method nufrost \
  --cache-dir data/cache/samples/sentinel-2_v1 \
  --n-eval 1000000 \
  --config config/nufrost.json
```

## Engineering Rules

- Keep algorithm changes in core crates, not the CLI.
- Keep full-scene and cache mechanics in `crates/gdal`.
- Use `rg` for source search.
- Do not remove external data under `data/` unless the user explicitly asks.
- Do not commit generated outputs unless they are small committed fixtures under
  `tests/data/`.
