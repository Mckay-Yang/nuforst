# NUFROST Project Skill

Use this skill when working on project structure, documentation, algorithm
boundaries, or explanations of the implemented NUFROST method.

## Repository Identity

This is a Rust-first remote-sensing reconstruction research repository. The
active code is under `crates/`.

Algorithm names:

- Write the method name as `NUFROST` in prose.
- Keep Rust crate identifiers lowercase/kebab-case:
  - `nufrost-core`
  - `hants-core`
  - `zhu2015-core`
  - `nufrost-cli`

Treat `NUFROST`, `HANTS`, and `Zhu2015` as separate algorithms. `HANTS` and
`Zhu2015` are baselines and should not be described as parts of NUFROST.

## Current Crate Boundaries

- `crates/gdal`: project raster I/O, GeoTIFF/VRT reading, timestamp parsing,
  full-scene helpers, scene cache, and sample cache. This crate may use external
  GDAL bindings, but it must not depend on algorithm crates.
- `crates/nufrost-core`: NUFROST algorithm, vector NUFFT, vector fitting,
  robust multi-output ridge, and prediction.
- `crates/hants-core`: HANTS baseline.
- `crates/zhu2015-core`: Zhu2015 baseline.
- `crates/nufrost-cli`: command-line orchestration only.

Dependency direction:

```text
gdal -> no algorithm dependencies
nufrost-core/hants-core/zhu2015-core -> may depend on gdal
nufrost-cli -> depends on gdal and all algorithm crates
```

## Active NUFROST Method Summary

For one pixel, model the observation as a vector-valued time series:

```math
\mathbf y(t_i)\in\mathbb R^B
```

Current Sentinel-2 band order is:

```text
B2, B3, B4, B8, B11, B12
```

The active path is:

1. Robustly standardize each band by median and MAD.
2. Compute vector-valued NUFFT over irregular timestamps.
3. Build a joint power spectrum by summing squared complex spectrum magnitudes
   across bands.
4. Select frequencies according to config. The current default is all positive
   modes plus high-frequency penalties and preferred phenology frequencies.
5. Build one shared harmonic/trend design matrix.
6. Fit all bands with date-level vector Huber IRLS and multi-output ridge.
7. Optionally perform joint outlier rejection and refit.
8. Apply multiband coefficient shrinkage when configured.
9. De-standardize the predicted vector.

Avoid describing the current method as six independent single-band fits.

## Documentation Rules

- Use GitHub-renderable fenced math blocks:

````markdown
```math
x = y + z
```
````

- Keep root README focused on the current Rust workflow, not removed Python
  parity scripts.
- If writing experiment notes, clearly separate current defaults from candidate
  research ideas.
