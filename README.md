# NUFROST

NUFROST is a Rust-first research workspace for optical remote-sensing time-series
reconstruction from irregular satellite observations. The active implementation
lives under `crates/` and focuses on Sentinel-2 style multi-band trajectories.

The repository contains three reconstruction algorithms:

- `NUFROST`: the research method developed in this project.
- `HANTS`: a harmonic-analysis baseline.
- `Zhu2015`: a Landsat-style harmonic/LASSO baseline.

`HANTS` and `Zhu2015` are comparison baselines. They are not submodules or
variants of NUFROST.

## Current Method

The active NUFROST path is a vector-valued reconstruction model. For one pixel,
the observation at time `t_i` is a multi-band vector:

```math
\mathbf y_i =
\begin{bmatrix}
y_{i,1} & y_{i,2} & \cdots & y_{i,B}
\end{bmatrix}
\in \mathbb R^B .
```

For the Sentinel-2 workflow, the current band order is:

```text
B2, B3, B4, B8, B11, B12
```

so usually `B = 6`.

NUFROST models each pixel as a vector-valued curve:

```math
\mathbf y(t): \mathbb R \rightarrow \mathbb R^B
```

and predicts a held-out or missing target vector:

```math
\hat{\mathbf y}(t_\star).
```

At a high level, the current model is:

```math
\mathbf y(t)
\approx
\boldsymbol\mu
+
\mathbf s \odot
\left(
\mathbf x(t)^\top \Theta
\right),
```

where:

- `mu in R^B` is the per-band robust center.
- `s in R^B` is the per-band robust scale.
- `x(t) in R^P` is the shared harmonic/trend basis.
- `Theta in R^(P x B)` is the multi-output coefficient matrix.
- `odot` is elementwise multiplication.

### Robust Standardization

Each band is standardized before frequency discovery and fitting:

```math
\mu_b = \operatorname{median}_i(y_{i,b})
```

```math
s_b =
1.4826
\operatorname{median}_i
\left|y_{i,b}-\mu_b\right|
```

```math
Z_{i,b} =
\frac{y_{i,b}-\mu_b}{s_b+\epsilon}.
```

This keeps the robust loss and penalties in a common standardized space without
forcing the raw reflectance levels of different bands to be equal. If the MAD is
degenerate, the implementation falls back to a standard-deviation scale and a
small positive floor.

### Vector NUFFT

Observation dates are irregular because cloud, snow, shadow, orbit, and quality
filters remove many acquisitions. NUFROST estimates a spectrum directly from
non-uniform time samples.

For band `b`:

```math
F_b(f_k)
=
\sum_{i=1}^{T}
z_b(t_i)
\exp(-\mathrm{i}\,2\pi f_k t_i).
```

The implementation computes a vector-valued NUFFT on the shared timestamp grid:

```math
\mathbf F(f_k)
=
\begin{bmatrix}
F_1(f_k)\\
F_2(f_k)\\
\vdots\\
F_B(f_k)
\end{bmatrix}
\in \mathbb C^B.
```

The Rust NUFFT implementation is in `crates/nufrost-core/src/nufft.rs`. It uses:

- Kaiser-Bessel spreading onto an oversampled uniform grid.
- A radix-2 FFT on that grid.
- Deconvolution by the periodized kernel spectrum.
- A joint vector power spectrum:

```math
P(f_k)
=
\|\mathbf F(f_k)\|_2^2
=
\sum_{b=1}^{B} |F_b(f_k)|^2 .
```

All bands share the same non-uniform time coordinates, spreading positions, FFT
grid, deconvolution, and output frequency bins.

### Frequency Set

The current default NUFROST config uses:

```json
"frequency_selection": "all",
"preferred_periods_days": "365.25,182.625,91.3125,30.4375",
"preferred_top_k": 4,
"spectral_top_k": 8,
"modes": 64
```

`all` keeps the available positive NUFFT frequency modes and relies on strong
frequency-weighted ridge penalties to suppress unstable high frequencies.
Preferred phenology frequencies are treated as low-penalty frequencies.

Other supported modes in code include `spectral`, `preferred`, and
`hybrid`/`shared_spectral`.

### Design Matrix

Given selected frequencies:

```math
\mathcal F = \{f_1,\dots,f_m\},
```

NUFROST builds one shared design matrix:

```math
X \in \mathbb R^{T \times P}.
```

With intercept and trend enabled:

```math
\mathbf x(t_i)
=
\begin{bmatrix}
1,\,
t_i-\bar t,\,
\cos(2\pi f_1t_i),\,
\sin(2\pi f_1t_i),\,
\dots,\,
\cos(2\pi f_mt_i),\,
\sin(2\pi f_mt_i)
\end{bmatrix}.
```

The number of columns is:

```math
P = 1 + \mathbf 1_{\mathrm{trend}} + 2m.
```

### Multi-Output Ridge

The standardized reconstruction model is:

```math
Z \approx X\Theta,
```

where:

```math
Z\in\mathbb R^{T\times B},\quad
X\in\mathbb R^{T\times P},\quad
\Theta\in\mathbb R^{P\times B}.
```

For fixed date weights, the core objective is:

```math
\min_\Theta
\frac{1}{2}
\left\|
W^{1/2}(Z-X\Theta)
\right\|_F^2
+
\frac{1}{2}
\left\|
\Lambda^{1/2}\Theta
\right\|_F^2 .
```

The normal equation is:

```math
\left(
X^\top W X+\Lambda
\right)\Theta
=
X^\top WZ.
```

The Rust solver builds the left-hand matrix once and solves all band right-hand
sides together. This is not six independent IRLS fits.

### Date-Level Vector Huber IRLS

NUFROST uses a date-level robust residual. At date `i`:

```math
\mathbf r_i = \mathbf z_i - \mathbf x_i^\top\Theta.
```

The residual magnitude is the RMS vector residual:

```math
e_i =
\sqrt{
\frac{1}{B}
\sum_{b=1}^{B} r_{i,b}^2
}.
```

Huber weights are updated as:

```math
w_i =
\begin{cases}
1, & e_i \leq \delta,\\
\delta/(e_i+\epsilon), & e_i > \delta.
\end{cases}
```

The current implementation damps the weight update and keeps a small minimum
weight to avoid unstable systems when few dates remain. After Huber fitting,
optional joint outlier rejection can remove high-residual dates and refit.

### Frequency-Weighted Ridge

High-frequency coefficients receive stronger penalties. Preferred phenology
frequencies can be exempted from the high-frequency penalty. The default config
currently uses:

```json
"ridge": 2.0,
"freq_weight": 256.0,
"lambda_high": 0.005,
"low_freq_period_days": 60.0
```

This makes the full-frequency model possible while strongly discouraging noisy
high-frequency oscillations.

### Multiband Coefficient Shrinkage

The current default config enables:

```json
"multiband_shrinkage": 1.0
```

The solver decomposes each coefficient row into an across-band mean component
and a band-specific contrast component:

```math
\Theta_{j,b}
=
\bar{\Theta}_j + \Delta_{j,b}.
```

The shrinkage term penalizes the contrast component:

```math
\lambda_s
\sum_{j,b}
\left(
\Theta_{j,b}-\bar{\Theta}_j
\right)^2 .
```

This does not force raw bands to have the same reflectance. It only discourages
unnecessary divergence in standardized temporal coefficient structure.

## Workspace Layout

```text
.
├── Cargo.toml
├── AGENTS.md
├── README.md
├── config/
├── crates/
│   ├── gdal/
│   ├── nufrost-core/
│   ├── hants-core/
│   ├── zhu2015-core/
│   └── nufrost-cli/
├── tests/data/
├── data -> /Volumes/T7/nufrost-data
└── .agents/
```

Crates:

| Crate | Role |
|---|---|
| `gdal` | Project raster I/O, timestamp parsing, scene cache, sample cache, full-scene helpers. Uses external `libgdal` through Rust GDAL bindings. |
| `nufrost-core` | NUFROST algorithm, NUFFT, vector fitting, prediction, and tests. |
| `hants-core` | HANTS comparison baseline. |
| `zhu2015-core` | Zhu2015 comparison baseline. |
| `nufrost-cli` | CLI entrypoint for single-pixel, full-scene, cache, and sample-cache workflows. |

Dependency direction:

```text
gdal
  ↑
  ├── nufrost-core
  ├── hants-core
  └── zhu2015-core

gdal + nufrost-core + hants-core + zhu2015-core
  ↑
  └── nufrost-cli
```

`gdal` must remain independent of algorithm crates.

## Data Layout

Root `data/` is intentionally not tracked by git. On this machine it is a
symlink to:

```text
/Volumes/T7/nufrost-data
```

Real imagery, generated reconstructions, figures, scene caches, and sample
caches should stay under `data/`.

Small committed test fixtures live under:

```text
tests/data/
```

`tests/data/cache/` and `tests/data/output/` are ignored because tests may
create them during execution.

## GDAL Runtime

The project uses system GDAL through the Rust `gdal`/`gdal-sys` bindings. On
this machine, GDAL-linked commands usually need:

```sh
export DYLD_LIBRARY_PATH=/opt/homebrew/Caskroom/miniforge/base/envs/geo-science/lib:$DYLD_LIBRARY_PATH
```

Long-term, GDAL should remain an import/export boundary. Hot loops should use
the internal scene cache and sample cache formats rather than repeatedly reading
GeoTIFF stacks.

## Build And Verify

```sh
cargo check --workspace
```

Run the local project `gdal` crate tests explicitly because the workspace also
depends on the upstream Rust crate named `gdal`:

```sh
DYLD_LIBRARY_PATH=/opt/homebrew/Caskroom/miniforge/base/envs/geo-science/lib \
cargo test -p gdal@0.1.0 --lib
```

Run the full-scene test-data smoke test:

```sh
DYLD_LIBRARY_PATH=/opt/homebrew/Caskroom/miniforge/base/envs/geo-science/lib \
cargo test -p nufrost-cli full_scene_test_data_runs_end_to_end_with_auto_cache -- --nocapture
```

## Common CLI Workflows

Build a release binary:

```sh
cargo build --release -p nufrost-cli
```

Build or refresh one scene cache:

```sh
DYLD_LIBRARY_PATH=/opt/homebrew/Caskroom/miniforge/base/envs/geo-science/lib \
target/release/nufrost-cli build-scene-cache \
  --source-name sentinel-2 \
  --lon 94.2605 \
  --lat 29.7733 \
  --data-root data \
  --cache-root data/cache/scenes
```

Run one full scene:

```sh
DYLD_LIBRARY_PATH=/opt/homebrew/Caskroom/miniforge/base/envs/geo-science/lib \
target/release/nufrost-cli full-scene \
  --source-name sentinel-2 \
  --lon 94.2605 \
  --lat 29.7733 \
  --methods nufrost \
  --data-root data \
  --output-root data/output \
  --n-jobs 8
```

Run all available scenes:

```sh
DYLD_LIBRARY_PATH=/opt/homebrew/Caskroom/miniforge/base/envs/geo-science/lib \
target/release/nufrost-cli batch-full-scene \
  --source-name sentinel-2 \
  --methods nufrost \
  --data-root data \
  --output-root data/output \
  --n-jobs 8 \
  --continue-on-error
```

Build a global sample cache:

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

Evaluate a method on the sample cache:

```sh
DYLD_LIBRARY_PATH=/opt/homebrew/Caskroom/miniforge/base/envs/geo-science/lib \
target/release/nufrost-cli eval-sample-cache \
  --method nufrost \
  --cache-dir data/cache/samples/sentinel-2_v1 \
  --n-eval 1000000 \
  --config config/nufrost.json \
  --output-json data/output/sample_cache_predictions/nufrost_holdout_1000000.json \
  --output-prediction-csv data/output/sample_cache_predictions/nufrost_holdout_1000000_predictions.csv
```

## Notes For Future Work

- Keep NUFROST method changes in `nufrost-core`; do not put algorithm logic in
  `nufrost-cli`.
- Keep raster/cache mechanics in `gdal`; do not make `gdal` depend on algorithm
  crates.
- Keep root `data/` untracked. Store committed smoke-test data only in
  `tests/data/`.
- Prefer sample-cache evaluation for global parameter decisions. Single-region
  full-scene tests are useful for inspection but can be extreme cases.
- Preserve `HANTS` and `Zhu2015` as baselines unless explicitly told otherwise.
