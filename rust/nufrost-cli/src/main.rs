// nufrost-cli — command-line entrypoint for NUFROST, HANTS, and Zhu2015
// reconstruction algorithms.  Supports single-pixel fixture NPZ input and
// raster GeoTIFF input via nufrost-gdal.

use std::collections::BTreeMap;
use std::fs;
use std::path::PathBuf;

use anyhow::{bail, Context, Result};
use clap::{Parser, Subcommand};
use ndarray::Array3;
use rayon::prelude::*;

use nufrost_core::{
    hants_pixel, nufrost_pixel, nufrost_pixel_with_shared, zhu2015::fit_predict_pixel,
    parse_iso8601_to_epoch_seconds, to_seconds_since_start, HantsConfig, NufrostConfig,
    Zhu2015Config,
};
use nufrost_gdal::{
    full_scene::{
        self, build_ground_truth_output_path, build_scene_stack_output_path,
        build_shared_frequency_pool, choose_shared_target_timestamp,
        discover_sentinel_band_stacks, make_masked_time_series, mask_invalid_sentinel2,
        sorted_band_names, write_band_stack,
    },
    extract_raw_band_descriptions, read_all_bands, read_all_bands_window,
    RasterMetadata, RasterReader,
    reconstruct_nufrost_geotiff, reconstruct_hants_geotiff, reconstruct_zhu2015_geotiff,
    synthetic_timestamps_from_bands,
};

// ═══════════════════════════════════════════════════════════════════════════
//  CLI definition (clap derive)
// ═══════════════════════════════════════════════════════════════════════════

/// NUFROST time-series reconstruction CLI.
///
/// Runs one of three algorithms on input time-series data.
/// Supports single-pixel NPZ fixtures and raster GeoTIFF input.
#[derive(Parser, Debug)]
#[command(
    name = "nufrost-cli",
    version = env!("CARGO_PKG_VERSION"),
    about = "NUFROST / HANTS / Zhu2015 time-series reconstruction CLI",
    long_about = None,
    after_help = "Examples:\n  \
                  nufrost-cli nufrost --data fixture.npz --target-time 372.7\n  \
                  nufrost-cli hants --config hants.json --data fixture.npz\n  \
                  nufrost-cli zhu2015 --data fixture.npz -t 372.7 -o pred.txt\n  \
                  nufrost-cli nufrost --input-geotiff input.tif --output pred.tif\n  \
                  nufrost-cli hants --input-geotiff input.tif -o pred.tif\n  \
                  nufrost-cli zhu2015 --input-geotiff input.tif -o pred.tif",
)]
struct Cli {
    #[command(subcommand)]
    algorithm: Algorithm,
}

/// Available reconstruction algorithms.
#[derive(Subcommand, Debug)]
enum Algorithm {
    /// Run NUFROST (Non-Uniform FFT-based) reconstruction.
    ///
    /// Uses NUFFT frequency discovery + Huber-ridge IRLS fitting.
    Nufrost(NufrostArgs),

    /// Run HANTS (Harmonic ANalysis of Time Series) reconstruction.
    ///
    /// Iterative harmonic fitting with outlier rejection.
    Hants(HantsArgs),

    /// Run Zhu2015 (Lasso-based synthetic Landsat) reconstruction.
    ///
    /// Piecewise harmonic fitting with L1-regularised LASSO.
    #[command(name = "zhu2015")]
    Zhu2015(Zhu2015Args),

    /// Run full-scene reconstruction for a location with all three algorithms.
    ///
    /// Discovers band stacks, selects a shared target timestamp, and runs
    /// NUFROST / HANTS / Zhu2015 per-band in parallel.  Writes merged
    /// multi-band prediction stacks, ground truth, and a summary JSON.
    #[command(name = "full-scene")]
    FullScene(FullSceneArgs),
}

/// Full-scene reconstruction args — runs all three algorithms for one
/// (lon, lat) location, discovering band stacks and writing merged outputs
/// in the Python-compatible directory layout.
#[derive(clap::Args, Debug)]
struct FullSceneArgs {
    /// Source name (sentinel-2 or hls).
    #[arg(long)]
    source_name: String,

    /// Longitude.
    #[arg(long)]
    lon: f64,

    /// Latitude.
    #[arg(long)]
    lat: f64,

    /// Output root directory.
    #[arg(long, default_value = "data/output")]
    output_root: PathBuf,

    /// Data root directory.
    #[arg(long, default_value = "data")]
    data_root: PathBuf,

    /// Comma-separated list of methods to run.
    #[arg(long, default_value = "nufrost,hants,zhu2015")]
    methods: String,

    /// Number of threads.
    #[arg(long)]
    n_jobs: Option<usize>,

    /// Optional crop window size in pixels.
    #[arg(long)]
    window_size: Option<usize>,

    /// Minimum valid ratio for target selection.
    #[arg(long, default_value = "0.9")]
    min_valid_ratio: f64,

    /// Late fraction for target selection.
    #[arg(long, default_value = "0.25")]
    late_fraction: f64,

    /// Frequency selection mode override for NUFROST config.
    #[arg(long)]
    frequency_selection: Option<String>,

    /// Spectral top-k override for NUFROST config.
    #[arg(long)]
    spectral_top_k: Option<usize>,

    /// Preferred top-k override for NUFROST config.
    #[arg(long)]
    preferred_top_k: Option<usize>,
}

#[derive(clap::Args, Debug)]
struct NufrostArgs {
    /// Path to NUFROST JSON config file.
    ///
    /// If omitted, uses built-in defaults matching Python config/nufrost.json.
    #[arg(short, long)]
    config: Option<PathBuf>,

    #[clap(flatten)]
    shared: SharedArgs,
}

#[derive(clap::Args, Debug)]
struct HantsArgs {
    /// Path to HANTS JSON config file.
    ///
    /// If omitted, uses built-in defaults matching Python config/hants.json.
    #[arg(short, long)]
    config: Option<PathBuf>,

    #[clap(flatten)]
    shared: SharedArgs,
}

#[derive(clap::Args, Debug)]
struct Zhu2015Args {
    /// Path to Zhu2015 JSON config file.
    ///
    /// If omitted, uses built-in defaults matching Python config/zhu2015.json.
    #[arg(short, long)]
    config: Option<PathBuf>,

    #[clap(flatten)]
    shared: SharedArgs,
}

/// Shared args for all three algorithm arg structs.
#[derive(clap::Args, Debug)]
struct SharedArgs {
    /// Path to NPZ fixture file (single-pixel mode).
    ///
    /// Expected keys: `timestamps_days`, `observations`, `target_time_day`.
    /// Mutually exclusive with `--input-geotiff`.
    #[arg(short, long)]
    data: Option<PathBuf>,

    /// Path to input GeoTIFF (raster reconstruction mode).
    ///
    /// Multi-band GeoTIFF where each band is a timestamp.
    /// Mutually exclusive with `--data`.
    #[arg(long = "input-geotiff")]
    input_geotiff: Option<PathBuf>,

    /// Target time in days since first observation.
    ///
    /// Overrides the `target_time_day` value embedded in a fixture NPZ,
    /// or the auto-detected last-band timestamp in GeoTIFF mode.
    #[arg(short = 't', long)]
    target_time: Option<f64>,

    /// Output file path.
    ///
    /// In NPZ mode: writes scalar prediction as text.
    /// In GeoTIFF mode: writes a single-band output GeoTIFF.
    #[arg(short, long)]
    output: Option<PathBuf>,

    /// Number of threads for parallel raster processing.
    #[arg(long, default_value = "1")]
    threads: usize,
}

// ═══════════════════════════════════════════════════════════════════════════
//  Fixture loading (single-pixel NPZ mode)
// ═══════════════════════════════════════════════════════════════════════════

/// Raw data extracted from an NPZ fixture.
#[derive(Debug)]
struct FixtureData {
    timestamps_days: Vec<f64>,
    observations: Vec<f64>,
    target_time_day: f64,
}

/// Load a single-pixel NPZ fixture.
fn load_fixture_npz(path: &std::path::Path) -> Result<FixtureData> {
    use ndarray::Array1;

    let file = fs::File::open(path)
        .with_context(|| format!("Cannot open fixture: {}", path.display()))?;
    let mut npz = ndarray_npy::NpzReader::new(file)
        .with_context(|| format!("Cannot parse NPZ: {}", path.display()))?;

    let timestamps_days: Array1<f64> = npz
        .by_name("timestamps_days.npy")
        .context("Key 'timestamps_days' not found in fixture NPZ")?;
    let observations: Array1<f64> = npz
        .by_name("observations.npy")
        .context("Key 'observations' not found in fixture NPZ")?;
    let target_time_day_arr: ndarray::Array0<f64> = npz
        .by_name("target_time_day.npy")
        .context("Key 'target_time_day' not found in fixture NPZ")?;

    if timestamps_days.len() != observations.len() {
        bail!(
            "timestamps_days (len={}) and observations (len={}) must have the same length",
            timestamps_days.len(),
            observations.len()
        );
    }

    Ok(FixtureData {
        timestamps_days: timestamps_days.to_vec(),
        observations: observations.to_vec(),
        target_time_day: target_time_day_arr[()],
    })
}

// ═══════════════════════════════════════════════════════════════════════════
//  Config loading
// ═══════════════════════════════════════════════════════════════════════════

/// Default NUFROST config matching Python `config/nufrost.json`.
fn default_nufrost_config() -> NufrostConfig {
    let full_json = r#"{
        "nufrost":{"modes":4096,"eps":1e-12,"num_peaks":10,"power_cum":0.7,"ignore_dc_hz":1e-10,"refine_peaks":true,"include_trend":true,"ridge_lam":0.005,"freq_weight":2.0,"huber_iters":3,"huber_delta":0.05,"min_obs":12},
        "hants":{"nof":3,"sf":"high","fet":500.0,"dod":5,"period":365.25,"valid_min":null,"valid_max":null},
        "zhu2015":{"lasso_alpha":0.1}
    }"#;
    let rc: nufrost_core::ReconstructionConfig = serde_json::from_str(full_json).unwrap();
    rc.nufrost
}

/// Default HANTS config matching Python `config/hants.json`.
fn default_hants_config() -> HantsConfig {
    serde_json::from_str(
        r#"{"nof":3,"sf":"high","fet":500.0,"dod":5,"period":365.25,"valid_min":null,"valid_max":null}"#,
    )
    .expect("hardcoded default HANTS config must be valid")
}

/// Default Zhu2015 config matching Python `config/zhu2015.json`.
fn default_zhu2015_config() -> Zhu2015Config {
    serde_json::from_str(r#"{"lasso_alpha":0.1}"#).expect("hardcoded default Zhu2015 config must be valid")
}

fn load_nufrost_config(path: Option<&std::path::Path>) -> Result<NufrostConfig> {
    match path {
        Some(p) => {
            let bytes = fs::read(p)
                .with_context(|| format!("Cannot read config: {}", p.display()))?;
            serde_json::from_slice::<nufrost_core::ReconstructionConfig>(&bytes)
                .map(|rc| rc.nufrost)
                .or_else(|_| NufrostConfig::from_json(&bytes))
                .with_context(|| format!("Invalid NUFROST config: {}", p.display()))
        }
        None => Ok(default_nufrost_config()),
    }
}

fn load_hants_config(path: Option<&std::path::Path>) -> Result<HantsConfig> {
    match path {
        Some(p) => {
            let bytes = fs::read(p)
                .with_context(|| format!("Cannot read config: {}", p.display()))?;
            serde_json::from_slice::<nufrost_core::ReconstructionConfig>(&bytes)
                .map(|rc| rc.hants)
                .or_else(|_| HantsConfig::from_json(&bytes))
                .with_context(|| format!("Invalid HANTS config: {}", p.display()))
        }
        None => Ok(default_hants_config()),
    }
}

fn load_zhu2015_config(path: Option<&std::path::Path>) -> Result<Zhu2015Config> {
    match path {
        Some(p) => {
            let bytes = fs::read(p)
                .with_context(|| format!("Cannot read config: {}", p.display()))?;
            serde_json::from_slice::<nufrost_core::ReconstructionConfig>(&bytes)
                .map(|rc| rc.zhu2015)
                .or_else(|_| Zhu2015Config::from_json(&bytes))
                .with_context(|| format!("Invalid Zhu2015 config: {}", p.display()))
        }
        None => Ok(default_zhu2015_config()),
    }
}

// ═══════════════════════════════════════════════════════════════════════════
//  Input mode detection
// ═══════════════════════════════════════════════════════════════════════════

enum InputMode {
    /// Single-pixel NPZ fixture.
    NpzFixture(FixtureData),
    /// Raster GeoTIFF with per-pixel reconstruction.
    GeoTiff { reader: RasterReader, output: PathBuf },
}

fn detect_input_mode(shared: &SharedArgs) -> Result<InputMode> {
    match (&shared.data, &shared.input_geotiff) {
        (Some(npz_path), None) => {
            let fixture = load_fixture_npz(npz_path)?;
            Ok(InputMode::NpzFixture(fixture))
        }
        (None, Some(tif_path)) => {
            let reader = RasterReader::open(tif_path)
                .with_context(|| format!("Cannot open GeoTIFF: {}", tif_path.display()))?;
            if reader.band_count() == 0 {
                bail!("GeoTIFF has no bands");
            }
            let (rows, cols) = reader.shape();
            if rows == 0 || cols == 0 {
                bail!("GeoTIFF has zero spatial dimensions ({rows}r × {cols}c)");
            }
            let output = shared.output.clone().ok_or_else(|| {
                anyhow::anyhow!("--output <PATH> is required in GeoTIFF mode")
            })?;
            Ok(InputMode::GeoTiff { reader, output })
        }
        (Some(_), Some(_)) => {
            bail!("--data and --input-geotiff are mutually exclusive")
        }
        (None, None) => {
            bail!(
                "No input data provided.  Use --data <fixture.npz> for single-pixel mode \
                 or --input-geotiff <input.tif> for raster mode."
            )
        }
    }
}

/// Resolve target time: explicit CLI arg beats fixture-embedded value.
fn resolve_target_time(fixture: &FixtureData, cli_target: Option<f64>) -> f64 {
    cli_target.unwrap_or(fixture.target_time_day)
}

/// Write or print a single scalar result (NPZ mode).
fn output_result(value: f64, output_path: Option<&PathBuf>, label: &str) -> Result<()> {
    let text = if label.is_empty() {
        format!("{value}\n")
    } else {
        format!("{label}: {value}\n")
    };
    match output_path {
        Some(p) => {
            fs::write(p, &text)
                .with_context(|| format!("Cannot write output: {}", p.display()))?;
        }
        None => {
            print!("{text}");
        }
    }
    Ok(())
}

// ── Metadata extraction from reader ──────────────────────────────────────

fn metadata_from_reader(reader: &RasterReader) -> RasterMetadata {
    RasterMetadata {
        geo_transform: reader.geo_transform().unwrap_or([0.0, 1.0, 0.0, 0.0, 0.0, -1.0]),
        crs_wkt: reader.crs_wkt(),
        nodata: reader.nodata(1),
    }
}

// ═══════════════════════════════════════════════════════════════════════════
//  NPZ mode — single-pixel reconstruction
// ═══════════════════════════════════════════════════════════════════════════

fn run_nufrost_npz(args: &NufrostArgs, fixture: &FixtureData) -> Result<()> {
    let config = load_nufrost_config(args.config.as_deref())?;
    let target_t = resolve_target_time(fixture, args.shared.target_time);

    let (pred, n_freqs) = nufrost_pixel(
        &fixture.timestamps_days,
        &fixture.observations,
        target_t,
        &config,
    );

    if !pred.is_finite() {
        bail!("NUFROST prediction is non-finite (NaN or Inf)");
    }

    output_result(
        pred,
        args.shared.output.as_ref(),
        &format!("nufrost_prediction (n_freqs={n_freqs})"),
    )?;

    eprintln!("NUFROST completed: pred={pred:.6}, n_freqs={n_freqs}");
    Ok(())
}

fn run_hants_npz(args: &HantsArgs, fixture: &FixtureData) -> Result<()> {
    let config = load_hants_config(args.config.as_deref())?;
    let target_t = resolve_target_time(fixture, args.shared.target_time);

    let pred = hants_pixel(
        &fixture.timestamps_days,
        &fixture.observations,
        target_t,
        config.nof,
        &config.sf,
        config.valid_min,
        config.valid_max,
        config.fet,
        config.dod,
        config.period,
    );

    if !pred.is_finite() {
        bail!("HANTS prediction is non-finite (NaN or Inf)");
    }

    output_result(pred, args.shared.output.as_ref(), "hants_prediction")?;
    eprintln!("HANTS completed: pred={pred:.6}");
    Ok(())
}

fn run_zhu2015_npz(args: &Zhu2015Args, fixture: &FixtureData) -> Result<()> {
    let config = load_zhu2015_config(args.config.as_deref())?;
    let target_t = resolve_target_time(fixture, args.shared.target_time);

    let result = fit_predict_pixel(
        &fixture.timestamps_days,
        &fixture.observations,
        target_t,
        config.lasso_alpha,
    );

    if !result.prediction.is_finite() {
        bail!("Zhu2015 prediction is non-finite (NaN or Inf)");
    }

    output_result(
        result.prediction,
        args.shared.output.as_ref(),
        "zhu2015_prediction",
    )?;

    eprintln!(
        "Zhu2015 completed: pred={:.6}",
        result.prediction
    );
    Ok(())
}

// ═══════════════════════════════════════════════════════════════════════════
//  GeoTIFF mode — full raster reconstruction
// ═══════════════════════════════════════════════════════════════════════════

fn run_nufrost_geotiff(
    args: &NufrostArgs,
    reader: &RasterReader,
    output: &PathBuf,
) -> Result<()> {
    let config = load_nufrost_config(args.config.as_deref())?;
    let meta = metadata_from_reader(reader);
    let (timestamps_days, target_t) = synthetic_timestamps_from_bands(reader.band_count());
    let target_t = args.shared.target_time.unwrap_or(target_t);

    reconstruct_nufrost_geotiff(reader, &timestamps_days, target_t, &config, output, &meta)
        .with_context(|| format!("NUFROST GeoTIFF reconstruction failed"))?;

    eprintln!("NUFROST GeoTIFF reconstruction written to {}", output.display());
    Ok(())
}

fn run_hants_geotiff(
    args: &HantsArgs,
    reader: &RasterReader,
    output: &PathBuf,
) -> Result<()> {
    let config = load_hants_config(args.config.as_deref())?;
    let meta = metadata_from_reader(reader);
    let (timestamps_days, target_t) = synthetic_timestamps_from_bands(reader.band_count());
    let target_t = args.shared.target_time.unwrap_or(target_t);

    reconstruct_hants_geotiff(
        reader,
        &timestamps_days,
        target_t,
        config.nof,
        &config.sf,
        config.valid_min,
        config.valid_max,
        config.fet,
        config.dod,
        config.period,
        output,
        &meta,
    )
    .with_context(|| "HANTS GeoTIFF reconstruction failed")?;

    eprintln!("HANTS GeoTIFF reconstruction written to {}", output.display());
    Ok(())
}

fn run_zhu2015_geotiff(
    args: &Zhu2015Args,
    reader: &RasterReader,
    output: &PathBuf,
) -> Result<()> {
    let config = load_zhu2015_config(args.config.as_deref())?;
    let meta = metadata_from_reader(reader);
    let (timestamps_days, target_t) = synthetic_timestamps_from_bands(reader.band_count());
    let target_t = args.shared.target_time.unwrap_or(target_t);

    reconstruct_zhu2015_geotiff(reader, &timestamps_days, target_t, config.lasso_alpha, output, &meta)
        .with_context(|| "Zhu2015 GeoTIFF reconstruction failed")?;

    eprintln!(
        "Zhu2015 GeoTIFF reconstruction written to {}",
        output.display()
    );
    Ok(())
}

// ═══════════════════════════════════════════════════════════════════════════
//  Full-scene reconstruction handler
// ═══════════════════════════════════════════════════════════════════════════

fn run_full_scene(args: &FullSceneArgs) -> Result<()> {
    if let Some(n) = args.n_jobs {
        rayon::ThreadPoolBuilder::new()
            .num_threads(n)
            .build_global()
            .context("Failed to build rayon thread pool")?;
    }

    let methods: Vec<&str> = args.methods.split(',').map(|s| s.trim()).collect();

    // 1. Discover band stacks
    let source_dir = args.data_root.join(&args.source_name);
    let band_stacks = discover_sentinel_band_stacks(&source_dir, args.lon, args.lat)?;
    if band_stacks.is_empty() {
        bail!(
            "No band stacks found for lon={}, lat={} in {}",
            args.lon, args.lat, source_dir.display()
        );
    }

    // 2. Load per-band cubes and ISO timestamps
    let mut band_cubes: BTreeMap<String, Array3<f64>> = BTreeMap::new();
    let mut band_timestamps: BTreeMap<String, Vec<String>> = BTreeMap::new();

    for (band_name, chunk_paths) in &band_stacks {
        let mut cube_parts: Vec<Array3<f64>> = Vec::new();
        let mut ts_parts: Vec<String> = Vec::new();

        for chunk_path in chunk_paths {
            let reader = RasterReader::open(chunk_path)?;
            let cube = if let Some(ws) = args.window_size {
                read_all_bands_window(&reader, ws, ws)?
            } else {
                read_all_bands(&reader)?
            };
            let descs = extract_raw_band_descriptions(&reader)?;
            cube_parts.push(cube);
            ts_parts.extend(descs);
        }

        let cube_merged = if cube_parts.len() == 1 {
            cube_parts.remove(0)
        } else {
            let views: Vec<_> = cube_parts.iter().map(|c| c.view()).collect();
            ndarray::concatenate(ndarray::Axis(0), &views)?
        };
        band_cubes.insert(band_name.clone(), cube_merged);
        band_timestamps.insert(band_name.clone(), ts_parts);
    }

    // 3. Mask invalid reflectance
    for cube in band_cubes.values_mut() {
        mask_invalid_sentinel2(cube);
    }

    // 4. Target timestamp selection
    let (target_time_str, _completeness) = choose_shared_target_timestamp(
        &band_cubes,
        &band_timestamps,
        args.min_valid_ratio,
        args.late_fraction,
    )?;
    eprintln!("Selected target timestamp: {target_time_str}");

    // 5. Ordered band list
    let ordered_bands: Vec<String> = sorted_band_names(&band_stacks)
        .into_iter()
        .map(|s| s.to_string())
        .collect();

    // 6. Hold-out: make masked time series per band
    let mut masked_cubes: BTreeMap<String, Array3<f64>> = BTreeMap::new();
    let mut masked_ts: BTreeMap<String, Vec<String>> = BTreeMap::new();
    let mut ground_truths: BTreeMap<String, ndarray::Array2<f64>> = BTreeMap::new();

    for band_name in &ordered_bands {
        let cube = band_cubes
            .get(band_name)
            .with_context(|| format!("Cube missing for band {band_name}"))?;
        let ts = band_timestamps
            .get(band_name)
            .with_context(|| format!("Timestamps missing for band {band_name}"))?;

        let (mc, mts, _target_idx, gt) = make_masked_time_series(cube, ts, &target_time_str)?;
        masked_cubes.insert(band_name.clone(), mc);
        masked_ts.insert(band_name.clone(), mts);
        ground_truths.insert(band_name.clone(), gt);
    }

    // 7. Metadata from first chunk
    let first_chunk = band_stacks.values().next().unwrap().first().unwrap();
    let first_reader = RasterReader::open(first_chunk)?;
    let meta = RasterMetadata {
        geo_transform: first_reader.geo_transform().unwrap_or([0.0, 1.0, 0.0, 0.0, 0.0, -1.0]),
        crs_wkt: first_reader.crs_wkt(),
        nodata: first_reader.nodata(1),
    };

    // 8. Build shared frequency pool (NUFROST, if spectral top-k requested)
    let shared_freqs: Option<Vec<f64>> = if methods.contains(&"nufrost")
        && (args.spectral_top_k.is_some() || args.preferred_top_k.is_some())
    {
        let top_k = args.spectral_top_k.unwrap_or(10);
        let nufrost_conf = default_nufrost_config();

        let all_epochs: Vec<f64> = band_timestamps
            .values()
            .flatten()
            .filter_map(|s| parse_iso8601_to_epoch_seconds(s))
            .collect();
        let ts_sec = to_seconds_since_start(&all_epochs);

        let pool = build_shared_frequency_pool(
            &band_cubes,
            &ts_sec,
            top_k,
            nufrost_conf.modes as usize,
            nufrost_conf.power_cum,
            nufrost_conf.ignore_dc_hz,
            500,
        );
        if !pool.is_empty() {
            eprintln!("Shared frequency pool: {} frequencies", pool.len());
            Some(pool)
        } else {
            None
        }
    } else {
        None
    };

    // 9. Target time as epoch
    let target_epoch = parse_iso8601_to_epoch_seconds(&target_time_str)
        .context("Failed to parse target timestamp")?;

    // 10. Load configs
    let nufrost_conf = default_nufrost_config();
    let hants_conf = default_hants_config();
    let zhu2015_conf = default_zhu2015_config();

    // 11. Per-method reconstruction
    let source_name = &args.source_name;
    let output_root = &args.output_root;
    let loc_token = full_scene::location_token(args.lon, args.lat);

    for method in &methods {
        let method_str = *method;
        eprintln!("Reconstructing with {method_str}...");

        let predictions: BTreeMap<String, ndarray::Array2<f64>> = ordered_bands
            .par_iter()
            .map(|band_name| {
                let cube = masked_cubes.get(band_name).unwrap();
                let ts_strs = masked_ts.get(band_name).unwrap();

                // Convert string timestamps to epoch seconds
                let mut epochs: Vec<f64> = ts_strs
                    .iter()
                    .filter_map(|s| parse_iso8601_to_epoch_seconds(s))
                    .collect();
                // Include target epoch for reference
                epochs.push(target_epoch);
                epochs.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));

                // Relative days from earliest epoch of this band+target set
                let ts_days: Vec<f64> = epochs
                    .iter()
                    .map(|&e| (e - epochs[0]) / 86400.0)
                    .collect();
                let target_day = (target_epoch - epochs[0]) / 86400.0;

                // The masked ts_days are all except the target
                let masked_days: Vec<f64> = ts_days
                    .iter()
                    .copied()
                    .filter(|&d| (d - target_day).abs() > 1e-9)
                    .collect();

                let (n_bands, rows, cols) = cube.dim();
                let mut pred = ndarray::Array2::<f64>::from_elem((rows, cols), f64::NAN);

                // Parallel per-row reconstruction
                pred.axis_iter_mut(ndarray::Axis(0))
                    .into_par_iter()
                    .enumerate()
                    .for_each(|(r, mut row_out)| {
                        let mut ts_buf = Vec::with_capacity(n_bands);
                        let mut obs_buf = Vec::with_capacity(n_bands);
                        for c in 0..cols {
                            ts_buf.clear();
                            obs_buf.clear();
                            for b in 0..n_bands {
                                let v = cube[[b, r, c]];
                                if v.is_finite() {
                                    ts_buf.push(masked_days[b]);
                                    obs_buf.push(v);
                                }
                            }
                            if !ts_buf.is_empty() {
                                match method_str {
                                    "nufrost" => {
                                        let (val, _) = if let Some(ref freqs) = shared_freqs
                                        {
                                            nufrost_pixel_with_shared(
                                                &ts_buf, &obs_buf, target_day,
                                                &nufrost_conf, freqs,
                                            )
                                        } else {
                                            nufrost_pixel(
                                                &ts_buf, &obs_buf, target_day,
                                                &nufrost_conf,
                                            )
                                        };
                                        row_out[c] = if val.is_finite() { val } else { f64::NAN };
                                    }
                                    "hants" => {
                                        let val = hants_pixel(
                                            &ts_buf, &obs_buf, target_day,
                                            hants_conf.nof, &hants_conf.sf,
                                            hants_conf.valid_min, hants_conf.valid_max,
                                            hants_conf.fet, hants_conf.dod, hants_conf.period,
                                        );
                                        row_out[c] = if val.is_finite() { val } else { f64::NAN };
                                    }
                                    "zhu2015" => {
                                        let result = fit_predict_pixel(
                                            &ts_buf, &obs_buf, target_day,
                                            zhu2015_conf.lasso_alpha,
                                        );
                                        row_out[c] = if result.prediction.is_finite() {
                                            result.prediction
                                        } else {
                                            f64::NAN
                                        };
                                    }
                                    _ => {}
                                }
                            }
                        }
                    });

                (band_name.clone(), pred)
            })
            .collect();

        // Write merged band stack
        let output_path = build_scene_stack_output_path(
            output_root,
            method_str,
            source_name,
            args.lon,
            args.lat,
            &target_time_str,
            "",
        );
        write_band_stack(&output_path, &predictions, &ordered_bands, &meta)?;
        eprintln!("  Wrote {}", output_path.display());
    }

    // 12. Write ground truth
    let gt_path = build_ground_truth_output_path(
        output_root,
        source_name,
        args.lon,
        args.lat,
        &target_time_str,
    );
    write_band_stack(&gt_path, &ground_truths, &ordered_bands, &meta)?;
    eprintln!("Ground truth written to {}", gt_path.display());

    // 13. Write summary JSON
    let now_iso = chrono::Utc::now().to_rfc3339();
    let summary: serde_json::Value = serde_json::json!({
        "source_name": source_name,
        "lon": args.lon,
        "lat": args.lat,
        "location_token": loc_token,
        "target_time": target_time_str,
        "target_epoch": target_epoch,
        "methods_run": methods,
        "ordered_bands": ordered_bands,
        "min_valid_ratio": args.min_valid_ratio,
        "late_fraction": args.late_fraction,
        "window_size": args.window_size,
        "generated_at": now_iso,
    });

    let summary_dir = output_root.join("run_summaries");
    fs::create_dir_all(&summary_dir)?;
    let safe_time = target_time_str.replace(':', "-");
    let loc_token6 = full_scene::location_output_token(args.lon, args.lat);
    let summary_path = summary_dir.join(format!(
        "reconstruction_summary_{source_name}_{loc_token6}_{safe_time}.json"
    ));
    fs::write(&summary_path, serde_json::to_string_pretty(&summary)?)?;
    eprintln!("Summary written to {}", summary_path.display());

    eprintln!("Full-scene reconstruction complete.");
    Ok(())
}

// ═══════════════════════════════════════════════════════════════════════════
//  Main
// ═══════════════════════════════════════════════════════════════════════════

fn main() -> Result<()> {
    let cli = Cli::parse();

    match &cli.algorithm {
        Algorithm::Nufrost(args) => {
            match detect_input_mode(&args.shared)? {
                InputMode::NpzFixture(fixture) => run_nufrost_npz(args, &fixture),
                InputMode::GeoTiff { reader, output } => run_nufrost_geotiff(args, &reader, &output),
            }
        }
        Algorithm::Hants(args) => {
            match detect_input_mode(&args.shared)? {
                InputMode::NpzFixture(fixture) => run_hants_npz(args, &fixture),
                InputMode::GeoTiff { reader, output } => run_hants_geotiff(args, &reader, &output),
            }
        }
        Algorithm::Zhu2015(args) => {
            match detect_input_mode(&args.shared)? {
                InputMode::NpzFixture(fixture) => run_zhu2015_npz(args, &fixture),
                InputMode::GeoTiff { reader, output } => run_zhu2015_geotiff(args, &reader, &output),
            }
        }
        Algorithm::FullScene(args) => run_full_scene(args),
    }
}

// ═══════════════════════════════════════════════════════════════════════════
//  Tests
// ═══════════════════════════════════════════════════════════════════════════

#[cfg(test)]
mod tests {
    use super::*;
    use clap::Parser;

    // ── Helpers ─────────────────────────────────────────────────────────

    /// Parse args and return the CLI struct.
    fn parse(args: &[&str]) -> Cli {
        Cli::try_parse_from(std::iter::once("nufrost-cli").chain(args.iter().copied()))
            .expect("valid args should parse")
    }

    /// Parse args and expect failure.
    fn parse_fails(args: &[&str]) -> String {
        Cli::try_parse_from(std::iter::once("nufrost-cli").chain(args.iter().copied()))
            .unwrap_err()
            .to_string()
    }

    // ── Subcommand routing ──────────────────────────────────────────────

    #[test]
    fn nufrost_subcommand() {
        let cli = parse(&["nufrost"]);
        assert!(matches!(cli.algorithm, Algorithm::Nufrost(_)));
    }

    #[test]
    fn hants_subcommand() {
        let cli = parse(&["hants"]);
        assert!(matches!(cli.algorithm, Algorithm::Hants(_)));
    }

    #[test]
    fn zhu2015_subcommand() {
        let cli = parse(&["zhu2015"]);
        assert!(matches!(cli.algorithm, Algorithm::Zhu2015(_)));
    }

    // ── NPZ mode option parsing ─────────────────────────────────────────

    #[test]
    fn nufrost_all_options_npz() {
        let cli = parse(&[
            "nufrost",
            "--config", "/tmp/cfg.json",
            "--data", "/tmp/data.npz",
            "--target-time", "372.7",
            "--output", "/tmp/out.txt",
            "--threads", "4",
        ]);
        match &cli.algorithm {
            Algorithm::Nufrost(args) => {
                assert_eq!(args.config.as_deref(), Some(std::path::Path::new("/tmp/cfg.json")));
                assert_eq!(args.shared.data.as_deref(), Some(std::path::Path::new("/tmp/data.npz")));
                assert_eq!(args.shared.target_time, Some(372.7));
                assert_eq!(args.shared.output.as_deref(), Some(std::path::Path::new("/tmp/out.txt")));
                assert_eq!(args.shared.threads, 4);
                assert!(args.shared.input_geotiff.is_none());
            }
            _ => panic!("expected Nufrost"),
        }
    }

    #[test]
    fn nufrost_geotiff_options() {
        let cli = parse(&[
            "nufrost",
            "--input-geotiff", "/tmp/input.tif",
            "--output", "/tmp/out.tif",
        ]);
        match &cli.algorithm {
            Algorithm::Nufrost(args) => {
                assert!(args.config.is_none());
                assert!(args.shared.data.is_none());
                assert_eq!(args.shared.input_geotiff.as_deref(), Some(std::path::Path::new("/tmp/input.tif")));
                assert_eq!(args.shared.output.as_deref(), Some(std::path::Path::new("/tmp/out.tif")));
            }
            _ => panic!("expected Nufrost"),
        }
    }

    #[test]
    fn hants_all_options() {
        let cli = parse(&[
            "hants",
            "--config", "/tmp/ch.json",
            "--data", "/tmp/d.npz",
            "-t", "200.0",
            "-o", "/tmp/o.txt",
        ]);
        match &cli.algorithm {
            Algorithm::Hants(args) => {
                assert_eq!(args.config.as_deref(), Some(std::path::Path::new("/tmp/ch.json")));
                assert_eq!(args.shared.target_time, Some(200.0));
                assert_eq!(args.shared.output.as_deref(), Some(std::path::Path::new("/tmp/o.txt")));
            }
            _ => panic!("expected Hants"),
        }
    }

    #[test]
    fn zhu2015_all_options() {
        let cli = parse(&[
            "zhu2015",
            "--config", "/tmp/cz.json",
            "--data", "/tmp/dz.npz",
            "-t", "500.0",
        ]);
        match &cli.algorithm {
            Algorithm::Zhu2015(args) => {
                assert_eq!(args.config.as_deref(), Some(std::path::Path::new("/tmp/cz.json")));
                assert_eq!(args.shared.target_time, Some(500.0));
            }
            _ => panic!("expected Zhu2015"),
        }
    }

    // ── Defaults ────────────────────────────────────────────────────────

    #[test]
    fn nufrost_minimal_args() {
        let cli = parse(&["nufrost"]);
        match &cli.algorithm {
            Algorithm::Nufrost(args) => {
                assert!(args.config.is_none());
                assert!(args.shared.data.is_none());
                assert!(args.shared.target_time.is_none());
                assert!(args.shared.output.is_none());
                assert_eq!(args.shared.threads, 1);
                assert!(args.shared.input_geotiff.is_none());
            }
            _ => panic!("expected Nufrost"),
        }
    }

    // ── Error handling ──────────────────────────────────────────────────

    #[test]
    fn unknown_algorithm() {
        let err = parse_fails(&["invalid_alg"]);
        assert!(
            err.contains("invalid_alg") || err.contains("unrecognized") || err.contains("doesn't"),
            "unexpected error: {err}"
        );
    }

    #[test]
    fn missing_required_data() {
        // No --data or --input-geotiff — but this is a runtime check, not clap parse error.
        // clap should accept this as valid since all args are optional.
        let cli = parse(&["nufrost", "--output", "/tmp/out.txt"]);
        match &cli.algorithm {
            Algorithm::Nufrost(args) => {
                assert!(args.shared.data.is_none());
                assert!(args.shared.input_geotiff.is_none());
            }
            _ => panic!("expected Nufrost"),
        }
    }
}
