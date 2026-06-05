// cli — command-line entrypoint for NUFROST, HANTS, and Zhu2015
// reconstruction algorithms.  Supports single-pixel fixture NPZ input and
// raster GeoTIFF input via gdal.

use std::collections::BTreeMap;
use std::ffi::OsString;
use std::fs;
use std::path::PathBuf;

use anyhow::{bail, Context, Result};
use clap::{Arg, ArgAction, Command};
use ndarray::Array3;
use rayon::prelude::*;
use serde::de::DeserializeOwned;

use nufrost_core::{
    nufrost_pixel, nufrost_pixel_vector, parse_iso8601_to_epoch_seconds,
    reconstruct_nufrost_geotiff, NufrostConfig,
};
use hants_core::{hants_pixel, reconstruct_hants_geotiff, HantsConfig};
use zhu2015_core::{fit_predict_pixel, reconstruct_zhu2015_geotiff, Zhu2015Config};
use gdal::{
    full_scene::{
        self, build_ground_truth_output_path, build_scene_stack_output_path,
        choose_shared_target_timestamp, discover_sentinel_band_stacks,
        make_masked_time_series, mask_invalid_sentinel2,
        sorted_band_names, write_band_stack,
    },
    collapse_duplicate_timestamps,
    extract_raw_band_descriptions, read_all_bands,
    read_all_bands_window_offset,
    RasterMetadata, RasterReader,
};

// ═══════════════════════════════════════════════════════════════════════════
//  CLI definition
// ═══════════════════════════════════════════════════════════════════════════

#[derive(Debug)]
struct Cli {
    algorithm: Algorithm,
}

#[derive(Debug)]
enum Algorithm {
    Nufrost(NufrostArgs),
    Hants(HantsArgs),
    Zhu2015(Zhu2015Args),
    FullScene(FullSceneArgs),
}

#[derive(Debug)]
struct FullSceneArgs {
    source_name: String,
    lon: f64,
    lat: f64,
    output_root: PathBuf,
    data_root: PathBuf,
    methods: String,
    n_jobs: Option<usize>,
    window_size: Option<usize>,
    window_lon: Option<f64>,
    window_lat: Option<f64>,
    min_valid_ratio: f64,
    late_fraction: f64,
    #[allow(dead_code)]
    frequency_selection: Option<String>,
}

#[derive(Debug)]
struct NufrostArgs {
    config: Option<PathBuf>,
    shared: SharedArgs,
}

#[derive(Debug)]
struct HantsArgs {
    config: Option<PathBuf>,
    shared: SharedArgs,
}

#[derive(Debug)]
struct Zhu2015Args {
    config: Option<PathBuf>,
    shared: SharedArgs,
}

#[derive(Debug)]
struct SharedArgs {
    data: Option<PathBuf>,
    input_geotiff: Option<PathBuf>,
    target_time: Option<f64>,
    output: Option<PathBuf>,
    #[allow(dead_code)]
    threads: usize,
}

impl Cli {
    fn parse() -> Self {
        Self::try_parse_from(std::env::args_os()).unwrap_or_else(|err| err.exit())
    }

    fn try_parse_from<I, T>(args: I) -> std::result::Result<Self, clap::Error>
    where
        I: IntoIterator<Item = T>,
        T: Into<OsString> + Clone,
    {
        let matches = Self::command().try_get_matches_from(args)?;
        let (name, sub) = matches
            .subcommand()
            .expect("subcommand_required ensures a subcommand is present");
        let algorithm = match name {
            "nufrost" => Algorithm::Nufrost(NufrostArgs {
                config: sub.get_one::<PathBuf>("config").cloned(),
                shared: parse_shared_args(sub),
            }),
            "hants" => Algorithm::Hants(HantsArgs {
                config: sub.get_one::<PathBuf>("config").cloned(),
                shared: parse_shared_args(sub),
            }),
            "zhu2015" => Algorithm::Zhu2015(Zhu2015Args {
                config: sub.get_one::<PathBuf>("config").cloned(),
                shared: parse_shared_args(sub),
            }),
            "full-scene" => Algorithm::FullScene(FullSceneArgs {
                source_name: sub.get_one::<String>("source_name").cloned().unwrap(),
                lon: *sub.get_one::<f64>("lon").unwrap(),
                lat: *sub.get_one::<f64>("lat").unwrap(),
                output_root: sub.get_one::<PathBuf>("output_root").cloned().unwrap(),
                data_root: sub.get_one::<PathBuf>("data_root").cloned().unwrap(),
                methods: sub.get_one::<String>("methods").cloned().unwrap(),
                n_jobs: sub.get_one::<usize>("n_jobs").copied(),
                window_size: sub.get_one::<usize>("window_size").copied(),
                window_lon: sub.get_one::<f64>("window_lon").copied(),
                window_lat: sub.get_one::<f64>("window_lat").copied(),
                min_valid_ratio: *sub.get_one::<f64>("min_valid_ratio").unwrap(),
                late_fraction: *sub.get_one::<f64>("late_fraction").unwrap(),
                frequency_selection: sub.get_one::<String>("frequency_selection").cloned(),
            }),
            _ => unreachable!("clap accepted an unknown subcommand: {name}"),
        };
        Ok(Self { algorithm })
    }

    fn command() -> Command {
        Command::new("cli")
            .version(env!("CARGO_PKG_VERSION"))
            .about("NUFROST / HANTS / Zhu2015 time-series reconstruction CLI")
            .after_help(
                "Examples:\n  \
                 cli nufrost --data fixture.npz --target-time 372.7\n  \
                 cli hants --config hants.json --data fixture.npz\n  \
                 cli zhu2015 --data fixture.npz -t 372.7 -o pred.txt\n  \
                 cli nufrost --input-geotiff input.tif --output pred.tif\n  \
                 cli hants --input-geotiff input.tif -o pred.tif\n  \
                 cli zhu2015 --input-geotiff input.tif -o pred.tif",
            )
            .subcommand_required(true)
            .arg_required_else_help(true)
            .subcommand(algorithm_command("nufrost", "Run NUFROST reconstruction."))
            .subcommand(algorithm_command("hants", "Run HANTS reconstruction."))
            .subcommand(algorithm_command("zhu2015", "Run Zhu2015 reconstruction."))
            .subcommand(full_scene_command())
    }
}

fn parse_shared_args(matches: &clap::ArgMatches) -> SharedArgs {
    SharedArgs {
        data: matches.get_one::<PathBuf>("data").cloned(),
        input_geotiff: matches.get_one::<PathBuf>("input_geotiff").cloned(),
        target_time: matches.get_one::<f64>("target_time").copied(),
        output: matches.get_one::<PathBuf>("output").cloned(),
        threads: matches.get_one::<usize>("threads").copied().unwrap_or(1),
    }
}

fn algorithm_command(name: &'static str, about: &'static str) -> Command {
    Command::new(name)
        .about(about)
        .arg(
            Arg::new("config")
                .short('c')
                .long("config")
                .value_name("PATH")
                .value_parser(clap::value_parser!(PathBuf)),
        )
        .args(shared_cli_args())
}

fn shared_cli_args() -> Vec<Arg> {
    vec![
        Arg::new("data")
            .short('d')
            .long("data")
            .value_name("PATH")
            .value_parser(clap::value_parser!(PathBuf)),
        Arg::new("input_geotiff")
            .long("input-geotiff")
            .value_name("PATH")
            .value_parser(clap::value_parser!(PathBuf)),
        Arg::new("target_time")
            .short('t')
            .long("target-time")
            .value_name("DAYS")
            .value_parser(clap::value_parser!(f64)),
        Arg::new("output")
            .short('o')
            .long("output")
            .value_name("PATH")
            .value_parser(clap::value_parser!(PathBuf)),
        Arg::new("threads")
            .long("threads")
            .value_name("N")
            .default_value("1")
            .value_parser(clap::value_parser!(usize)),
    ]
}

fn full_scene_command() -> Command {
    Command::new("full-scene")
        .about("Run full-scene reconstruction for one location.")
        .arg(
            Arg::new("source_name")
                .long("source-name")
                .required(true)
                .value_name("NAME")
                .action(ArgAction::Set),
        )
        .arg(
            Arg::new("lon")
                .long("lon")
                .required(true)
                .value_name("LON")
                .value_parser(clap::value_parser!(f64)),
        )
        .arg(
            Arg::new("lat")
                .long("lat")
                .required(true)
                .value_name("LAT")
                .value_parser(clap::value_parser!(f64)),
        )
        .arg(
            Arg::new("output_root")
                .long("output-root")
                .value_name("PATH")
                .default_value("data/output")
                .value_parser(clap::value_parser!(PathBuf)),
        )
        .arg(
            Arg::new("data_root")
                .long("data-root")
                .value_name("PATH")
                .default_value("data")
                .value_parser(clap::value_parser!(PathBuf)),
        )
        .arg(
            Arg::new("methods")
                .long("methods")
                .value_name("LIST")
                .default_value("nufrost,hants,zhu2015")
                .action(ArgAction::Set),
        )
        .arg(
            Arg::new("n_jobs")
                .long("n-jobs")
                .value_name("N")
                .value_parser(clap::value_parser!(usize)),
        )
        .arg(
            Arg::new("window_size")
                .long("window-size")
                .value_name("PX")
                .value_parser(clap::value_parser!(usize)),
        )
        .arg(
            Arg::new("window_lon")
                .long("window-lon")
                .value_name("LON")
                .value_parser(clap::value_parser!(f64)),
        )
        .arg(
            Arg::new("window_lat")
                .long("window-lat")
                .value_name("LAT")
                .value_parser(clap::value_parser!(f64)),
        )
        .arg(
            Arg::new("min_valid_ratio")
                .long("min-valid-ratio")
                .value_name("RATIO")
                .default_value("0.9")
                .value_parser(clap::value_parser!(f64)),
        )
        .arg(
            Arg::new("late_fraction")
                .long("late-fraction")
                .value_name("RATIO")
                .default_value("0.25")
                .value_parser(clap::value_parser!(f64)),
        )
        .arg(
            Arg::new("frequency_selection")
                .long("frequency-selection")
                .value_name("MODE")
                .action(ArgAction::Set),
        )
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
        "nufrost":{"modes":4096,"eps":1e-12,"num_peaks":10,"power_cum":0.7,"ignore_dc_hz":1e-10,"frequency_selection":"shared_spectral","preferred_periods_days":"365.25,182.625,91.3125,30.4375","preferred_top_k":4,"spectral_top_k":8,"spectral_merge_tol":0.15,"refine_peaks":true,"include_trend":true,"ridge_lam":0.005,"freq_weight":2.0,"huber_iters":3,"huber_delta":0.05,"min_obs":12,"outlier_sigma":2.5,"lambda_step":0.05,"lambda_high":0.005,"low_freq_period_days":60.0,"step_dt_weighting":true,"max_outer_iter":5,"outer_tol":0.001,"joint_outlier":true,"joint_outlier_sigma":2.5,"admm_rho":1.0,"admm_max_iter":80,"admm_tol":0.0001,"private_top_k_per_band":2,"private_freq_penalty_mult":1.5},
        "hants":{"nof":3,"sf":"high","fet":500.0,"dod":5,"period":365.25,"valid_min":null,"valid_max":null},
        "zhu2015":{"lasso_alpha":0.1}
    }"#;
    parse_grouped_config_section(full_json.as_bytes(), "nufrost")
        .expect("hardcoded grouped default config must be valid")
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

fn parse_grouped_config_section<T: DeserializeOwned>(
    bytes: &[u8],
    section: &str,
) -> serde_json::Result<T> {
    let value: serde_json::Value = serde_json::from_slice(bytes)?;
    serde_json::from_value(value[section].clone())
}

#[allow(deprecated)]
fn synthetic_geotiff_timestamps(n_bands: usize) -> (Vec<f64>, f64) {
    gdal::synthetic_timestamps_from_bands(n_bands)
}

fn load_nufrost_config(path: Option<&std::path::Path>) -> Result<NufrostConfig> {
    match path {
        Some(p) => {
            let bytes = fs::read(p)
                .with_context(|| format!("Cannot read config: {}", p.display()))?;
            parse_grouped_config_section(&bytes, "nufrost")
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
            parse_grouped_config_section(&bytes, "hants")
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
            parse_grouped_config_section(&bytes, "zhu2015")
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
    let (timestamps_days, target_t) = synthetic_geotiff_timestamps(reader.band_count());
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
    let (timestamps_days, target_t) = synthetic_geotiff_timestamps(reader.band_count());
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
    let (timestamps_days, target_t) = synthetic_geotiff_timestamps(reader.band_count());
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

fn full_scene_window_offset(
    reader: &RasterReader,
    window_size: usize,
    window_lon: Option<f64>,
    window_lat: Option<f64>,
) -> Result<(usize, usize)> {
    let (rows, cols) = reader.shape();
    let ws = window_size.min(rows).min(cols);
    if let (Some(lon), Some(lat)) = (window_lon, window_lat) {
        let gt = reader
            .geo_transform()
            .context("--window-lon/--window-lat require a georeferenced raster")?;
        let det = gt[1] * gt[5] - gt[2] * gt[4];
        if det.abs() <= 1e-15 {
            bail!("Cannot invert raster geotransform for --window-lon/--window-lat");
        }
        let dx = lon - gt[0];
        let dy = lat - gt[3];
        let col = (gt[5] * dx - gt[2] * dy) / det;
        let row = (-gt[4] * dx + gt[1] * dy) / det;
        let center_col = col.floor() as isize;
        let center_row = row.floor() as isize;
        let half = (ws / 2) as isize;
        let max_row_off = rows.saturating_sub(ws) as isize;
        let max_col_off = cols.saturating_sub(ws) as isize;
        let row_off = (center_row - half).clamp(0, max_row_off) as usize;
        let col_off = (center_col - half).clamp(0, max_col_off) as usize;
        Ok((row_off, col_off))
    } else {
        let row_off = if rows > ws { (rows - ws) / 2 } else { 0 };
        let col_off = if cols > ws { (cols - ws) / 2 } else { 0 };
        Ok((row_off, col_off))
    }
}

fn reconstruct_nufrost_vector_scene(
    ordered_bands: &[String],
    masked_cubes: &BTreeMap<String, Array3<f64>>,
    masked_ts: &BTreeMap<String, Vec<String>>,
    target_epoch: f64,
    config: &NufrostConfig,
) -> Result<BTreeMap<String, ndarray::Array2<f64>>> {
    let first_band = ordered_bands
        .first()
        .context("NUFROST vector reconstruction requires at least one band")?;
    let ref_ts = masked_ts
        .get(first_band)
        .with_context(|| format!("Timestamps missing for band {first_band}"))?;
    let first_cube = masked_cubes
        .get(first_band)
        .with_context(|| format!("Cube missing for band {first_band}"))?;
    let (n_times, rows, cols) = first_cube.dim();

    let cube_refs: Vec<&Array3<f64>> = ordered_bands
        .iter()
        .map(|band_name| {
            let cube = masked_cubes
                .get(band_name)
                .with_context(|| format!("Cube missing for band {band_name}"))?;
            let ts = masked_ts
                .get(band_name)
                .with_context(|| format!("Timestamps missing for band {band_name}"))?;
            if ts != ref_ts {
                bail!("NUFROST vector reconstruction requires aligned timestamps; band {band_name} differs from {first_band}");
            }
            if cube.dim() != (n_times, rows, cols) {
                bail!("NUFROST vector reconstruction requires aligned cube shapes; band {band_name} has {:?}, expected {:?}", cube.dim(), (n_times, rows, cols));
            }
            Ok(cube)
        })
        .collect::<Result<Vec<_>>>()?;

    let mut epochs: Vec<f64> = ref_ts
        .iter()
        .filter_map(|s| parse_iso8601_to_epoch_seconds(s))
        .collect();
    epochs.push(target_epoch);
    epochs.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let ts_days: Vec<f64> = epochs
        .iter()
        .map(|&e| (e - epochs[0]) / 86400.0)
        .collect();
    let target_day = (target_epoch - epochs[0]) / 86400.0;
    let masked_days: Vec<f64> = ts_days
        .iter()
        .copied()
        .filter(|&d| (d - target_day).abs() > 1e-9)
        .collect();
    if masked_days.len() != n_times {
        bail!(
            "NUFROST vector reconstruction timestamp count mismatch: {} masked days for {} cube slices",
            masked_days.len(),
            n_times
        );
    }

    let n_bands = ordered_bands.len();
    let mut pred_cube = Array3::<f64>::from_elem((n_bands, rows, cols), f64::NAN);
    pred_cube
        .axis_iter_mut(ndarray::Axis(1))
        .into_par_iter()
        .enumerate()
        .for_each(|(r, mut row_out)| {
            let mut obs_by_band: Vec<Vec<f64>> = (0..n_bands)
                .map(|_| Vec::with_capacity(n_times))
                .collect();
            for c in 0..cols {
                for obs in obs_by_band.iter_mut() {
                    obs.clear();
                }
                for (bi, cube) in cube_refs.iter().enumerate() {
                    for ti in 0..n_times {
                        obs_by_band[bi].push(cube[[ti, r, c]]);
                    }
                }
                let pred = nufrost_pixel_vector(&masked_days, &obs_by_band, target_day, config);
                for (bi, &val) in pred.iter().enumerate() {
                    row_out[[bi, c]] = if val.is_finite() { val } else { f64::NAN };
                }
            }
        });

    Ok(ordered_bands
        .iter()
        .enumerate()
        .map(|(bi, band_name)| {
            (
                band_name.clone(),
                pred_cube.index_axis(ndarray::Axis(0), bi).to_owned(),
            )
        })
        .collect())
}

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
                let (row_off, col_off) =
                    full_scene_window_offset(&reader, ws, args.window_lon, args.window_lat)?;
                read_all_bands_window_offset(&reader, row_off, col_off, ws, ws)?
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

        let ts_before = ts.len();
        let (cube, ts) = collapse_duplicate_timestamps(cube, ts);
        eprintln!(
            "Band {band_name}: collapsed {} timestamp slices to {}",
            ts_before,
            ts.len(),
        );

        let (mc, mts, _target_idx, gt) = make_masked_time_series(&cube, &ts, &target_time_str)?;
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

    // 8. Target time as epoch
    let target_epoch = parse_iso8601_to_epoch_seconds(&target_time_str)
        .context("Failed to parse target timestamp")?;

    // 10. Load configs
    let nufrost_conf = load_nufrost_config(Some(std::path::Path::new("config/nufrost.json")))
        .unwrap_or_else(|_| default_nufrost_config());
    let hants_conf = default_hants_config();
    let zhu2015_conf = default_zhu2015_config();

    // 11. Per-method reconstruction
    let source_name = &args.source_name;
    let output_root = &args.output_root;
    let loc_token = full_scene::location_token(args.lon, args.lat);

    for method in &methods {
        let method_str = *method;
        eprintln!("Reconstructing with {method_str}...");

        let predictions: BTreeMap<String, ndarray::Array2<f64>> = if method_str == "nufrost" {
            reconstruct_nufrost_vector_scene(
                &ordered_bands,
                &masked_cubes,
                &masked_ts,
                target_epoch,
                &nufrost_conf,
            )?
        } else {
            // Per-band independent reconstruction for HANTS and Zhu2015 baselines.
            ordered_bands
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
                                            let val = nufrost_pixel(&ts_buf, &obs_buf, target_day, &nufrost_conf).0;
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
                .collect()
        };

        // Write merged band stack
        let output_path = build_scene_stack_output_path(
            output_root,
            method_str,
            source_name,
            args.lon,
            args.lat,
            &target_time_str,
            "prediction",
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
        "window_lon": args.window_lon,
        "window_lat": args.window_lat,
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

    // ── Helpers ─────────────────────────────────────────────────────────

    /// Parse args and return the CLI struct.
    fn parse(args: &[&str]) -> Cli {
        Cli::try_parse_from(std::iter::once("cli").chain(args.iter().copied()))
            .expect("valid args should parse")
    }

    /// Parse args and expect failure.
    fn parse_fails(args: &[&str]) -> String {
        Cli::try_parse_from(std::iter::once("cli").chain(args.iter().copied()))
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

    // ── Contract: shared pool flags must be removed ──────────────────────

    /// Contract test: `--spectral-top-k` must NOT be accepted by `full-scene`.
    ///
    /// FAILS now (flag exists) → will PASS after T6 removes it from
    /// `FullSceneArgs`.
    #[test]
    fn full_scene_rejects_spectral_top_k() {
        let err = parse_fails(&[
            "full-scene",
            "--source-name", "sentinel-2",
            "--lon", "100.0",
            "--lat", "25.0",
            "--spectral-top-k", "4",
        ]);
        assert!(
            err.contains("spectral-top-k")
                || err.contains("unrecognized")
                || err.contains("unexpected")
                || err.contains("error")
                || err.contains("invalid"),
            "expected '--spectral-top-k' to be rejected, got: {err}"
        );
    }

    /// Contract test: `--preferred-top-k` must NOT be accepted by `full-scene`.
    ///
    /// FAILS now (flag exists) → will PASS after T6 removes it from
    /// `FullSceneArgs`.
    #[test]
    fn full_scene_rejects_preferred_top_k() {
        let err = parse_fails(&[
            "full-scene",
            "--source-name", "sentinel-2",
            "--lon", "100.0",
            "--lat", "25.0",
            "--preferred-top-k", "4",
        ]);
        assert!(
            err.contains("preferred-top-k")
                || err.contains("unrecognized")
                || err.contains("unexpected")
                || err.contains("error")
                || err.contains("invalid"),
            "expected '--preferred-top-k' to be rejected, got: {err}"
        );
    }

    /// Contract test: `full-scene --help` must NOT mention shared-pool flags.
    ///
    /// FAILS now (help text includes them) → will PASS after T6 removes them.
    #[test]
    fn full_scene_help_omits_shared_pool_flags() {
        let mut cmd = Cli::command();
        let full_scene_cmd = cmd
            .find_subcommand_mut("full-scene")
            .expect("full-scene subcommand must exist");
        let mut buf: Vec<u8> = Vec::new();
        full_scene_cmd.write_help(&mut buf).unwrap();
        let help = String::from_utf8(buf).unwrap();

        assert!(
            !help.contains("spectral-top-k"),
            "help must NOT mention --spectral-top-k (shared pool flag to be removed)\nHelp:\n{help}"
        );
        assert!(
            !help.contains("preferred-top-k"),
            "help must NOT mention --preferred-top-k (shared pool flag to be removed)\nHelp:\n{help}"
        );
    }
}
