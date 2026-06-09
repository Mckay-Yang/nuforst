// cli — command-line entrypoint for NUFROST, HANTS, and Zhu2015
// reconstruction algorithms.  Supports single-pixel fixture NPZ input and
// raster GeoTIFF input via gdal.

use std::collections::BTreeMap;
use std::collections::BTreeSet;
use std::ffi::OsString;
use std::fs;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicUsize, Ordering};
use std::time::Instant;

use anyhow::{bail, Context, Result};
use clap::{Arg, ArgAction, Command};
use ndarray::Array3;
use rayon::prelude::*;
use serde::de::DeserializeOwned;

use gdal::{
    collapse_duplicate_timestamps, extract_raw_band_descriptions,
    full_scene::{
        self, build_ground_truth_output_path, build_scene_stack_output_path,
        choose_shared_target_timestamp, discover_sentinel_band_stacks, make_masked_time_series,
        mask_invalid_sentinel2, sorted_band_names, write_band_stack,
    },
    read_all_bands_window_offset, sample_cache, scene_cache, RasterMetadata, RasterReader,
};
use hants_core::{hants_pixel, reconstruct_hants_geotiff, HantsConfig};
use nufrost_core::{
    nufrost_pixel, nufrost_pixel_vector, parse_iso8601_to_epoch_seconds,
    reconstruct_nufrost_geotiff, NufrostConfig,
};
use zhu2015_core::{fit_predict_pixel, reconstruct_zhu2015_geotiff, Zhu2015Config};

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
    BuildSceneCache(BuildSceneCacheArgs),
    BuildSampleCache(BuildSampleCacheArgs),
    BatchFullScene(BatchFullSceneArgs),
    PixelBench(PixelBenchArgs),
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
    scene_cache: Option<PathBuf>,
    nufrost_config: Option<PathBuf>,
    #[allow(dead_code)]
    frequency_selection: Option<String>,
}

#[derive(Debug)]
struct BuildSceneCacheArgs {
    source_name: String,
    lon: f64,
    lat: f64,
    data_root: PathBuf,
    cache_root: PathBuf,
    output: Option<PathBuf>,
}

#[derive(Debug)]
struct BuildSampleCacheArgs {
    source_name: String,
    scene_cache_root: PathBuf,
    output: PathBuf,
    n_samples: usize,
    min_joint_valid: usize,
    seed: u64,
    max_attempts_multiplier: usize,
    limit_scenes: Option<usize>,
}

#[derive(Debug)]
struct BatchFullSceneArgs {
    source_name: String,
    output_root: PathBuf,
    data_root: PathBuf,
    methods: String,
    n_jobs: Option<usize>,
    window_size: Option<usize>,
    min_valid_ratio: f64,
    late_fraction: f64,
    limit: Option<usize>,
    continue_on_error: bool,
}

#[derive(Debug)]
struct PixelBenchArgs {
    source_name: String,
    lon: f64,
    lat: f64,
    data_root: PathBuf,
    row: Option<usize>,
    col: Option<usize>,
    pixel_lon: Option<f64>,
    pixel_lat: Option<f64>,
    repeats: usize,
    min_valid_ratio: f64,
    late_fraction: f64,
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
                scene_cache: sub.get_one::<PathBuf>("scene_cache").cloned(),
                nufrost_config: sub.get_one::<PathBuf>("nufrost_config").cloned(),
                frequency_selection: sub.get_one::<String>("frequency_selection").cloned(),
            }),
            "build-scene-cache" => Algorithm::BuildSceneCache(BuildSceneCacheArgs {
                source_name: sub.get_one::<String>("source_name").cloned().unwrap(),
                lon: *sub.get_one::<f64>("lon").unwrap(),
                lat: *sub.get_one::<f64>("lat").unwrap(),
                data_root: sub.get_one::<PathBuf>("data_root").cloned().unwrap(),
                cache_root: sub.get_one::<PathBuf>("cache_root").cloned().unwrap(),
                output: sub.get_one::<PathBuf>("output").cloned(),
            }),
            "build-sample-cache" => Algorithm::BuildSampleCache(BuildSampleCacheArgs {
                source_name: sub.get_one::<String>("source_name").cloned().unwrap(),
                scene_cache_root: sub.get_one::<PathBuf>("scene_cache_root").cloned().unwrap(),
                output: sub.get_one::<PathBuf>("output").cloned().unwrap(),
                n_samples: *sub.get_one::<usize>("n_samples").unwrap(),
                min_joint_valid: *sub.get_one::<usize>("min_joint_valid").unwrap(),
                seed: *sub.get_one::<u64>("seed").unwrap(),
                max_attempts_multiplier: *sub.get_one::<usize>("max_attempts_multiplier").unwrap(),
                limit_scenes: sub.get_one::<usize>("limit_scenes").copied(),
            }),
            "batch-full-scene" => Algorithm::BatchFullScene(BatchFullSceneArgs {
                source_name: sub.get_one::<String>("source_name").cloned().unwrap(),
                output_root: sub.get_one::<PathBuf>("output_root").cloned().unwrap(),
                data_root: sub.get_one::<PathBuf>("data_root").cloned().unwrap(),
                methods: sub.get_one::<String>("methods").cloned().unwrap(),
                n_jobs: sub.get_one::<usize>("n_jobs").copied(),
                window_size: sub.get_one::<usize>("window_size").copied(),
                min_valid_ratio: *sub.get_one::<f64>("min_valid_ratio").unwrap(),
                late_fraction: *sub.get_one::<f64>("late_fraction").unwrap(),
                limit: sub.get_one::<usize>("limit").copied(),
                continue_on_error: sub.get_flag("continue_on_error"),
            }),
            "pixel-bench" => Algorithm::PixelBench(PixelBenchArgs {
                source_name: sub.get_one::<String>("source_name").cloned().unwrap(),
                lon: *sub.get_one::<f64>("lon").unwrap(),
                lat: *sub.get_one::<f64>("lat").unwrap(),
                data_root: sub.get_one::<PathBuf>("data_root").cloned().unwrap(),
                row: sub.get_one::<usize>("row").copied(),
                col: sub.get_one::<usize>("col").copied(),
                pixel_lon: sub.get_one::<f64>("pixel_lon").copied(),
                pixel_lat: sub.get_one::<f64>("pixel_lat").copied(),
                repeats: *sub.get_one::<usize>("repeats").unwrap(),
                min_valid_ratio: *sub.get_one::<f64>("min_valid_ratio").unwrap(),
                late_fraction: *sub.get_one::<f64>("late_fraction").unwrap(),
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
            .subcommand(build_scene_cache_command())
            .subcommand(build_sample_cache_command())
            .subcommand(batch_full_scene_command())
            .subcommand(pixel_bench_command())
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
                .default_value("nufrost")
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
            Arg::new("scene_cache")
                .long("scene-cache")
                .value_name("DIR")
                .help("Read full-scene cubes from a prebuilt mmap scene cache directory.")
                .value_parser(clap::value_parser!(PathBuf)),
        )
        .arg(
            Arg::new("nufrost_config")
                .long("nufrost-config")
                .value_name("PATH")
                .help("NUFROST config JSON used by full-scene reconstruction.")
                .value_parser(clap::value_parser!(PathBuf)),
        )
        .arg(
            Arg::new("frequency_selection")
                .long("frequency-selection")
                .value_name("MODE")
                .action(ArgAction::Set),
        )
}

fn build_scene_cache_command() -> Command {
    Command::new("build-scene-cache")
        .about("Build a mmap-ready full-scene cache from source GeoTIFF stacks.")
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
            Arg::new("data_root")
                .long("data-root")
                .value_name("PATH")
                .default_value("data")
                .value_parser(clap::value_parser!(PathBuf)),
        )
        .arg(
            Arg::new("cache_root")
                .long("cache-root")
                .value_name("PATH")
                .default_value("data/cache/scenes")
                .value_parser(clap::value_parser!(PathBuf)),
        )
        .arg(
            Arg::new("output")
                .long("output")
                .value_name("DIR")
                .help("Explicit cache directory. Overrides --cache-root derived path.")
                .value_parser(clap::value_parser!(PathBuf)),
        )
}

fn build_sample_cache_command() -> Command {
    Command::new("build-sample-cache")
        .about("Build a mmap-ready random time-series sample cache from scene caches.")
        .arg(
            Arg::new("source_name")
                .long("source-name")
                .required(true)
                .value_name("NAME")
                .action(ArgAction::Set),
        )
        .arg(
            Arg::new("scene_cache_root")
                .long("scene-cache-root")
                .value_name("PATH")
                .default_value("data/cache/scenes")
                .value_parser(clap::value_parser!(PathBuf)),
        )
        .arg(
            Arg::new("output")
                .long("output")
                .value_name("DIR")
                .default_value("data/cache/samples/sentinel-2_v1")
                .value_parser(clap::value_parser!(PathBuf)),
        )
        .arg(
            Arg::new("n_samples")
                .long("n-samples")
                .value_name("N")
                .default_value("1000000")
                .value_parser(clap::value_parser!(usize)),
        )
        .arg(
            Arg::new("min_joint_valid")
                .long("min-joint-valid")
                .value_name("N")
                .default_value("12")
                .value_parser(clap::value_parser!(usize)),
        )
        .arg(
            Arg::new("seed")
                .long("seed")
                .value_name("U64")
                .default_value("20260608")
                .value_parser(clap::value_parser!(u64)),
        )
        .arg(
            Arg::new("max_attempts_multiplier")
                .long("max-attempts-multiplier")
                .value_name("N")
                .default_value("20")
                .value_parser(clap::value_parser!(usize)),
        )
        .arg(
            Arg::new("limit_scenes")
                .long("limit-scenes")
                .value_name("N")
                .help("Debug/smoke option: only use the first N aligned scene caches.")
                .value_parser(clap::value_parser!(usize)),
        )
}

fn batch_full_scene_command() -> Command {
    Command::new("batch-full-scene")
        .about("Run full-scene reconstruction for every complete location.")
        .arg(
            Arg::new("source_name")
                .long("source-name")
                .required(true)
                .value_name("NAME")
                .action(ArgAction::Set),
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
                .default_value("nufrost")
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
            Arg::new("limit")
                .long("limit")
                .value_name("N")
                .value_parser(clap::value_parser!(usize)),
        )
        .arg(
            Arg::new("continue_on_error")
                .long("continue-on-error")
                .action(ArgAction::SetTrue),
        )
}

fn pixel_bench_command() -> Command {
    Command::new("pixel-bench")
        .about("Benchmark one real multi-band pixel time series with vector NUFROST.")
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
            Arg::new("data_root")
                .long("data-root")
                .value_name("PATH")
                .default_value("data")
                .value_parser(clap::value_parser!(PathBuf)),
        )
        .arg(
            Arg::new("row")
                .long("row")
                .value_name("ROW")
                .value_parser(clap::value_parser!(usize)),
        )
        .arg(
            Arg::new("col")
                .long("col")
                .value_name("COL")
                .value_parser(clap::value_parser!(usize)),
        )
        .arg(
            Arg::new("pixel_lon")
                .long("pixel-lon")
                .value_name("LON")
                .value_parser(clap::value_parser!(f64)),
        )
        .arg(
            Arg::new("pixel_lat")
                .long("pixel-lat")
                .value_name("LAT")
                .value_parser(clap::value_parser!(f64)),
        )
        .arg(
            Arg::new("repeats")
                .long("repeats")
                .value_name("N")
                .default_value("10")
                .value_parser(clap::value_parser!(usize)),
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

    let file =
        fs::File::open(path).with_context(|| format!("Cannot open fixture: {}", path.display()))?;
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
        "nufrost":{"modes":64,"eps":1e-12,"num_peaks":10,"power_cum":0.7,"ignore_dc_hz":1e-10,"frequency_selection":"all","preferred_periods_days":"365.25,182.625,91.3125,30.4375","preferred_top_k":4,"spectral_top_k":8,"spectral_merge_tol":0.15,"refine_peaks":true,"include_trend":true,"ridge_lam":2.0,"freq_weight":256.0,"huber_iters":3,"huber_delta":0.05,"min_obs":12,"outlier_sigma":2.5,"outlier_reject_iters":2,"outlier_reject_sigma":2.5,"outlier_reject_max_fraction":0.35,"lambda_step":0.05,"lambda_high":0.005,"low_freq_period_days":60.0,"step_dt_weighting":true,"max_outer_iter":5,"outer_tol":0.001,"joint_outlier":true,"joint_outlier_sigma":2.5,"admm_rho":1.0,"admm_max_iter":80,"admm_tol":0.0001,"private_top_k_per_band":2,"private_freq_penalty_mult":1.5},
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
    serde_json::from_str(r#"{"lasso_alpha":0.1}"#)
        .expect("hardcoded default Zhu2015 config must be valid")
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
            let bytes =
                fs::read(p).with_context(|| format!("Cannot read config: {}", p.display()))?;
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
            let bytes =
                fs::read(p).with_context(|| format!("Cannot read config: {}", p.display()))?;
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
            let bytes =
                fs::read(p).with_context(|| format!("Cannot read config: {}", p.display()))?;
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
    GeoTiff {
        reader: RasterReader,
        output: PathBuf,
    },
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
            let output = shared
                .output
                .clone()
                .ok_or_else(|| anyhow::anyhow!("--output <PATH> is required in GeoTIFF mode"))?;
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
            fs::write(p, &text).with_context(|| format!("Cannot write output: {}", p.display()))?;
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
        geo_transform: reader
            .geo_transform()
            .unwrap_or([0.0, 1.0, 0.0, 0.0, 0.0, -1.0]),
        crs_wkt: reader.crs_wkt(),
        nodata: reader.nodata(1),
    }
}

fn run_build_scene_cache(args: &BuildSceneCacheArgs) -> Result<()> {
    let cache_dir = scene_cache::build_scene_cache(
        &args.data_root,
        &args.cache_root,
        &args.source_name,
        args.lon,
        args.lat,
        args.output.as_deref(),
    )?;
    eprintln!("Scene cache written to {}", cache_dir.display());
    eprintln!("  data: {}", cache_dir.join("cube.f32.bin").display());
    eprintln!("  meta: {}", cache_dir.join("meta.json").display());
    Ok(())
}

fn run_build_sample_cache(args: &BuildSampleCacheArgs) -> Result<()> {
    let start = Instant::now();
    let meta = sample_cache::build_sample_cache(&sample_cache::SampleCacheBuildOptions {
        scene_cache_root: args.scene_cache_root.clone(),
        source_name: args.source_name.clone(),
        output_dir: args.output.clone(),
        n_samples: args.n_samples,
        min_joint_valid: args.min_joint_valid,
        seed: args.seed,
        max_attempts_multiplier: args.max_attempts_multiplier,
        limit_scenes: args.limit_scenes,
    })?;
    eprintln!("Sample cache written to {}", args.output.display());
    eprintln!(
        "  samples: {}",
        args.output.join(&meta.sample_file).display()
    );
    eprintln!("  mask: {}", args.output.join(&meta.mask_file).display());
    eprintln!(
        "  scene times: {}",
        args.output.join(&meta.scene_time_file).display()
    );
    eprintln!("  index: {}", args.output.join(&meta.index_file).display());
    eprintln!("  meta: {}", args.output.join("meta.json").display());
    eprintln!(
        "  shape: samples=[{}, {}, {}], scenes={}, elapsed={:.3}s",
        meta.n_samples,
        meta.max_times,
        meta.n_bands,
        meta.scenes.len(),
        start.elapsed().as_secs_f64()
    );
    Ok(())
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

    eprintln!("Zhu2015 completed: pred={:.6}", result.prediction);
    Ok(())
}

// ═══════════════════════════════════════════════════════════════════════════
//  GeoTIFF mode — full raster reconstruction
// ═══════════════════════════════════════════════════════════════════════════

fn run_nufrost_geotiff(args: &NufrostArgs, reader: &RasterReader, output: &PathBuf) -> Result<()> {
    let config = load_nufrost_config(args.config.as_deref())?;
    let meta = metadata_from_reader(reader);
    let (timestamps_days, target_t) = synthetic_geotiff_timestamps(reader.band_count());
    let target_t = args.shared.target_time.unwrap_or(target_t);

    reconstruct_nufrost_geotiff(reader, &timestamps_days, target_t, &config, output, &meta)
        .with_context(|| format!("NUFROST GeoTIFF reconstruction failed"))?;

    eprintln!(
        "NUFROST GeoTIFF reconstruction written to {}",
        output.display()
    );
    Ok(())
}

fn run_hants_geotiff(args: &HantsArgs, reader: &RasterReader, output: &PathBuf) -> Result<()> {
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

    eprintln!(
        "HANTS GeoTIFF reconstruction written to {}",
        output.display()
    );
    Ok(())
}

fn run_zhu2015_geotiff(args: &Zhu2015Args, reader: &RasterReader, output: &PathBuf) -> Result<()> {
    let config = load_zhu2015_config(args.config.as_deref())?;
    let meta = metadata_from_reader(reader);
    let (timestamps_days, target_t) = synthetic_geotiff_timestamps(reader.band_count());
    let target_t = args.shared.target_time.unwrap_or(target_t);

    reconstruct_zhu2015_geotiff(
        reader,
        &timestamps_days,
        target_t,
        config.lasso_alpha,
        output,
        &meta,
    )
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
    let (_, rows, cols) = first_cube.dim();

    let ts_sets: BTreeMap<&str, BTreeSet<&str>> = ordered_bands
        .iter()
        .map(|band_name| {
            let ts = masked_ts
                .get(band_name)
                .with_context(|| format!("Timestamps missing for band {band_name}"))?;
            Ok((band_name.as_str(), ts.iter().map(String::as_str).collect()))
        })
        .collect::<Result<_>>()?;
    let common_ts: Vec<String> = ref_ts
        .iter()
        .filter(|ts| {
            ts_sets
                .values()
                .all(|band_ts| band_ts.contains(ts.as_str()))
        })
        .cloned()
        .collect();
    if common_ts.is_empty() {
        bail!("NUFROST vector reconstruction found no common masked timestamps across bands");
    }
    if common_ts.len() != ref_ts.len() {
        eprintln!(
            "NUFROST vector scene: using {} common masked timestamps out of {} in reference band {first_band}",
            common_ts.len(),
            ref_ts.len()
        );
    }
    let n_times = common_ts.len();

    let band_inputs: Vec<(&Array3<f64>, Vec<usize>)> = ordered_bands
        .iter()
        .map(|band_name| {
            let cube = masked_cubes
                .get(band_name)
                .with_context(|| format!("Cube missing for band {band_name}"))?;
            let ts = masked_ts
                .get(band_name)
                .with_context(|| format!("Timestamps missing for band {band_name}"))?;
            if cube.dim().1 != rows || cube.dim().2 != cols {
                bail!(
                    "NUFROST vector reconstruction requires aligned spatial shapes; band {band_name} has {:?}, expected rows={rows}, cols={cols}",
                    cube.dim()
                );
            }
            if cube.dim().0 != ts.len() {
                bail!(
                    "NUFROST vector reconstruction timestamp count mismatch for band {band_name}: {} timestamps for {:?}",
                    ts.len(),
                    cube.dim()
                );
            }
            let index_by_ts: BTreeMap<&str, usize> = ts
                .iter()
                .enumerate()
                .map(|(idx, value)| (value.as_str(), idx))
                .collect();
            let indices = common_ts
                .iter()
                .map(|value| {
                    index_by_ts
                        .get(value.as_str())
                        .copied()
                        .with_context(|| format!("Timestamp {value} missing for band {band_name}"))
                })
                .collect::<Result<Vec<_>>>()?;
            Ok((cube, indices))
        })
        .collect::<Result<Vec<_>>>()?;

    let mut epochs: Vec<f64> = common_ts
        .iter()
        .filter_map(|s| parse_iso8601_to_epoch_seconds(s))
        .collect();
    epochs.push(target_epoch);
    epochs.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let ts_days: Vec<f64> = epochs.iter().map(|&e| (e - epochs[0]) / 86400.0).collect();
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
    eprintln!("NUFROST vector scene: {rows} rows x {cols} cols x {n_bands} bands");
    let progress = AtomicUsize::new(0);
    let progress_step = (rows / 10).max(1);
    pred_cube
        .axis_iter_mut(ndarray::Axis(1))
        .into_par_iter()
        .enumerate()
        .for_each(|(r, mut row_out)| {
            let mut obs_by_band: Vec<Vec<f64>> =
                (0..n_bands).map(|_| Vec::with_capacity(n_times)).collect();
            for c in 0..cols {
                for obs in obs_by_band.iter_mut() {
                    obs.clear();
                }
                for (bi, (cube, time_indices)) in band_inputs.iter().enumerate() {
                    for &ti in time_indices {
                        obs_by_band[bi].push(cube[[ti, r, c]]);
                    }
                }
                let pred = nufrost_pixel_vector(&masked_days, &obs_by_band, target_day, config);
                for (bi, &val) in pred.iter().enumerate() {
                    row_out[[bi, c]] = if val.is_finite() { val } else { f64::NAN };
                }
            }
            let done = progress.fetch_add(1, Ordering::Relaxed) + 1;
            if done == rows || done % progress_step == 0 {
                eprintln!("  NUFROST rows: {done}/{rows}");
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

fn validate_full_scene_methods(methods: &str) -> Result<Vec<String>> {
    let parsed: Vec<String> = methods
        .split(',')
        .map(str::trim)
        .filter(|s| !s.is_empty())
        .map(str::to_string)
        .collect();
    if parsed.is_empty() {
        bail!("--methods must contain at least one method");
    }
    for method in &parsed {
        match method.as_str() {
            "nufrost" | "hants" | "zhu2015" => {}
            _ => bail!(
                "Unknown full-scene method '{method}'. Expected one of: nufrost,hants,zhu2015"
            ),
        }
    }
    Ok(parsed)
}

fn resolve_full_scene_band_stacks(
    data_root: &Path,
    source_name: &str,
    lon: f64,
    lat: f64,
) -> Result<BTreeMap<String, Vec<PathBuf>>> {
    let source_dir = data_root.join(source_name);
    let mut band_stacks = discover_sentinel_band_stacks(&source_dir, lon, lat)?;
    if band_stacks.is_empty() {
        bail!(
            "No band stacks found for lon={}, lat={} in {}",
            lon,
            lat,
            source_dir.display()
        );
    }

    let vrt_dir = data_root.join("cache").join("local").join("vrts");
    let loc_token6 = full_scene::location_output_token(lon, lat);
    for (band_name, paths) in band_stacks.iter_mut() {
        if paths.len() <= 1 {
            continue;
        }

        let vrt_path = vrt_dir.join(format!("sentinel_{band_name}_{loc_token6}.vrt"));
        if vrt_path.exists() {
            eprintln!(
                "Band {band_name}: using cached mosaic VRT {} for {} source chunks",
                vrt_path.display(),
                paths.len()
            );
            *paths = vec![vrt_path];
        } else {
            bail!(
                "Band {band_name} has {} source chunks but no cached mosaic VRT at {}. \
                 Build or restore the VRT before full-scene reconstruction.",
                paths.len(),
                vrt_path.display()
            );
        }
    }

    Ok(band_stacks)
}

fn prediction_metrics_json(
    predictions: &BTreeMap<String, ndarray::Array2<f64>>,
    ground_truths: &BTreeMap<String, ndarray::Array2<f64>>,
    ordered_bands: &[String],
) -> Result<serde_json::Value> {
    let mut bands = serde_json::Map::new();
    let mut all_n = 0usize;
    let mut all_sum_err = 0.0f64;
    let mut all_sum_abs = 0.0f64;
    let mut all_sum_sq = 0.0f64;

    for band in ordered_bands {
        let pred = predictions
            .get(band)
            .with_context(|| format!("Prediction missing for band {band}"))?;
        let truth = ground_truths
            .get(band)
            .with_context(|| format!("Ground truth missing for band {band}"))?;
        if pred.dim() != truth.dim() {
            bail!(
                "Prediction/ground truth shape mismatch for band {band}: {:?} vs {:?}",
                pred.dim(),
                truth.dim()
            );
        }

        let mut n = 0usize;
        let mut sum_err = 0.0f64;
        let mut sum_abs = 0.0f64;
        let mut sum_sq = 0.0f64;
        for (&p, &t) in pred.iter().zip(truth.iter()) {
            if p.is_finite() && t.is_finite() {
                let err = p - t;
                n += 1;
                sum_err += err;
                sum_abs += err.abs();
                sum_sq += err * err;
            }
        }

        all_n += n;
        all_sum_err += sum_err;
        all_sum_abs += sum_abs;
        all_sum_sq += sum_sq;

        let (bias, mae, rmse) = if n == 0 {
            (f64::NAN, f64::NAN, f64::NAN)
        } else {
            (
                sum_err / n as f64,
                sum_abs / n as f64,
                (sum_sq / n as f64).sqrt(),
            )
        };
        bands.insert(
            band.clone(),
            serde_json::json!({
                "n": n,
                "bias": bias,
                "mae": mae,
                "rmse": rmse,
            }),
        );
    }

    let (overall_bias, overall_mae, overall_rmse) = if all_n == 0 {
        (f64::NAN, f64::NAN, f64::NAN)
    } else {
        (
            all_sum_err / all_n as f64,
            all_sum_abs / all_n as f64,
            (all_sum_sq / all_n as f64).sqrt(),
        )
    };

    Ok(serde_json::json!({
        "n": all_n,
        "overall_bias": overall_bias,
        "overall_mae": overall_mae,
        "overall_rmse": overall_rmse,
        "bands": bands,
    }))
}

fn parse_sentinel_location_from_filename(name: &str) -> Option<(String, String, String)> {
    let rest = name.strip_prefix("COPERNICUS_S2_HARMONIZED_")?;
    let (band, rest) = rest.split_once("_lon")?;
    let (lon, lat_with_suffix) = rest.split_once("_lat")?;
    let lat_stem = lat_with_suffix.strip_suffix(".tif")?;
    let lat = lat_stem.split('-').next().unwrap_or(lat_stem);
    Some((band.to_string(), lon.to_string(), lat.to_string()))
}

fn discover_complete_sentinel_locations(
    data_root: &Path,
    source_name: &str,
) -> Result<Vec<(f64, f64)>> {
    let source_dir = data_root.join(source_name);
    let entries = fs::read_dir(&source_dir)
        .with_context(|| format!("cannot read data directory: {}", source_dir.display()))?;
    let required: BTreeSet<String> = ["B2", "B3", "B4", "B8", "B11", "B12"]
        .into_iter()
        .map(str::to_string)
        .collect();
    let mut by_location: BTreeMap<(String, String), BTreeSet<String>> = BTreeMap::new();

    for entry in entries {
        let path = entry?.path();
        if !path.is_file() {
            continue;
        }
        let Some(name) = path.file_name().and_then(|s| s.to_str()) else {
            continue;
        };
        let Some((band, lon, lat)) = parse_sentinel_location_from_filename(name) else {
            continue;
        };
        by_location.entry((lon, lat)).or_default().insert(band);
    }

    let mut locations = Vec::new();
    for ((lon, lat), bands) in by_location {
        if required.is_subset(&bands) {
            locations.push((
                lon.parse::<f64>()
                    .with_context(|| format!("invalid lon in source filename: {lon}"))?,
                lat.parse::<f64>()
                    .with_context(|| format!("invalid lat in source filename: {lat}"))?,
            ));
        }
    }
    locations.sort_by(|a, b| {
        a.0.partial_cmp(&b.0)
            .unwrap_or(std::cmp::Ordering::Equal)
            .then_with(|| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal))
    });
    Ok(locations)
}

fn finite_median(values: &[f64]) -> f64 {
    let mut finite: Vec<f64> = values.iter().copied().filter(|v| v.is_finite()).collect();
    if finite.is_empty() {
        return f64::NAN;
    }
    finite.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let n = finite.len();
    if n % 2 == 1 {
        finite[n / 2]
    } else {
        0.5 * (finite[n / 2 - 1] + finite[n / 2])
    }
}

fn robust_standardize(values: &[f64]) -> Vec<f64> {
    let center = finite_median(values);
    let abs_dev: Vec<f64> = values
        .iter()
        .map(|&v| {
            if v.is_finite() {
                (v - center).abs()
            } else {
                f64::NAN
            }
        })
        .collect();
    let mut scale = 1.4826 * finite_median(&abs_dev);
    if !scale.is_finite() || scale <= 1e-6 {
        let finite: Vec<f64> = values.iter().copied().filter(|v| v.is_finite()).collect();
        let mean = finite.iter().copied().sum::<f64>() / finite.len().max(1) as f64;
        let var = finite
            .iter()
            .map(|&v| {
                let d = v - mean;
                d * d
            })
            .sum::<f64>()
            / finite.len().max(1) as f64;
        scale = var.sqrt();
    }
    let center = if center.is_finite() {
        center
    } else {
        values
            .iter()
            .copied()
            .filter(|v| v.is_finite())
            .sum::<f64>()
            / values.iter().filter(|v| v.is_finite()).count().max(1) as f64
    };
    let scale = scale.max(1e-6);
    values.iter().map(|&v| (v - center) / scale).collect()
}

fn run_pixel_bench(args: &PixelBenchArgs) -> Result<()> {
    if args.repeats == 0 {
        bail!("--repeats must be > 0");
    }
    if (args.row.is_some() || args.col.is_some()) && (args.row.is_none() || args.col.is_none()) {
        bail!("--row and --col must be provided together");
    }
    if (args.pixel_lon.is_some() || args.pixel_lat.is_some())
        && (args.pixel_lon.is_none() || args.pixel_lat.is_none())
    {
        bail!("--pixel-lon and --pixel-lat must be provided together");
    }

    let band_stacks =
        resolve_full_scene_band_stacks(&args.data_root, &args.source_name, args.lon, args.lat)?;
    let ordered_bands: Vec<String> = sorted_band_names(&band_stacks)
        .into_iter()
        .map(|s| s.to_string())
        .collect();

    let first_chunk = band_stacks.values().next().unwrap().first().unwrap();
    let first_reader = RasterReader::open(first_chunk)?;
    let (rows, cols) = first_reader.shape();
    let (row_off, col_off) = if let (Some(r), Some(c)) = (args.row, args.col) {
        if r >= rows || c >= cols {
            bail!("pixel row/col out of bounds: row={r}, col={c}, shape=({rows}, {cols})");
        }
        (r, c)
    } else if let (Some(lon), Some(lat)) = (args.pixel_lon, args.pixel_lat) {
        full_scene_window_offset(&first_reader, 1, Some(lon), Some(lat))?
    } else {
        (rows / 2, cols / 2)
    };

    let mut band_cubes: BTreeMap<String, Array3<f64>> = BTreeMap::new();
    let mut band_timestamps: BTreeMap<String, Vec<String>> = BTreeMap::new();

    let load_start = Instant::now();
    for (band_name, chunk_paths) in &band_stacks {
        if chunk_paths.len() != 1 {
            bail!(
                "pixel-bench expects resolved one-path band stacks; got {} paths for {band_name}",
                chunk_paths.len()
            );
        }
        let chunk_path = &chunk_paths[0];
        eprintln!(
            "Loading {band_name} pixel row={row_off}, col={col_off} from {}",
            chunk_path.display()
        );
        let reader = RasterReader::open(chunk_path)?;
        let cube = read_all_bands_window_offset(&reader, row_off, col_off, 1, 1)?;
        let descs = extract_raw_band_descriptions(&reader)?;
        band_cubes.insert(band_name.clone(), cube);
        band_timestamps.insert(band_name.clone(), descs);
    }
    for cube in band_cubes.values_mut() {
        mask_invalid_sentinel2(cube);
    }
    let load_elapsed = load_start.elapsed();

    let target_start = Instant::now();
    let (target_time_str, _) = choose_shared_target_timestamp(
        &band_cubes,
        &band_timestamps,
        args.min_valid_ratio,
        args.late_fraction,
    )?;
    let target_elapsed = target_start.elapsed();
    let target_epoch = parse_iso8601_to_epoch_seconds(&target_time_str)
        .context("Failed to parse target timestamp")?;

    let mut masked_series: Vec<Vec<f64>> = Vec::with_capacity(ordered_bands.len());
    let mut ref_ts: Option<Vec<String>> = None;
    let mut raw_slices = 0usize;
    let mut collapsed_slices = 0usize;
    let mut target_values = Vec::with_capacity(ordered_bands.len());

    for band_name in &ordered_bands {
        let cube = band_cubes
            .get(band_name)
            .with_context(|| format!("Cube missing for band {band_name}"))?;
        let ts = band_timestamps
            .get(band_name)
            .with_context(|| format!("Timestamps missing for band {band_name}"))?;
        raw_slices = raw_slices.max(ts.len());
        let (collapsed_cube, collapsed_ts) = collapse_duplicate_timestamps(cube, ts);
        collapsed_slices = collapsed_slices.max(collapsed_ts.len());
        let (masked_cube, masked_ts, _target_idx, gt) =
            make_masked_time_series(&collapsed_cube, &collapsed_ts, &target_time_str)?;
        if let Some(existing) = &ref_ts {
            if existing != &masked_ts {
                bail!("pixel-bench requires aligned masked timestamps; band {band_name} differs");
            }
        } else {
            ref_ts = Some(masked_ts.clone());
        }
        masked_series.push(masked_cube.iter().copied().collect());
        target_values.push(gt[[0, 0]]);
    }

    let ref_ts = ref_ts.context("No masked timestamps built")?;
    let mut epochs: Vec<f64> = ref_ts
        .iter()
        .filter_map(|s| parse_iso8601_to_epoch_seconds(s))
        .collect();
    epochs.push(target_epoch);
    epochs.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let ts_days: Vec<f64> = epochs.iter().map(|&e| (e - epochs[0]) / 86400.0).collect();
    let target_day = (target_epoch - epochs[0]) / 86400.0;
    let masked_days: Vec<f64> = ts_days
        .iter()
        .copied()
        .filter(|&d| (d - target_day).abs() > 1e-9)
        .collect();
    if masked_days.len() != masked_series.first().map_or(0, Vec::len) {
        bail!(
            "masked timestamp count mismatch: {} days vs {} observations",
            masked_days.len(),
            masked_series.first().map_or(0, Vec::len)
        );
    }

    let config = load_nufrost_config(Some(std::path::Path::new("config/nufrost.json")))
        .unwrap_or_else(|_| default_nufrost_config());
    let valid_joint = (0..masked_days.len())
        .filter(|&i| masked_series.iter().all(|band| band[i].is_finite()))
        .count();

    let mut nufft_bins = 0usize;
    let mut nufft_power_sum = 0.0f64;
    let nufft_start = Instant::now();
    for _ in 0..args.repeats {
        let valid_idx: Vec<usize> = (0..masked_days.len())
            .filter(|&i| {
                masked_days[i].is_finite() && masked_series.iter().all(|b| b[i].is_finite())
            })
            .collect();
        if valid_idx.len() < (config.min_obs as usize).max(3) {
            continue;
        }
        let t_sec: Vec<f64> = valid_idx
            .iter()
            .map(|&i| masked_days[i] * 86400.0)
            .collect();
        let t_min = t_sec.iter().copied().fold(f64::INFINITY, f64::min);
        let t_rel: Vec<f64> = t_sec.iter().map(|&t| t - t_min).collect();

        let mut spectrum_dims = Vec::with_capacity(masked_series.len());
        for band in &masked_series {
            let col: Vec<f64> = valid_idx.iter().map(|&i| band[i]).collect();
            spectrum_dims.push(robust_standardize(&col));
        }

        let (_freqs, power) = nufrost_core::nufft::type1_vector_power_kb(
            &t_rel,
            &spectrum_dims,
            config.modes as usize,
            nufrost_core::nufft::NufftOptions::default(),
        );
        nufft_bins = power.len();
        nufft_power_sum = power.iter().copied().filter(|v| v.is_finite()).sum();
    }
    let nufft_elapsed = nufft_start.elapsed();
    let nufft_per_call = nufft_elapsed.as_secs_f64() / args.repeats as f64;

    let mut predictions = Vec::new();
    let bench_start = Instant::now();
    for _ in 0..args.repeats {
        predictions = nufrost_pixel_vector(&masked_days, &masked_series, target_day, &config);
    }
    let bench_elapsed = bench_start.elapsed();
    let per_call = bench_elapsed.as_secs_f64() / args.repeats as f64;

    let rmse = {
        let mut n = 0usize;
        let mut sum_sq = 0.0;
        for (&p, &t) in predictions.iter().zip(target_values.iter()) {
            if p.is_finite() && t.is_finite() {
                let d = p - t;
                n += 1;
                sum_sq += d * d;
            }
        }
        if n == 0 {
            f64::NAN
        } else {
            (sum_sq / n as f64).sqrt()
        }
    };

    println!(
        "{}",
        serde_json::to_string_pretty(&serde_json::json!({
            "source_name": args.source_name,
            "scene_lon": args.lon,
            "scene_lat": args.lat,
            "row": row_off,
            "col": col_off,
            "ordered_bands": ordered_bands,
            "target_time": target_time_str,
            "raw_slices": raw_slices,
            "collapsed_slices": collapsed_slices,
            "masked_slices": masked_days.len(),
            "joint_valid_slices": valid_joint,
            "repeats": args.repeats,
            "load_seconds": load_elapsed.as_secs_f64(),
            "target_select_seconds": target_elapsed.as_secs_f64(),
            "nufft_prefix_total_seconds": nufft_elapsed.as_secs_f64(),
            "nufft_prefix_seconds_per_call": nufft_per_call,
            "nufft_bins": nufft_bins,
            "nufft_power_sum": nufft_power_sum,
            "bench_total_seconds": bench_elapsed.as_secs_f64(),
            "bench_seconds_per_call": per_call,
            "prediction": predictions,
            "target": target_values,
            "rmse": rmse,
        }))?
    );
    Ok(())
}

fn run_full_scene(args: &FullSceneArgs) -> Result<()> {
    if let Some(n) = args.n_jobs {
        rayon::ThreadPoolBuilder::new()
            .num_threads(n)
            .build_global()
            .context("Failed to build rayon thread pool")?;
    }

    let methods = validate_full_scene_methods(&args.methods)?;

    let load_start = Instant::now();
    let scene_cache::LoadedScene {
        ordered_bands,
        mut band_cubes,
        band_timestamps,
        meta,
        cache_dir,
    } = if let Some(cache_dir) = &args.scene_cache {
        if args.window_size.is_some() {
            bail!("--scene-cache currently supports full-scene runs only, not --window-size");
        }
        eprintln!("Loading scene from mmap cache: {}", cache_dir.display());
        scene_cache::load_scene_cache(cache_dir)?
    } else if let Some(window_size) = args.window_size {
        scene_cache::load_scene_from_geotiffs_window(
            &args.data_root,
            &args.source_name,
            args.lon,
            args.lat,
            window_size,
            args.window_lon,
            args.window_lat,
        )?
    } else {
        scene_cache::load_or_build_scene_cache(
            &args.data_root,
            &args.data_root.join("cache").join("scenes"),
            &args.source_name,
            args.lon,
            args.lat,
        )?
    };
    let scene_load_seconds = load_start.elapsed().as_secs_f64();
    eprintln!("Scene load completed in {scene_load_seconds:.3}s");

    // 3. Mask invalid reflectance
    for cube in band_cubes.values_mut() {
        mask_invalid_sentinel2(cube);
    }

    // 4. Target timestamp selection
    let target_time_str = if let Some(cache_dir) = &cache_dir {
        if let Some(target) = scene_cache::load_cached_target_timestamp(
            cache_dir,
            args.min_valid_ratio,
            args.late_fraction,
        )? {
            eprintln!("Using cached shared target timestamp: {target}");
            target
        } else {
            eprintln!("Selecting shared target timestamp...");
            let (target, _completeness) = choose_shared_target_timestamp(
                &band_cubes,
                &band_timestamps,
                args.min_valid_ratio,
                args.late_fraction,
            )?;
            scene_cache::store_cached_target_timestamp(
                cache_dir,
                args.min_valid_ratio,
                args.late_fraction,
                &target,
            )?;
            target
        }
    } else {
        eprintln!("Selecting shared target timestamp...");
        let (target, _completeness) = choose_shared_target_timestamp(
            &band_cubes,
            &band_timestamps,
            args.min_valid_ratio,
            args.late_fraction,
        )?;
        target
    };
    eprintln!("Selected target timestamp: {target_time_str}");

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

    // 8. Target time as epoch
    let target_epoch = parse_iso8601_to_epoch_seconds(&target_time_str)
        .context("Failed to parse target timestamp")?;

    // 10. Load configs
    let nufrost_config_path = args
        .nufrost_config
        .as_deref()
        .unwrap_or_else(|| std::path::Path::new("config/nufrost.json"));
    let mut nufrost_conf =
        load_nufrost_config(Some(nufrost_config_path)).unwrap_or_else(|_| default_nufrost_config());
    if let Some(mode) = &args.frequency_selection {
        nufrost_conf.frequency_selection = mode.clone();
    }
    let hants_conf = default_hants_config();
    let zhu2015_conf = default_zhu2015_config();

    // 11. Per-method reconstruction
    let source_name = &args.source_name;
    let output_root = &args.output_root;
    let loc_token = full_scene::location_token(args.lon, args.lat);

    let mut prediction_outputs = serde_json::Map::new();
    let mut metrics_by_method = serde_json::Map::new();

    for method in &methods {
        let method_str = method.as_str();
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
                    let ts_days: Vec<f64> =
                        epochs.iter().map(|&e| (e - epochs[0]) / 86400.0).collect();
                    let target_day = (target_epoch - epochs[0]) / 86400.0;

                    // The masked ts_days are all except the target
                    let masked_days: Vec<f64> = ts_days
                        .iter()
                        .copied()
                        .filter(|&d| (d - target_day).abs() > 1e-9)
                        .collect();

                    let (n_bands, rows, cols) = cube.dim();
                    let mut pred = ndarray::Array2::<f64>::from_elem((rows, cols), f64::NAN);
                    eprintln!(
                        "  {method_str}/{band_name}: {rows} rows x {cols} cols, {n_bands} time slices"
                    );
                    let progress = AtomicUsize::new(0);
                    let progress_step = (rows / 10).max(1);

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
                                            let val = nufrost_pixel(
                                                &ts_buf,
                                                &obs_buf,
                                                target_day,
                                                &nufrost_conf,
                                            )
                                            .0;
                                            row_out[c] =
                                                if val.is_finite() { val } else { f64::NAN };
                                        }
                                        "hants" => {
                                            let val = hants_pixel(
                                                &ts_buf,
                                                &obs_buf,
                                                target_day,
                                                hants_conf.nof,
                                                &hants_conf.sf,
                                                hants_conf.valid_min,
                                                hants_conf.valid_max,
                                                hants_conf.fet,
                                                hants_conf.dod,
                                                hants_conf.period,
                                            );
                                            row_out[c] =
                                                if val.is_finite() { val } else { f64::NAN };
                                        }
                                        "zhu2015" => {
                                            let result = fit_predict_pixel(
                                                &ts_buf,
                                                &obs_buf,
                                                target_day,
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
                            let done = progress.fetch_add(1, Ordering::Relaxed) + 1;
                            if done == rows || done % progress_step == 0 {
                                eprintln!("    {method_str}/{band_name} rows: {done}/{rows}");
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

        let metrics = prediction_metrics_json(&predictions, &ground_truths, &ordered_bands)?;
        if let Some(rmse) = metrics.get("overall_rmse").and_then(|v| v.as_f64()) {
            eprintln!("  Overall RMSE: {rmse:.6}");
        }
        prediction_outputs.insert(
            method.clone(),
            serde_json::Value::String(output_path.display().to_string()),
        );
        metrics_by_method.insert(method.clone(), metrics);
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

    // 13. Write summary JSON. When rerunning only a subset of methods, merge
    // into the existing summary so older baseline metrics remain available.
    let summary_dir = output_root.join("run_summaries");
    fs::create_dir_all(&summary_dir)?;
    let safe_time = target_time_str.replace(':', "-");
    let loc_token6 = full_scene::location_output_token(args.lon, args.lat);
    let summary_path = summary_dir.join(format!(
        "reconstruction_summary_{source_name}_{loc_token6}_{safe_time}.json"
    ));

    let mut merged_prediction_outputs = if let Ok(bytes) = fs::read(&summary_path) {
        serde_json::from_slice::<serde_json::Value>(&bytes)
            .ok()
            .and_then(|v| {
                v.get("prediction_outputs")
                    .and_then(|m| m.as_object())
                    .cloned()
            })
            .unwrap_or_default()
    } else {
        serde_json::Map::new()
    };
    for (method, output) in prediction_outputs {
        merged_prediction_outputs.insert(method, output);
    }

    let mut merged_metrics = if let Ok(bytes) = fs::read(&summary_path) {
        serde_json::from_slice::<serde_json::Value>(&bytes)
            .ok()
            .and_then(|v| v.get("metrics").and_then(|m| m.as_object()).cloned())
            .unwrap_or_default()
    } else {
        serde_json::Map::new()
    };
    for (method, metric) in metrics_by_method {
        merged_metrics.insert(method, metric);
    }

    let mut merged_methods: Vec<String> = merged_metrics.keys().cloned().collect();
    for method in &methods {
        if !merged_methods.iter().any(|m| m == method) {
            merged_methods.push(method.clone());
        }
    }
    merged_methods.sort();

    let now_iso = chrono::Utc::now().to_rfc3339();
    let summary: serde_json::Value = serde_json::json!({
        "source_name": source_name,
        "lon": args.lon,
        "lat": args.lat,
        "location_token": loc_token,
        "target_time": target_time_str,
        "target_epoch": target_epoch,
        "methods_run": merged_methods,
        "methods_updated": methods,
        "ordered_bands": ordered_bands,
        "min_valid_ratio": args.min_valid_ratio,
        "late_fraction": args.late_fraction,
        "window_size": args.window_size,
        "window_lon": args.window_lon,
        "window_lat": args.window_lat,
        "scene_cache": cache_dir.as_ref().map(|p| p.display().to_string()),
        "scene_load_seconds": scene_load_seconds,
        "prediction_outputs": merged_prediction_outputs,
        "ground_truth_output": gt_path.display().to_string(),
        "metrics": merged_metrics,
        "generated_at": now_iso,
    });
    fs::write(&summary_path, serde_json::to_string_pretty(&summary)?)?;
    eprintln!("Summary written to {}", summary_path.display());

    eprintln!("Full-scene reconstruction complete.");
    Ok(())
}

fn run_batch_full_scene(args: &BatchFullSceneArgs) -> Result<()> {
    validate_full_scene_methods(&args.methods)?;
    if let Some(n) = args.n_jobs {
        rayon::ThreadPoolBuilder::new()
            .num_threads(n)
            .build_global()
            .context("Failed to build rayon thread pool")?;
    }

    let mut locations = discover_complete_sentinel_locations(&args.data_root, &args.source_name)?;
    if let Some(limit) = args.limit {
        locations.truncate(limit);
    }
    if locations.is_empty() {
        bail!(
            "No complete Sentinel-2 locations found in {}",
            args.data_root.join(&args.source_name).display()
        );
    }
    eprintln!("Discovered {} complete locations.", locations.len());

    let mut runs = Vec::new();
    let mut failures = Vec::new();

    for (idx, (lon, lat)) in locations.iter().copied().enumerate() {
        eprintln!(
            "Batch location {}/{}: lon={lon:.4}, lat={lat:.4}",
            idx + 1,
            locations.len()
        );
        let scene_args = FullSceneArgs {
            source_name: args.source_name.clone(),
            lon,
            lat,
            output_root: args.output_root.clone(),
            data_root: args.data_root.clone(),
            methods: args.methods.clone(),
            n_jobs: None,
            window_size: args.window_size,
            window_lon: None,
            window_lat: None,
            min_valid_ratio: args.min_valid_ratio,
            late_fraction: args.late_fraction,
            scene_cache: None,
            nufrost_config: None,
            frequency_selection: None,
        };

        match run_full_scene(&scene_args) {
            Ok(()) => {
                runs.push(serde_json::json!({
                    "lon": lon,
                    "lat": lat,
                    "status": "ok",
                }));
            }
            Err(err) => {
                eprintln!("Location lon={lon:.4}, lat={lat:.4} failed: {err:#}");
                failures.push(serde_json::json!({
                    "lon": lon,
                    "lat": lat,
                    "status": "failed",
                    "error": format!("{err:#}"),
                }));
                if !args.continue_on_error {
                    bail!("Batch stopped after location lon={lon:.4}, lat={lat:.4} failed");
                }
            }
        }
    }

    let summary_dir = args.output_root.join("run_summaries");
    fs::create_dir_all(&summary_dir)?;
    let summary_path = summary_dir.join(format!(
        "batch_summary_{}_{}.json",
        args.source_name,
        chrono::Utc::now().format("%Y%m%dT%H%M%SZ")
    ));
    let summary = serde_json::json!({
        "source_name": args.source_name,
        "data_root": args.data_root.display().to_string(),
        "output_root": args.output_root.display().to_string(),
        "methods": args.methods,
        "window_size": args.window_size,
        "min_valid_ratio": args.min_valid_ratio,
        "late_fraction": args.late_fraction,
        "n_locations": locations.len(),
        "n_success": runs.len(),
        "n_failed": failures.len(),
        "runs": runs,
        "failures": failures,
    });
    fs::write(&summary_path, serde_json::to_string_pretty(&summary)?)?;
    eprintln!("Batch summary written to {}", summary_path.display());

    if !failures.is_empty() {
        bail!(
            "Batch completed with {} failed locations; see {}",
            failures.len(),
            summary_path.display()
        );
    }
    Ok(())
}

// ═══════════════════════════════════════════════════════════════════════════
//  Main
// ═══════════════════════════════════════════════════════════════════════════

fn main() -> Result<()> {
    let cli = Cli::parse();

    match &cli.algorithm {
        Algorithm::Nufrost(args) => match detect_input_mode(&args.shared)? {
            InputMode::NpzFixture(fixture) => run_nufrost_npz(args, &fixture),
            InputMode::GeoTiff { reader, output } => run_nufrost_geotiff(args, &reader, &output),
        },
        Algorithm::Hants(args) => match detect_input_mode(&args.shared)? {
            InputMode::NpzFixture(fixture) => run_hants_npz(args, &fixture),
            InputMode::GeoTiff { reader, output } => run_hants_geotiff(args, &reader, &output),
        },
        Algorithm::Zhu2015(args) => match detect_input_mode(&args.shared)? {
            InputMode::NpzFixture(fixture) => run_zhu2015_npz(args, &fixture),
            InputMode::GeoTiff { reader, output } => run_zhu2015_geotiff(args, &reader, &output),
        },
        Algorithm::FullScene(args) => run_full_scene(args),
        Algorithm::BuildSceneCache(args) => run_build_scene_cache(args),
        Algorithm::BuildSampleCache(args) => run_build_sample_cache(args),
        Algorithm::BatchFullScene(args) => run_batch_full_scene(args),
        Algorithm::PixelBench(args) => run_pixel_bench(args),
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

    #[test]
    fn build_sample_cache_options() {
        let cli = parse(&[
            "build-sample-cache",
            "--source-name",
            "sentinel-2",
            "--scene-cache-root",
            "data/cache/scenes",
            "--output",
            "data/cache/samples/test",
            "--n-samples",
            "128",
            "--min-joint-valid",
            "8",
            "--seed",
            "123",
            "--max-attempts-multiplier",
            "5",
            "--limit-scenes",
            "2",
        ]);
        match cli.algorithm {
            Algorithm::BuildSampleCache(args) => {
                assert_eq!(args.source_name, "sentinel-2");
                assert_eq!(args.scene_cache_root, PathBuf::from("data/cache/scenes"));
                assert_eq!(args.output, PathBuf::from("data/cache/samples/test"));
                assert_eq!(args.n_samples, 128);
                assert_eq!(args.min_joint_valid, 8);
                assert_eq!(args.seed, 123);
                assert_eq!(args.max_attempts_multiplier, 5);
                assert_eq!(args.limit_scenes, Some(2));
            }
            other => panic!("expected BuildSampleCache, got {other:?}"),
        }
    }

    #[test]
    fn batch_full_scene_options() {
        let cli = parse(&[
            "batch-full-scene",
            "--source-name",
            "sentinel-2",
            "--data-root",
            "data",
            "--output-root",
            "/tmp/out",
            "--methods",
            "nufrost,hants",
            "--window-size",
            "3",
            "--n-jobs",
            "4",
            "--limit",
            "2",
            "--continue-on-error",
        ]);
        match &cli.algorithm {
            Algorithm::BatchFullScene(args) => {
                assert_eq!(args.source_name, "sentinel-2");
                assert_eq!(args.data_root, std::path::PathBuf::from("data"));
                assert_eq!(args.output_root, std::path::PathBuf::from("/tmp/out"));
                assert_eq!(args.methods, "nufrost,hants");
                assert_eq!(args.window_size, Some(3));
                assert_eq!(args.n_jobs, Some(4));
                assert_eq!(args.limit, Some(2));
                assert!(args.continue_on_error);
            }
            _ => panic!("expected BatchFullScene"),
        }
    }

    #[test]
    fn build_scene_cache_options() {
        let cli = parse(&[
            "build-scene-cache",
            "--source-name",
            "sentinel-2",
            "--lon",
            "94.2605",
            "--lat",
            "29.7733",
            "--data-root",
            "data",
            "--cache-root",
            "data/cache/scenes",
            "--output",
            "/tmp/scene-cache",
        ]);
        match &cli.algorithm {
            Algorithm::BuildSceneCache(args) => {
                assert_eq!(args.source_name, "sentinel-2");
                assert_eq!(args.lon, 94.2605);
                assert_eq!(args.lat, 29.7733);
                assert_eq!(args.data_root, std::path::PathBuf::from("data"));
                assert_eq!(
                    args.cache_root,
                    std::path::PathBuf::from("data/cache/scenes")
                );
                assert_eq!(
                    args.output,
                    Some(std::path::PathBuf::from("/tmp/scene-cache"))
                );
            }
            _ => panic!("expected BuildSceneCache"),
        }
    }

    #[test]
    fn full_scene_scene_cache_option() {
        let cli = parse(&[
            "full-scene",
            "--source-name",
            "sentinel-2",
            "--lon",
            "94.2605",
            "--lat",
            "29.7733",
            "--scene-cache",
            "data/cache/scenes/sentinel-2/lon94_260500_lat29_773300",
        ]);
        match &cli.algorithm {
            Algorithm::FullScene(args) => {
                assert_eq!(
                    args.scene_cache,
                    Some(std::path::PathBuf::from(
                        "data/cache/scenes/sentinel-2/lon94_260500_lat29_773300"
                    ))
                );
            }
            _ => panic!("expected FullScene"),
        }
    }

    #[test]
    fn full_scene_test_data_runs_end_to_end_with_auto_cache() {
        let workspace = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .parent()
            .unwrap()
            .parent()
            .unwrap()
            .to_path_buf();
        let data_root = workspace.join("data/test_data");
        let output_root = data_root.join("output/full_pipeline_test");
        let cache_dir = data_root
            .join("cache/scenes/sentinel-2")
            .join("lon100.112000_lat25.654000");

        let _ = std::fs::remove_dir_all(&cache_dir);
        let _ = std::fs::remove_dir_all(&output_root);

        let args = FullSceneArgs {
            source_name: "sentinel-2".to_string(),
            lon: 100.1120,
            lat: 25.6540,
            output_root: output_root.clone(),
            data_root: data_root.clone(),
            methods: "nufrost".to_string(),
            n_jobs: None,
            window_size: None,
            window_lon: None,
            window_lat: None,
            min_valid_ratio: 0.5,
            late_fraction: 0.25,
            scene_cache: None,
            nufrost_config: None,
            frequency_selection: None,
        };

        run_full_scene(&args).expect("test_data full-scene run should complete");
        assert!(cache_dir.join("meta.json").is_file());
        assert!(cache_dir.join("cube.f32.bin").is_file());

        let summary_dir = output_root.join("run_summaries");
        let summary_path = std::fs::read_dir(&summary_dir)
            .expect("summary dir should exist")
            .filter_map(Result::ok)
            .map(|entry| entry.path())
            .find(|path| {
                path.file_name()
                    .and_then(|name| name.to_str())
                    .is_some_and(|name| name.starts_with("reconstruction_summary_sentinel-2_"))
            })
            .expect("summary file should be written");
        let summary: serde_json::Value =
            serde_json::from_slice(&std::fs::read(&summary_path).unwrap()).unwrap();
        let rmse = summary["metrics"]["nufrost"]["overall_rmse"]
            .as_f64()
            .expect("nufrost rmse should be present");
        assert!(rmse.is_finite());
        assert_eq!(summary["methods_run"], serde_json::json!(["nufrost"]));
    }

    #[test]
    fn pixel_bench_options() {
        let cli = parse(&[
            "pixel-bench",
            "--source-name",
            "sentinel-2",
            "--lon",
            "94.2605",
            "--lat",
            "29.7733",
            "--row",
            "10",
            "--col",
            "20",
            "--repeats",
            "3",
        ]);
        match &cli.algorithm {
            Algorithm::PixelBench(args) => {
                assert_eq!(args.source_name, "sentinel-2");
                assert_eq!(args.lon, 94.2605);
                assert_eq!(args.lat, 29.7733);
                assert_eq!(args.row, Some(10));
                assert_eq!(args.col, Some(20));
                assert_eq!(args.repeats, 3);
            }
            _ => panic!("expected PixelBench"),
        }
    }

    // ── NPZ mode option parsing ─────────────────────────────────────────

    #[test]
    fn nufrost_all_options_npz() {
        let cli = parse(&[
            "nufrost",
            "--config",
            "/tmp/cfg.json",
            "--data",
            "/tmp/data.npz",
            "--target-time",
            "372.7",
            "--output",
            "/tmp/out.txt",
            "--threads",
            "4",
        ]);
        match &cli.algorithm {
            Algorithm::Nufrost(args) => {
                assert_eq!(
                    args.config.as_deref(),
                    Some(std::path::Path::new("/tmp/cfg.json"))
                );
                assert_eq!(
                    args.shared.data.as_deref(),
                    Some(std::path::Path::new("/tmp/data.npz"))
                );
                assert_eq!(args.shared.target_time, Some(372.7));
                assert_eq!(
                    args.shared.output.as_deref(),
                    Some(std::path::Path::new("/tmp/out.txt"))
                );
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
            "--input-geotiff",
            "/tmp/input.tif",
            "--output",
            "/tmp/out.tif",
        ]);
        match &cli.algorithm {
            Algorithm::Nufrost(args) => {
                assert!(args.config.is_none());
                assert!(args.shared.data.is_none());
                assert_eq!(
                    args.shared.input_geotiff.as_deref(),
                    Some(std::path::Path::new("/tmp/input.tif"))
                );
                assert_eq!(
                    args.shared.output.as_deref(),
                    Some(std::path::Path::new("/tmp/out.tif"))
                );
            }
            _ => panic!("expected Nufrost"),
        }
    }

    #[test]
    fn hants_all_options() {
        let cli = parse(&[
            "hants",
            "--config",
            "/tmp/ch.json",
            "--data",
            "/tmp/d.npz",
            "-t",
            "200.0",
            "-o",
            "/tmp/o.txt",
        ]);
        match &cli.algorithm {
            Algorithm::Hants(args) => {
                assert_eq!(
                    args.config.as_deref(),
                    Some(std::path::Path::new("/tmp/ch.json"))
                );
                assert_eq!(args.shared.target_time, Some(200.0));
                assert_eq!(
                    args.shared.output.as_deref(),
                    Some(std::path::Path::new("/tmp/o.txt"))
                );
            }
            _ => panic!("expected Hants"),
        }
    }

    #[test]
    fn zhu2015_all_options() {
        let cli = parse(&[
            "zhu2015",
            "--config",
            "/tmp/cz.json",
            "--data",
            "/tmp/dz.npz",
            "-t",
            "500.0",
        ]);
        match &cli.algorithm {
            Algorithm::Zhu2015(args) => {
                assert_eq!(
                    args.config.as_deref(),
                    Some(std::path::Path::new("/tmp/cz.json"))
                );
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
            "--source-name",
            "sentinel-2",
            "--lon",
            "100.0",
            "--lat",
            "25.0",
            "--spectral-top-k",
            "4",
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
            "--source-name",
            "sentinel-2",
            "--lon",
            "100.0",
            "--lat",
            "25.0",
            "--preferred-top-k",
            "4",
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
