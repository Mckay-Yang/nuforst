// nufrost-cli — command-line entrypoint for NUFROST, HANTS, and Zhu2015
// reconstruction algorithms.  Supports single-pixel fixture NPZ input and
// raster GeoTIFF input via nufrost-gdal.

use std::fs;
use std::path::PathBuf;

use anyhow::{bail, Context, Result};
use clap::{Parser, Subcommand};

use nufrost_core::{
    hants_pixel, nufrost_pixel, zhu2015::fit_predict_pixel, HantsConfig, NufrostConfig,
    Zhu2015Config,
};
use nufrost_gdal::{
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
    /// In GeoTIFF mode: writes output GeoTIFF (single-band for NUFROST/HANTS,
    /// 2-band for Zhu2015).
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
        &format!("zhu2015_prediction (QA={})", result.qa),
    )?;

    eprintln!(
        "Zhu2015 completed: pred={:.6}, QA={}",
        result.prediction, result.qa
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
        "Zhu2015 GeoTIFF reconstruction written to {} (2-band: pred + QA)",
        output.display()
    );
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
