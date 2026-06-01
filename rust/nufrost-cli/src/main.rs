// nufrost-cli — command-line entrypoint for NUFROST, HANTS, and Zhu2015
// reconstruction algorithms.  Supports single-pixel fixture NPZ input and
// (future) raster GeoTIFF input via nufrost-gdal.

use std::fs;
use std::path::PathBuf;

use anyhow::{bail, Context, Result};
use clap::{Parser, Subcommand};

use nufrost_core::{
    hants_pixel, nufrost_pixel, zhu2015::fit_predict_pixel, HantsConfig, NufrostConfig,
    Zhu2015Config,
};

// ═══════════════════════════════════════════════════════════════════════════
//  CLI definition (clap derive)
// ═══════════════════════════════════════════════════════════════════════════

/// NUFROST time-series reconstruction CLI.
///
/// Runs one of three algorithms on input time-series data.
/// Supports single-pixel NPZ fixtures and (future) raster GeoTIFF input.
#[derive(Parser, Debug)]
#[command(
    name = "nufrost-cli",
    version = env!("CARGO_PKG_VERSION"),
    about = "NUFROST / HANTS / Zhu2015 time-series reconstruction CLI",
    long_about = None,
    after_help = "Examples:\n  \
                  nufrost-cli nufrost --data fixture.npz --target-time 372.7\n  \
                  nufrost-cli hants --config hants.json --data fixture.npz\n  \
                  nufrost-cli zhu2015 --data fixture.npz -t 372.7 -o pred.txt",
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

// ── Shared args for all algorithms ──────────────────────────────────────

// ── Per-algorithm args ──────────────────────────────────────────────────

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

// Shared args type used by all three algorithm arg structs.
#[derive(clap::Args, Debug)]
struct SharedArgs {
    /// Path to NPZ fixture file.
    ///
    /// Expected keys: `timestamps_days`, `observations`, `target_time_day`.
    /// Used for single-pixel reconstruction.
    #[arg(short, long)]
    data: Option<PathBuf>,

    /// Target time in days since first observation.
    ///
    /// Overrides the `target_time_day` value embedded in the fixture NPZ.
    #[arg(short = 't', long)]
    target_time: Option<f64>,

    /// Output file path (writes scalar prediction to text file).
    ///
    /// If omitted, prints to stdout.
    #[arg(short, long)]
    output: Option<PathBuf>,

    /// Number of threads (reserved for future parallel raster processing).
    #[arg(long, default_value = "1")]
    threads: usize,
}

// ═══════════════════════════════════════════════════════════════════════════
//  Fixture loading
// ═══════════════════════════════════════════════════════════════════════════

/// Raw data extracted from an NPZ fixture.
#[derive(Debug)]
struct FixtureData {
    timestamps_days: Vec<f64>,
    observations: Vec<f64>,
    target_time_day: f64,
}

/// Load a single-pixel NPZ fixture.
///
/// Expected keys: `timestamps_days`, `observations`, `target_time_day`.
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
    // Full reconstruction config fixture, filtered to nufrost sub-config.
    // Matches tests/fixtures/rust_parity/real/small_window/config.json#nufrost.
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
    .unwrap()
}

/// Default Zhu2015 config matching Python `config/zhu2015.json`.
fn default_zhu2015_config() -> Zhu2015Config {
    serde_json::from_str(r#"{"lasso_alpha":0.1}"#).unwrap()
}

fn load_nufrost_config(path: Option<&std::path::Path>) -> Result<NufrostConfig> {
    match path {
        Some(p) => {
            let bytes = fs::read(p)
                .with_context(|| format!("Cannot read config: {}", p.display()))?;
            // Try parsing as ReconstructionConfig (full), fall back to NufrostConfig (standalone).
            serde_json::from_slice::<nufrost_core::ReconstructionConfig>(&bytes)
                .map(|rc| rc.nufrost)
                .or_else(|_| {
                    NufrostConfig::from_json(&bytes)
                })
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
//  Run helpers
// ═══════════════════════════════════════════════════════════════════════════

/// Resolve input data: require a fixture NPZ for now (raster input is future work).
fn require_fixture(data_path: Option<&PathBuf>) -> Result<FixtureData> {
    match data_path {
        Some(p) => load_fixture_npz(p),
        None => bail!(
            "No input data provided.  Use --data <fixture.npz> to provide a single-pixel \
             NPZ fixture with keys: timestamps_days, observations, target_time_day."
        ),
    }
}

/// Resolve target time: explicit CLI arg beats fixture-embedded value.
fn resolve_target_time(fixture: &FixtureData, cli_target: Option<f64>) -> f64 {
    cli_target.unwrap_or(fixture.target_time_day)
}

/// Write or print a single scalar result.
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

/// Run NUFROST on a single-pixel fixture.
fn run_nufrost(args: &NufrostArgs) -> Result<()> {
    let fixture = require_fixture(args.shared.data.as_ref())?;
    let config = load_nufrost_config(args.config.as_deref())?;
    let target_t = resolve_target_time(&fixture, args.shared.target_time);

    // nufrost_pixel uses internal time unit conversion; days work fine.
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

/// Run HANTS on a single-pixel fixture.
fn run_hants(args: &HantsArgs) -> Result<()> {
    let fixture = require_fixture(args.shared.data.as_ref())?;
    let config = load_hants_config(args.config.as_deref())?;
    let target_t = resolve_target_time(&fixture, args.shared.target_time);

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

/// Run Zhu2015 on a single-pixel fixture.
fn run_zhu2015(args: &Zhu2015Args) -> Result<()> {
    let fixture = require_fixture(args.shared.data.as_ref())?;
    let config = load_zhu2015_config(args.config.as_deref())?;
    let target_t = resolve_target_time(&fixture, args.shared.target_time);

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
//  Main
// ═══════════════════════════════════════════════════════════════════════════

fn main() -> Result<()> {
    let cli = Cli::parse();

    match cli.algorithm {
        Algorithm::Nufrost(args) => run_nufrost(&args),
        Algorithm::Hants(args) => run_hants(&args),
        Algorithm::Zhu2015(args) => run_zhu2015(&args),
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
        matches!(cli.algorithm, Algorithm::Nufrost(_));
    }

    #[test]
    fn hants_subcommand() {
        let cli = parse(&["hants"]);
        matches!(cli.algorithm, Algorithm::Hants(_));
    }

    #[test]
    fn zhu2015_subcommand() {
        let cli = parse(&["zhu2015"]);
        matches!(cli.algorithm, Algorithm::Zhu2015(_));
    }

    // ── Full option parsing ─────────────────────────────────────────────

    #[test]
    fn nufrost_all_options() {
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
    fn missing_required_subcommand() {
        let err = parse_fails(&[]);
        assert!(
            err.contains("subcommand") || err.contains("required"),
            "unexpected error: {err}"
        );
    }

    #[test]
    fn bad_target_time() {
        let err = parse_fails(&["nufrost", "--target-time", "not_a_number"]);
        assert!(
            err.contains("target-time") || err.contains("not_a_number") || err.contains("invalid"),
            "unexpected error: {err}"
        );
    }

    #[test]
    fn bad_threads_value() {
        // clap rejects non-numeric values for usize args
        let err = parse_fails(&["nufrost", "--threads", "abc"]);
        assert!(
            err.contains("threads") || err.contains("abc") || err.contains("invalid"),
            "unexpected error: {err}"
        );
    }

    #[test]
    fn unexpected_flag() {
        let err = parse_fails(&["nufrost", "--nonexistent"]);
        assert!(
            err.contains("nonexistent") || err.contains("unexpected") || err.contains("not found"),
            "unexpected error: {err}"
        );
    }

    // ── Short flags ─────────────────────────────────────────────────────

    #[test]
    fn short_flags() {
        let cli = parse(&[
            "hants",
            "-c", "/tmp/c.json",
            "-d", "/tmp/d.npz",
            "-t", "100.0",
            "-o", "/tmp/o.txt",
        ]);
        match &cli.algorithm {
            Algorithm::Hants(args) => {
                assert!(args.config.is_some());
                assert!(args.shared.data.is_some());
                assert_eq!(args.shared.target_time, Some(100.0));
                assert!(args.shared.output.is_some());
            }
            _ => panic!("expected Hants"),
        }
    }

    // ── Help text ───────────────────────────────────────────────────────

    #[test]
    fn help_includes_all_algorithms() {
        // Use try_parse_from to avoid actual exit(0).
        let err = Cli::try_parse_from(["nufrost-cli", "--help"])
            .unwrap_err();
        let msg = err.to_string();
        assert!(msg.contains("nufrost"), "help should mention 'nufrost': {msg}");
        assert!(msg.contains("hants"), "help should mention 'hants': {msg}");
        assert!(msg.contains("zhu2015"), "help should mention 'zhu2015': {msg}");
    }

    #[test]
    fn subcommand_help() {
        let err = Cli::try_parse_from(["nufrost-cli", "nufrost", "--help"])
            .unwrap_err();
        let msg = err.to_string();
        assert!(msg.contains("--config"), "help missing --config: {msg}");
        assert!(msg.contains("--data"), "help missing --data: {msg}");
        assert!(msg.contains("--target-time"), "help missing --target-time: {msg}");
        assert!(msg.contains("--output"), "help missing --output: {msg}");
        assert!(msg.contains("--threads"), "help missing --threads: {msg}");
    }

    // ── Integration: run each algorithm on a synthetic fixture ──────────

    /// Helper: path to a synthetic fixture NPZ.
    fn synthetic_fixture(name: &str) -> PathBuf {
        PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join("../../tests/fixtures/rust_parity/synthetic")
            .join(name)
            .join("data.npz")
    }

    #[test]
    fn integration_nufrost_simple_harmonic() {
        let fixture = synthetic_fixture("simple_harmonic");
        assert!(fixture.exists(), "fixture not found: {}", fixture.display());

        let data = load_fixture_npz(&fixture).expect("load fixture");
        let config = default_nufrost_config();
        let (pred, _n_freqs) = nufrost_pixel(
            &data.timestamps_days,
            &data.observations,
            data.target_time_day,
            &config,
        );
        assert!(pred.is_finite(), "NUFROST prediction should be finite");
    }

    #[test]
    fn integration_hants_simple_harmonic() {
        let fixture = synthetic_fixture("simple_harmonic");
        assert!(fixture.exists(), "fixture not found: {}", fixture.display());

        let data = load_fixture_npz(&fixture).expect("load fixture");
        let config = default_hants_config();
        let pred = hants_pixel(
            &data.timestamps_days,
            &data.observations,
            data.target_time_day,
            config.nof,
            &config.sf,
            config.valid_min,
            config.valid_max,
            config.fet,
            config.dod,
            config.period,
        );
        assert!(pred.is_finite(), "HANTS prediction should be finite");
    }

    #[test]
    fn integration_zhu2015_simple_harmonic() {
        let fixture = synthetic_fixture("simple_harmonic");
        assert!(fixture.exists(), "fixture not found: {}", fixture.display());

        let data = load_fixture_npz(&fixture).expect("load fixture");
        let config = default_zhu2015_config();
        let result = fit_predict_pixel(
            &data.timestamps_days,
            &data.observations,
            data.target_time_day,
            config.lasso_alpha,
        );
        assert!(result.prediction.is_finite(), "Zhu2015 prediction should be finite");
    }

    // ── Integration: error on missing data ──────────────────────────────

    #[test]
    fn require_fixture_missing() {
        let err = require_fixture(None).unwrap_err();
        let msg = err.to_string();
        assert!(
            msg.contains("data") || msg.contains("fixture") || msg.contains("input"),
            "error should mention data input: {msg}"
        );
    }

    #[test]
    fn require_fixture_not_found() {
        let err = require_fixture(Some(&PathBuf::from("/nonexistent/path.npz"))).unwrap_err();
        let msg = err.to_string();
        assert!(
            msg.contains("nonexistent") || msg.contains("Cannot open"),
            "error should mention file path: {msg}"
        );
    }

    // ── Config loading ──────────────────────────────────────────────────

    #[test]
    fn load_nufrost_default_config() {
        let cfg = load_nufrost_config(None).expect("default config");
        assert!(cfg.modes > 0);
        assert!(cfg.num_peaks > 0);
    }

    #[test]
    fn load_hants_default_config() {
        let cfg = load_hants_config(None).expect("default config");
        assert!(cfg.nof > 0);
        assert!(!cfg.sf.is_empty());
    }

    #[test]
    fn load_zhu2015_default_config() {
        let cfg = load_zhu2015_config(None).expect("default config");
        assert!(cfg.lasso_alpha > 0.0);
    }
}
