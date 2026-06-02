use serde::{Deserialize, Serialize};

use crate::error::NufrostError;

// ── Per-algorithm configuration structs ───────────────────────────────────

/// NUFROST reconstruction parameters.
///
/// Field names match the Python `config/nufrost.json` keys exactly,
/// except `ridge_lam` which maps to Python's `"ridge"` key.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct NufrostConfig {
    pub modes: u32,
    pub eps: f64,
    pub num_peaks: u32,
    pub power_cum: f64,
    pub ignore_dc_hz: f64,
    #[serde(default = "default_frequency_selection")]
    pub frequency_selection: String,
    #[serde(default = "default_empty_string")]
    pub preferred_periods_days: String,
    #[serde(default)]
    pub preferred_top_k: u32,
    #[serde(default = "default_spectral_top_k")]
    pub spectral_top_k: u32,
    #[serde(default = "default_spectral_merge_tol")]
    pub spectral_merge_tol: f64,
    pub refine_peaks: bool,
    pub include_trend: bool,
    /// Rust field name is `ridge_lam`; JSON key is `ridge` (matching Python config).
    #[serde(alias = "ridge_lam", alias = "ridge")]
    pub ridge_lam: f64,
    pub freq_weight: f64,
    pub huber_iters: u32,
    pub huber_delta: f64,
    pub min_obs: u32,
    #[serde(default = "default_outlier_sigma")]
    pub outlier_sigma: f64,
    #[serde(default = "default_lambda_step")]
    pub lambda_step: f64,
    #[serde(default = "default_lambda_high")]
    pub lambda_high: f64,
    #[serde(default)]
    pub low_freq_period_days: f64,
    #[serde(default)]
    pub step_dt_weighting: bool,
    #[serde(default = "default_max_outer_iter")]
    pub max_outer_iter: u32,
    #[serde(default = "default_outer_tol")]
    pub outer_tol: f64,
    #[serde(default)]
    pub joint_outlier: bool,
    #[serde(default = "default_outlier_sigma")]
    pub joint_outlier_sigma: f64,
    #[serde(default = "default_admm_rho")]
    pub admm_rho: f64,
    #[serde(default = "default_admm_max_iter")]
    pub admm_max_iter: u32,
    #[serde(default = "default_admm_tol")]
    pub admm_tol: f64,
}

// Default helpers — match Python config/nufrost.json defaults.
fn default_frequency_selection() -> String { "shared_spectral".into() }
fn default_empty_string() -> String { String::new() }
fn default_spectral_top_k() -> u32 { 8 }
fn default_spectral_merge_tol() -> f64 { 0.15 }
fn default_outlier_sigma() -> f64 { 2.5 }
fn default_lambda_step() -> f64 { 1e30 }
fn default_lambda_high() -> f64 { 0.005 }
fn default_max_outer_iter() -> u32 { 5 }
fn default_outer_tol() -> f64 { 1e-3 }
fn default_admm_rho() -> f64 { 1.0 }
fn default_admm_max_iter() -> u32 { 80 }
fn default_admm_tol() -> f64 { 1e-4 }

/// HANTS reconstruction parameters.
///
/// Field names match `config/hants.json` exactly, with `Option<f64>` for the
/// optional `valid_min` / `valid_max` thresholds.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct HantsConfig {
    pub nof: u32,
    pub sf: String,
    pub fet: f64,
    pub dod: u32,
    pub valid_min: Option<f64>,
    pub valid_max: Option<f64>,
    pub period: f64,
}

/// Zhu2015 reconstruction parameters.
///
/// Field names match `config/zhu2015.json`.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct Zhu2015Config {
    pub lasso_alpha: f64,
}

// ── Unified config container ──────────────────────────────────────────────

/// Grouped reconstruction configuration for all three algorithms.
///
/// This struct directly mirrors the fixture `config.json` layout where each
/// algorithm is a nested key.  Serde deserialisation is driven by the JSON
/// structure, not by any runtime switching.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct ReconstructionConfig {
    pub nufrost: NufrostConfig,
    pub hants: HantsConfig,
    pub zhu2015: Zhu2015Config,
}

// ── Public constructors / validators ──────────────────────────────────────

impl NufrostConfig {
    /// Load from a JSON byte slice.
    pub fn from_json(data: &[u8]) -> Result<Self, NufrostError> {
        serde_json::from_slice(data).map_err(NufrostError::Json)
    }

    /// Validate that required numeric fields are in sensible ranges.
    ///
    /// Returns `Ok(())` or an [`NufrostError::InvalidConfigValue`].
    pub fn validate(&self) -> Result<(), NufrostError> {
        if self.modes == 0 {
            return Err(NufrostError::InvalidConfigValue {
                field: "modes".into(),
                reason: "must be > 0".into(),
            });
        }
        if self.num_peaks == 0 {
            return Err(NufrostError::InvalidConfigValue {
                field: "num_peaks".into(),
                reason: "must be > 0".into(),
            });
        }
        if self.eps <= 0.0 {
            return Err(NufrostError::InvalidConfigValue {
                field: "eps".into(),
                reason: "must be > 0".into(),
            });
        }
        Ok(())
    }
}

impl HantsConfig {
    /// Load from a JSON byte slice.
    pub fn from_json(data: &[u8]) -> Result<Self, NufrostError> {
        serde_json::from_slice(data).map_err(NufrostError::Json)
    }

    /// Validate required fields.
    pub fn validate(&self) -> Result<(), NufrostError> {
        if self.nof == 0 {
            return Err(NufrostError::InvalidConfigValue {
                field: "nof".into(),
                reason: "number of frequencies must be > 0".into(),
            });
        }
        if self.dod == 0 {
            return Err(NufrostError::InvalidConfigValue {
                field: "dod".into(),
                reason: "degree of over-determination must be > 0".into(),
            });
        }
        Ok(())
    }
}

impl Zhu2015Config {
    /// Load from a JSON byte slice.
    pub fn from_json(data: &[u8]) -> Result<Self, NufrostError> {
        serde_json::from_slice(data).map_err(NufrostError::Json)
    }

    /// Validate required fields.
    pub fn validate(&self) -> Result<(), NufrostError> {
        if self.lasso_alpha < 0.0 {
            return Err(NufrostError::InvalidConfigValue {
                field: "lasso_alpha".into(),
                reason: "must be >= 0".into(),
            });
        }
        Ok(())
    }
}

impl ReconstructionConfig {
    /// Load a grouped config from JSON bytes (matches fixture layout).
    pub fn from_json(data: &[u8]) -> Result<Self, NufrostError> {
        serde_json::from_slice(data).map_err(NufrostError::Json)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // ── Helper ────────────────────────────────────────────────────────────

    const NUFROST_JSON: &str = r#"{
        "modes": 4096,
        "eps": 1e-12,
        "num_peaks": 10,
        "power_cum": 0.7,
        "ignore_dc_hz": 1e-10,
        "refine_peaks": true,
        "include_trend": true,
        "ridge_lam": 0.005,
        "freq_weight": 2.0,
        "huber_iters": 3,
        "huber_delta": 0.05,
        "min_obs": 12
    }"#;

    const HANTS_JSON: &str = r#"{
        "nof": 3,
        "sf": "high",
        "fet": 500.0,
        "dod": 5,
        "valid_min": null,
        "valid_max": null,
        "period": 365.25
    }"#;

    const ZHU2015_JSON: &str = r#"{
        "lasso_alpha": 0.1
    }"#;

    const FIXTURE_GROUPED_JSON: &str = r#"{
        "nufrost": {
            "modes": 4096,
            "eps": 1e-12,
            "num_peaks": 10,
            "power_cum": 0.7,
            "ignore_dc_hz": 1e-10,
            "refine_peaks": true,
            "include_trend": true,
            "ridge_lam": 0.005,
            "freq_weight": 2.0,
            "huber_iters": 3,
            "huber_delta": 0.05,
            "min_obs": 12
        },
        "hants": {
            "nof": 3,
            "sf": "high",
            "fet": 500.0,
            "dod": 5,
            "valid_min": null,
            "valid_max": null,
            "period": 365.25
        },
        "zhu2015": {
            "lasso_alpha": 0.1
        }
    }"#;

    // ── Individual config tests ───────────────────────────────────────────

    #[test]
    fn parse_nufrost_config() {
        let cfg = NufrostConfig::from_json(NUFROST_JSON.as_bytes()).unwrap();
        assert_eq!(cfg.modes, 4096);
        assert_eq!(cfg.num_peaks, 10);
        assert!((cfg.eps - 1e-12).abs() < 1e-20);
        assert!((cfg.ridge_lam - 0.005).abs() < 1e-20);
        assert_eq!(cfg.min_obs, 12);
        // Defaults kick in for omitted fields
        assert_eq!(cfg.frequency_selection, "shared_spectral");
        assert_eq!(cfg.spectral_top_k, 8);
        assert_eq!(cfg.outlier_sigma, 2.5);
    }

    #[test]
    fn parse_hants_config() {
        let cfg = HantsConfig::from_json(HANTS_JSON.as_bytes()).unwrap();
        assert_eq!(cfg.nof, 3);
        assert_eq!(cfg.sf, "high");
        assert!((cfg.fet - 500.0).abs() < 1e-10);
        assert_eq!(cfg.dod, 5);
        assert!(cfg.valid_min.is_none());
        assert!(cfg.valid_max.is_none());
        assert!((cfg.period - 365.25).abs() < 1e-10);
    }

    #[test]
    fn parse_zhu2015_config() {
        let cfg = Zhu2015Config::from_json(ZHU2015_JSON.as_bytes()).unwrap();
        assert!((cfg.lasso_alpha - 0.1).abs() < 1e-10);
    }

    #[test]
    fn parse_grouped_fixture_config() {
        let cfg = ReconstructionConfig::from_json(FIXTURE_GROUPED_JSON.as_bytes()).unwrap();
        assert_eq!(cfg.nufrost.modes, 4096);
        assert_eq!(cfg.hants.nof, 3);
        assert!((cfg.zhu2015.lasso_alpha - 0.1).abs() < 1e-10);
    }

    // ── Validation tests ──────────────────────────────────────────────────

    #[test]
    fn nufrost_validate_zero_modes() {
        let mut cfg = NufrostConfig::from_json(NUFROST_JSON.as_bytes()).unwrap();
        cfg.modes = 0;
        let err = cfg.validate().unwrap_err();
        assert!(format!("{err}").contains("modes"));
    }

    #[test]
    fn hants_validate_zero_nof() {
        let mut cfg = HantsConfig::from_json(HANTS_JSON.as_bytes()).unwrap();
        cfg.nof = 0;
        let err = cfg.validate().unwrap_err();
        assert!(format!("{err}").contains("nof"));
    }

    #[test]
    fn zhu2015_validate_negative_alpha() {
        let mut cfg = Zhu2015Config::from_json(ZHU2015_JSON.as_bytes()).unwrap();
        cfg.lasso_alpha = -0.5;
        let err = cfg.validate().unwrap_err();
        assert!(format!("{err}").contains("lasso_alpha"));
    }

    // ── Missing field error ───────────────────────────────────────────────

    #[test]
    fn missing_required_field_is_error() {
        // "ridge_lam" is required (deny_unknown_fields, no default) but
        // "ridge" alias exists.  Drop "modes" which has no default.
        let bad_json = r#"{"eps": 1e-12, "num_peaks": 10}"#;
        let result = NufrostConfig::from_json(bad_json.as_bytes());
        assert!(result.is_err());
        let err = result.unwrap_err();
        let msg = format!("{err}");
        assert!(msg.contains("missing"), "expected missing field error, got: {msg}");
    }

    #[test]
    fn parse_ridge_alias() {
        // Python config uses "ridge", Rust struct uses "ridge_lam" with alias.
        let json_with_ridge = r#"{
            "modes": 4096,
            "eps": 1e-12,
            "num_peaks": 10,
            "power_cum": 0.7,
            "ignore_dc_hz": 1e-10,
            "refine_peaks": true,
            "include_trend": true,
            "ridge": 0.005,
            "freq_weight": 2.0,
            "huber_iters": 3,
            "huber_delta": 0.05,
            "min_obs": 12
        }"#;
        let cfg = NufrostConfig::from_json(json_with_ridge.as_bytes()).unwrap();
        assert!((cfg.ridge_lam - 0.005).abs() < 1e-10);
    }
}
