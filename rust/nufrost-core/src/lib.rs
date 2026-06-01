// nufrost-core — shared types, error handling, config parsing, and validation
// helpers for the NUFROST / HANTS / Zhu2015 reconstruction suite.
//
// Algorithms live in sibling crates; this crate provides the canonical data
// model that all three algorithms share.

pub mod config;
pub mod error;
pub mod time;
pub mod types;

pub use config::{NufrostConfig, HantsConfig, Zhu2015Config, ReconstructionConfig};
pub use error::NufrostError;
pub use time::{parse_iso8601_to_epoch_seconds, to_seconds_since_start, parse_timestamps_to_epoch_seconds, parse_to_relative_days};
pub use types::{Algorithm, TimeSeries, BandMetadata};

/// Default valid reflectance range for Sentinel-2 L2A scaled DN values.
///
/// Matches the Python pipeline helper `_mask_invalid_reflectance_values()`
/// which masks observations where `value <= valid_min` or `value >= valid_max`.
pub const SENTINEL2_VALID_MIN: f64 = 0.0;
pub const SENTINEL2_VALID_MAX: f64 = 10000.0;

/// Check whether a single reflectance value falls within the Sentinel-2 valid
/// range `(0.0, 10000.0)`.
pub fn is_valid_reflectance(value: f64, valid_min: f64, valid_max: f64) -> bool {
    value.is_finite() && value > valid_min && value < valid_max
}

/// Build a boolean valid mask from an observation slice using the
/// Sentinel-2 reflectance range `(0.0, 10000.0)`.
pub fn sentinel2_valid_mask(observations: &[f64]) -> Vec<bool> {
    observations
        .iter()
        .map(|&v| is_valid_reflectance(v, SENTINEL2_VALID_MIN, SENTINEL2_VALID_MAX))
        .collect()
}

/// Count valid entries in a mask.
pub fn count_valid(mask: &[bool]) -> usize {
    mask.iter().filter(|&&b| b).count()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn valid_reflectance_inside_range() {
        assert!(is_valid_reflectance(500.0, 0.0, 10000.0));
        assert!(is_valid_reflectance(1.0, 0.0, 10000.0));
        assert!(is_valid_reflectance(9999.0, 0.0, 10000.0));
    }

    #[test]
    fn valid_reflectance_outside_range() {
        assert!(!is_valid_reflectance(0.0, 0.0, 10000.0));
        assert!(!is_valid_reflectance(10000.0, 0.0, 10000.0));
        assert!(!is_valid_reflectance(-1.0, 0.0, 10000.0));
        assert!(!is_valid_reflectance(10001.0, 0.0, 10000.0));
    }

    #[test]
    fn valid_reflectance_nan_invalid() {
        assert!(!is_valid_reflectance(f64::NAN, 0.0, 10000.0));
        assert!(!is_valid_reflectance(f64::INFINITY, 0.0, 10000.0));
        assert!(!is_valid_reflectance(f64::NEG_INFINITY, 0.0, 10000.0));
    }

    #[test]
    fn sentinel2_mask() {
        let obs = vec![500.0, f64::NAN, 0.0, 10000.0, 3000.0, -1.0];
        let mask = sentinel2_valid_mask(&obs);
        assert_eq!(mask, vec![true, false, false, false, true, false]);
        assert_eq!(count_valid(&mask), 2);
    }

    #[test]
    fn count_valid_empty() {
        assert_eq!(count_valid(&[]), 0);
    }

    #[test]
    fn sentinel2_constants_match_python() {
        assert!((SENTINEL2_VALID_MIN - 0.0).abs() < 1e-10);
        assert!((SENTINEL2_VALID_MAX - 10000.0).abs() < 1e-10);
    }
}
