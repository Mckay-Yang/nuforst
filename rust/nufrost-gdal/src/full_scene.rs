// Full-scene Sentinel-2 band discovery.
//
// Mirrors Python `discover_location_band_stacks()` and helpers from
// `src/full_scene_reconstruction/pipeline.py:125-193`.

use std::collections::BTreeMap;
use std::path::{Path, PathBuf};

use anyhow::{Context, Result};
use regex::Regex;

// ---------------------------------------------------------------------------
// Location token helpers
// ---------------------------------------------------------------------------

/// Format lon/lat into an opaque location token matching Python
/// `_location_token` (4 decimal places).
///
/// Examples:
/// ```
/// assert_eq!(nufrost_gdal::full_scene::location_token(104.2595, 31.2170),
///            "lon104.2595_lat31.2170");
/// ```
pub fn location_token(lon: f64, lat: f64) -> String {
    format!("lon{:.4}_lat{:.4}", lon, lat)
}

/// Format lon/lat with 6 decimal places (used in VRT filenames by Python).
pub fn location_output_token(lon: f64, lat: f64) -> String {
    format!("lon{:.6}_lat{:.6}", lon, lat)
}

// ---------------------------------------------------------------------------
// Band sort key
// ---------------------------------------------------------------------------

/// Compute a sort key that orders Sentinel-2 band names numerically.
///
/// Mirrors Python `_sentinel_band_sort_key` in pipeline.py:133-137.
/// Returns `(band_number, suffix)` — e.g. `"B2"` → `(2, "")`,
/// `"B8A"` → `(8, "A")`. Unknown names get `(10_000, name)`.
pub fn sentinel_band_sort_key(name: &str) -> (u32, String) {
    let re = Regex::new(r"^B(\d+)([A-Z]?)$").unwrap();
    if let Some(caps) = re.captures(name) {
        let num: u32 = caps.get(1).unwrap().as_str().parse().unwrap_or(10_000);
        let suffix = caps.get(2).map_or(String::new(), |m| m.as_str().to_string());
        (num, suffix)
    } else {
        (10_000, name.to_string())
    }
}

/// Return band names sorted by Sentinel-2 band-order convention.
///
/// Equivalent to Python `sorted(stacks.keys(), key=_sentinel_band_sort_key)`.
pub fn sorted_band_names<'a>(stack: &'a BTreeMap<String, Vec<PathBuf>>) -> Vec<&'a str> {
    let mut names: Vec<&str> = stack.keys().map(String::as_str).collect();
    // Stable sort: secondary key is the string itself for tie-breaking (B8 vs B8A).
    names.sort_by(|a, b| {
        sentinel_band_sort_key(a)
            .cmp(&sentinel_band_sort_key(b))
            .then_with(|| a.cmp(b))
    });
    names
}

// ---------------------------------------------------------------------------
// Sentinel-2 band stack discovery
// ---------------------------------------------------------------------------

/// Discover Sentinel-2 band GeoTIFF stacks for a given `(lon, lat)`.
///
/// Scans `data_dir` for files matching the pattern
/// `COPERNICUS_S2_HARMONIZED_{band}_lon{:.4}_lat{:.4}*.tif`,
/// groups them by spectral band, and returns an ordered map.
///
/// Multiple chunks of the same band are kept as separate entries in the `Vec`
/// (VRT construction is **not** performed by this function — the Python
/// `_build_multi_file_vrt` equivalent lives elsewhere).
///
/// # Errors
///
/// Returns an error if `data_dir` does not exist or is not readable.
pub fn discover_sentinel_band_stacks(
    data_dir: &Path,
    lon: f64,
    lat: f64,
) -> Result<BTreeMap<String, Vec<PathBuf>>> {
    let token = location_token(lon, lat);

    // Band extraction pattern: COPERNICUS_S2_HARMONIZED_{band}_lon...
    let band_re = Regex::new(r"COPERNICUS_S2_HARMONIZED_(?P<band>B\d+[A-Z]?)_lon")
        .context("failed to compile S2 band regex")?;

    let mut stacks: BTreeMap<String, Vec<PathBuf>> = BTreeMap::new();

    let dir = std::fs::read_dir(data_dir)
        .with_context(|| format!("cannot read data directory: {}", data_dir.display()))?;

    for entry in dir {
        let entry = entry?;
        let path = entry.path();

        // Quick filter by filename (avoids regex on every file unnecessarily).
        if !path.is_file() {
            continue;
        }
        let fname = path
            .file_name()
            .and_then(|n| n.to_str())
            .unwrap_or("");

        if !fname.contains(&token) {
            continue;
        }

        // Extract band name from the filename.
        if let Some(caps) = band_re.captures(fname) {
            if let Some(band_match) = caps.name("band") {
                let band_name = band_match.as_str().to_string();
                stacks.entry(band_name).or_default().push(path);
            }
        }
    }

    // Sort paths within each band for deterministic ordering.
    for paths in stacks.values_mut() {
        paths.sort();
    }

    Ok(stacks)
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::Path;

    // -----------------------------------------------------------------------
    // location_token / location_output_token
    // -----------------------------------------------------------------------

    #[test]
    fn test_location_token_format() {
        assert_eq!(
            location_token(104.2595, 31.2170),
            "lon104.2595_lat31.2170"
        );
        assert_eq!(
            location_token(-100.1234, 0.0),
            "lon-100.1234_lat0.0000"
        );
    }

    #[test]
    fn test_location_output_token_format() {
        let t = location_output_token(104.2595, 31.2170);
        assert!(t.starts_with("lon104.259500_lat31.217000"), "got: {t}");
    }

    // -----------------------------------------------------------------------
    // sentinel_band_sort_key
    // -----------------------------------------------------------------------

    #[test]
    fn test_band_sort_key_simple() {
        assert_eq!(sentinel_band_sort_key("B2"), (2, "".into()));
        assert_eq!(sentinel_band_sort_key("B12"), (12, "".into()));
    }

    #[test]
    fn test_band_sort_key_b8a() {
        assert_eq!(sentinel_band_sort_key("B8A"), (8, "A".into()));
        assert_eq!(sentinel_band_sort_key("B09"), (9, "".into()));
    }

    #[test]
    fn test_band_sort_key_unknown() {
        assert_eq!(sentinel_band_sort_key("Z99"), (10_000, "Z99".into()));
        assert_eq!(sentinel_band_sort_key(""), (10_000, "".into()));
    }

    #[test]
    fn test_band_sort_ordering() {
        let mut bands = vec!["B12", "B2", "B8A", "B8", "B3", "B11", "B4"];
        bands.sort_by(|a, b| {
            sentinel_band_sort_key(a)
                .cmp(&sentinel_band_sort_key(b))
                .then_with(|| a.cmp(b))
        });
        assert_eq!(bands, vec!["B2", "B3", "B4", "B8", "B8A", "B11", "B12"]);
    }

    // -----------------------------------------------------------------------
    // discover_sentinel_band_stacks — integration test against real data
    // -----------------------------------------------------------------------

    #[test]
    fn test_discover_sentinel_bands_real_location() {
        // The test data directory relative to workspace root.
        // CI won't have this; skip gracefully if missing.
        let data_dir = Path::new("../../data/sentinel-2");
        if !data_dir.is_dir() {
            eprintln!("Skipping test: data/sentinel-2 not found");
            return;
        }

        let lon = 104.2595;
        let lat = 31.2170;

        let stacks = discover_sentinel_band_stacks(data_dir, lon, lat)
            .expect("discovery should succeed for real data");

        // Expected bands: B2, B3, B4, B8, B11, B12
        let expected: Vec<&str> = vec!["B2", "B3", "B4", "B8", "B11", "B12"];
        let found: Vec<&str> = stacks.keys().map(String::as_str).collect();

        // Verify set equality (all expected bands present, no extras).
        let found_set: std::collections::HashSet<&str> =
            found.iter().copied().collect();
        let expected_set: std::collections::HashSet<&str> =
            expected.iter().copied().collect();
        assert_eq!(
            found_set, expected_set,
            "band set mismatch: got {found_set:?}, expected {expected_set:?}"
        );

        // Verify each band has at least one path.
        for band in &expected {
            let paths = stacks.get(*band).expect("band should exist");
            assert!(!paths.is_empty(), "{band} has no files");
        }

        // Verify sorted band names match Python ordering.
        let ordered = sorted_band_names(&stacks);
        assert_eq!(
            ordered,
            vec!["B2", "B3", "B4", "B8", "B11", "B12"],
            "band ordering should match Python _sentinel_band_sort_key"
        );
    }
}
