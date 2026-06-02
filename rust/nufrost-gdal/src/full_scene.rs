// Full-scene Sentinel-2 band discovery.
//
// Mirrors Python `discover_location_band_stacks()` and helpers from
// `src/full_scene_reconstruction/pipeline.py:125-193`.

use std::collections::BTreeMap;
use std::path::{Path, PathBuf};

use anyhow::{Context, Result};
use ndarray::Array2;
use regex::Regex;

use crate::{RasterMetadata, RasterWriter, RasterReader};

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
// Full-scene timestamp extraction (always uses real GDAL band descriptions)
// ---------------------------------------------------------------------------

/// Extract relative-day timestamps and target day from a GeoTIFF/VRT path.
///
/// Opens the raster via [`RasterReader`], parses real ISO‑8601 timestamps from
/// GDAL band descriptions using [`crate::extract_timestamps_from_band_descriptions`],
/// and returns the result.  This is the canonical full‑scene timestamp entrypoint;
/// synthetic band‑index timestamps are **never** used.
///
/// Returns `(timestamps_days, target_time_day)` where `target_time_day` is
/// the last timestamp (hold‑out convention matching Python pipeline).
pub fn extract_full_scene_timestamps(
    path: &Path,
) -> anyhow::Result<(Vec<f64>, f64)> {
    let reader = RasterReader::open(path)?;
    Ok(crate::extract_timestamps_from_band_descriptions(&reader)?)
}

// ---------------------------------------------------------------------------
// Output path builders
// ---------------------------------------------------------------------------

/// Build the output path for a multi-band scene stack prediction.
///
/// Mirrors Python `build_scene_stack_output_path()` in pipeline.py:322-332.
///
/// Format:
/// `{output_root}/{source_name}_recon/{lon:.4}_{lat:.4}/[{method}]_{source_name}_{location_output_token}_{safe_time}_{suffix}.tif`
pub fn build_scene_stack_output_path(
    output_root: &Path,
    method_name: &str,
    source_name: &str,
    lon: f64,
    lat: f64,
    target_time: &str,
    suffix: &str,
) -> PathBuf {
    let safe_time = target_time.replace(':', "-");
    let loc_token = location_output_token(lon, lat);
    output_root
        .join(format!("{source_name}_recon"))
        .join(format!("{lon:.4}_{lat:.4}"))
        .join(format!(
            "[{method_name}]_{source_name}_{loc_token}_{safe_time}_{suffix}.tif"
        ))
}

/// Build the output path for a ground truth GeoTIFF.
///
/// Mirrors Python `build_ground_truth_output_path()` in pipeline.py:311-319.
///
/// Format:
/// `{output_root}/{source_name}_recon/{lon:.4}_{lat:.4}/[ground_truth]_{source_name}_{location_output_token}_{safe_time}.tif`
pub fn build_ground_truth_output_path(
    output_root: &Path,
    source_name: &str,
    lon: f64,
    lat: f64,
    target_time: &str,
) -> PathBuf {
    let safe_time = target_time.replace(':', "-");
    let loc_token = location_output_token(lon, lat);
    output_root
        .join(format!("{source_name}_recon"))
        .join(format!("{lon:.4}_{lat:.4}"))
        .join(format!(
            "[ground_truth]_{source_name}_{loc_token}_{safe_time}.tif"
        ))
}

// ---------------------------------------------------------------------------
// Multiband writer
// ---------------------------------------------------------------------------

/// Write per-band prediction arrays as a multi-band GeoTIFF.
///
/// Mirrors Python `write_band_stack()` in pipeline.py:621-631:
/// 1. Stack arrays in `ordered_bands` order → Float32
/// 2. Write as GTiff using `meta` for CRS, transform, and nodata
/// 3. Set each band's description to the band name
pub fn write_band_stack(
    output_path: &Path,
    arrays_by_band: &BTreeMap<String, Array2<f64>>,
    ordered_bands: &[String],
    meta: &RasterMetadata,
) -> Result<()> {
    if ordered_bands.is_empty() {
        anyhow::bail!("ordered_bands must not be empty");
    }

    let first_band = &ordered_bands[0];
    let first_array = arrays_by_band
        .get(first_band)
        .with_context(|| format!("band {first_band} not found in arrays_by_band"))?;
    let (rows, cols) = first_array.dim();

    if let Some(parent) = output_path.parent() {
        std::fs::create_dir_all(parent)
            .with_context(|| format!("creating parent dirs for {}", output_path.display()))?;
    }

    let mut writer = RasterWriter::create(
        output_path,
        rows,
        cols,
        ordered_bands.len(),
        &meta.geo_transform,
        meta.crs_wkt.as_deref(),
        meta.nodata,
    )?;

    for (idx, band_name) in ordered_bands.iter().enumerate() {
        let band_idx = idx + 1;
        let array = arrays_by_band
            .get(band_name)
            .with_context(|| format!("band {band_name} not found in arrays_by_band"))?;
        writer.write_band(band_idx, array)?;
        writer.set_band_description(band_idx, band_name)?;
    }

    writer.flush()?;
    Ok(())
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

    // -----------------------------------------------------------------------
    // Output path builders
    // -----------------------------------------------------------------------

    #[test]
    fn test_scene_stack_output_path_format() {
        let root = Path::new("/tmp/output");
        let path = build_scene_stack_output_path(
            root,
            "NUFROST",
            "sentinel-2",
            104.2595,
            31.2170,
            "2025-05-20T03:52:01",
            "prediction",
        );
        let expected = "/tmp/output/sentinel-2_recon/104.2595_31.2170/[NUFROST]_sentinel-2_lon104.259500_lat31.217000_2025-05-20T03-52-01_prediction.tif";
        assert_eq!(path.to_str().unwrap(), expected);
    }

    #[test]
    fn test_ground_truth_output_path_format() {
        let root = Path::new("/tmp/output");
        let path = build_ground_truth_output_path(
            root,
            "sentinel-2",
            104.2595,
            31.2170,
            "2025-05-20T03:52:01",
        );
        let expected = "/tmp/output/sentinel-2_recon/104.2595_31.2170/[ground_truth]_sentinel-2_lon104.259500_lat31.217000_2025-05-20T03-52-01.tif";
        assert_eq!(path.to_str().unwrap(), expected);
    }

    #[test]
    fn test_output_path_safe_time_colon_replacement() {
        let root = Path::new("/out");
        let path = build_scene_stack_output_path(
            root, "HANTS", "hls", 10.0, 20.0,
            "2024-01-15T12:30:45", "eval",
        );
        let s = path.to_str().unwrap();
        assert!(s.contains("_2024-01-15T12-30-45_"));
        assert!(!s.contains(":12:"));
        assert!(!s.contains(":30:"));
        assert!(!s.contains(":45"));
    }

    // -----------------------------------------------------------------------
    // write_band_stack
    // -----------------------------------------------------------------------

    /// Helper: create a simple 2×3 ndarray filled with a constant value.
    fn make_array2(rows: usize, cols: usize, fill: f64) -> Array2<f64> {
        Array2::from_elem((rows, cols), fill)
    }

    /// Helper: create default RasterMetadata for testing.
    fn test_meta() -> RasterMetadata {
        RasterMetadata {
            geo_transform: [0.0, 1.0, 0.0, 0.0, 0.0, -1.0],
            crs_wkt: None,
            nodata: None,
        }
    }

    #[test]
    fn test_write_band_stack_creates_multiband() {
        let tmp = std::env::temp_dir().join("nufrost_test_stack.tif");

        let mut arrays = BTreeMap::new();
        arrays.insert("B2".to_string(), make_array2(3, 4, 100.0));
        arrays.insert("B3".to_string(), make_array2(3, 4, 200.0));
        arrays.insert("B4".to_string(), make_array2(3, 4, 300.0));

        let ordered: Vec<String> = vec!["B2", "B3", "B4"]
            .into_iter()
            .map(String::from)
            .collect();
        let meta = test_meta();

        write_band_stack(&tmp, &arrays, &ordered, &meta)
            .expect("write_band_stack should succeed");

        // Verify with GDAL.
        use gdal::Metadata;
        let ds = gdal::Dataset::open(&tmp).expect("re-open should succeed");
        assert_eq!(ds.raster_count(), 3, "band count");

        // Check band descriptions.
        for (i, name) in ["B2", "B3", "B4"].iter().enumerate() {
            let band = ds.rasterband(i + 1).unwrap();
            let desc = band.description().unwrap_or_default();
            assert_eq!(desc, *name, "band {i} description mismatch");
        }

        // Check spatial dimensions.
        let (cols, rows) = ds.raster_size();
        assert_eq!(rows, 3, "rows");
        assert_eq!(cols, 4, "cols");

        // Clean up.
        let _ = std::fs::remove_file(&tmp);
    }

    #[test]
    fn test_write_band_stack_empty_bands_panics() {
        let tmp = std::env::temp_dir().join("nufrost_test_empty.tif");
        let arrays: BTreeMap<String, Array2<f64>> = BTreeMap::new();
        let ordered: Vec<String> = vec![];
        let meta = test_meta();

        let result = write_band_stack(&tmp, &arrays, &ordered, &meta);
        assert!(result.is_err(), "empty ordered_bands should error");
    }

    // -------------------------------------------------------------------
    // Timestamp extraction & duplicate collapsing
    // -------------------------------------------------------------------

    #[test]
    fn test_find_timestamp_substring_sentinel2() {
        assert_eq!(
            nufrost_core::find_timestamp_substring("20151227T035152_20151227T035506_T48RVV_B2"),
            Some("20151227T035152")
        );
        assert_eq!(
            nufrost_core::find_timestamp_substring("20200101T045859_20200101T045859_T48RVV_B2"),
            Some("20200101T045859")
        );
    }

    #[test]
    fn test_find_timestamp_substring_no_match() {
        assert_eq!(nufrost_core::find_timestamp_substring("no_timestamp_here"), None);
        assert_eq!(nufrost_core::find_timestamp_substring(""), None);
    }

    #[test]
    fn test_extract_timestamps_from_real_file() {
        let data_dir = Path::new("../../data/sentinel-2");
        if !data_dir.is_dir() {
            eprintln!("Skipping test: data/sentinel-2 not found");
            return;
        }
        let path = data_dir.join("COPERNICUS_S2_HARMONIZED_B2_lon104.2595_lat31.2170.tif");
        if !path.is_file() {
            eprintln!("Skipping test: {path:?} not found");
            return;
        }

        let reader = crate::RasterReader::open(&path).expect("should open real file");
        let (days, target) =
            crate::extract_timestamps_from_band_descriptions(&reader)
                .expect("should parse timestamps");

        let n = reader.band_count();
        assert_eq!(days.len(), n, "days count should match band count");
        assert!(n >= 202, "expected >= 202 bands, got {n}");

        // Timestamps must be non-decreasing (days since first).
        // Allow tiny negative jitter from float arithmetic by checking
        // that adjacent diffs are >= -1e-6.
        for w in days.windows(2) {
            assert!(w[1] - w[0] >= -1e-6, "timestamps must be non-decreasing: {} -> {}", w[0], w[1]);
        }

        // First day should be ~0
        assert!(days[0].abs() < 1.0, "first day should be near 0, got {}", days[0]);

        // Target should be the last element
        assert!((target - days[days.len() - 1]).abs() < 1e-10);
    }

    #[test]
    fn test_extract_full_scene_timestamps_smoke() {
        let data_dir = Path::new("../../data/sentinel-2");
        if !data_dir.is_dir() {
            eprintln!("Skipping test: data/sentinel-2 not found");
            return;
        }
        let path = data_dir.join("COPERNICUS_S2_HARMONIZED_B2_lon104.2595_lat31.2170.tif");
        if !path.is_file() {
            eprintln!("Skipping test: {path:?} not found");
            return;
        }

        let (days, target) =
            extract_full_scene_timestamps(&path).expect("full_scene path should work");
        assert!(!days.is_empty());
        assert!(target > 0.0);
    }

    #[test]
    fn test_collapse_duplicate_timestamps_no_dups() {
        let cube = ndarray::Array3::from_shape_vec(
            (3, 2, 2),
            vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0],
        )
        .unwrap();
        let timestamps: Vec<String> = vec!["A", "B", "C"].into_iter().map(String::from).collect();

        let (deduped, deduped_ts) =
            crate::collapse_duplicate_timestamps(&cube, &timestamps);

        assert_eq!(deduped.shape(), &[3, 2, 2]);
        assert_eq!(deduped_ts, vec!["A", "B", "C"]);
        // Values should be unchanged (identity transform for unique timestamps)
        assert!((deduped[[0, 0, 0]] - 1.0).abs() < 1e-10);
        assert!((deduped[[2, 1, 1]] - 12.0).abs() < 1e-10);
    }

    #[test]
    fn test_collapse_duplicate_timestamps_with_dups() {
        let cube = ndarray::Array3::from_shape_vec(
            (4, 1, 2),
            vec![10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0, 80.0],
        )
        .unwrap();
        // Bands 0 and 2 share timestamp "A", bands 1 and 3 share "B"
        let timestamps: Vec<String> =
            vec!["A", "B", "A", "B"].into_iter().map(String::from).collect();

        let (deduped, deduped_ts) =
            crate::collapse_duplicate_timestamps(&cube, &timestamps);

        assert_eq!(deduped.shape(), &[2, 1, 2], "should collapse to 2 bands");
        assert_eq!(deduped_ts, vec!["A", "B"], "timestamps should be deduped");

        // Band A: nanmean of bands 0 (10,20) and 2 (50,60) = (30,40)
        assert!((deduped[[0, 0, 0]] - 30.0).abs() < 1e-10);
        assert!((deduped[[0, 0, 1]] - 40.0).abs() < 1e-10);

        // Band B: nanmean of bands 1 (30,40) and 3 (70,80) = (50,60)
        assert!((deduped[[1, 0, 0]] - 50.0).abs() < 1e-10);
        assert!((deduped[[1, 0, 1]] - 60.0).abs() < 1e-10);
    }

    #[test]
    fn test_collapse_duplicate_with_nan_pixels() {
        let mut cube = ndarray::Array3::from_shape_vec(
            (3, 1, 2),
            vec![
                f64::NAN, 100.0,  // band 0: "A"
                50.0, f64::NAN,    // band 1: "A" (same timestamp)
                1.0, 2.0           // band 2: "B"
            ],
        )
        .unwrap();
        cube[[1, 0, 0]] = 50.0; // already set
        cube[[1, 0, 1]] = f64::NAN;

        let timestamps: Vec<String> =
            vec!["A", "A", "B"].into_iter().map(String::from).collect();

        let (deduped, _) = crate::collapse_duplicate_timestamps(&cube, &timestamps);

        // Band A: nanmean of (NaN, 100) and (50, NaN) = (50, 100)
        assert!((deduped[[0, 0, 0]] - 50.0).abs() < 1e-10, "nanmean(50, NaN) = 50");
        assert!((deduped[[0, 0, 1]] - 100.0).abs() < 1e-10, "nanmean(100, NaN) = 100");

        // Band B: unchanged = (1, 2)
        assert!((deduped[[1, 0, 0]] - 1.0).abs() < 1e-10);
        assert!((deduped[[1, 0, 1]] - 2.0).abs() < 1e-10);
    }
}
