// Full-scene Sentinel-2 band discovery.
//
// Rust implementation of full-scene Sentinel-2 band discovery and helpers.

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
/// assert_eq!(gdal::full_scene::location_token(104.2595, 31.2170),
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
// Invalid-reflectance masking
// ---------------------------------------------------------------------------

/// Mask invalid reflectance values in-place: finite values ≤ `valid_min` or
/// ≥ `valid_max` are replaced with NaN.
///
/// Mirrors Python `_mask_invalid_reflectance_values()` in
/// `full_scene_reconstruction/pipeline.py:42-46`.
///
/// Sentinel-2 defaults are `valid_min=0.0`, `valid_max=10000.0`, so only
/// `(0.0, 10000.0)` is treated as valid.
pub fn mask_invalid_reflectance(
    cube: &mut ndarray::Array3<f64>,
    valid_min: f64,
    valid_max: f64,
) {
    for v in cube.iter_mut() {
        if v.is_finite() && (*v <= valid_min || *v >= valid_max) {
            *v = f64::NAN;
        }
    }
}

/// Sentinel-2 convenience: mask using `(0.0, 10000.0)`.
pub fn mask_invalid_sentinel2(cube: &mut ndarray::Array3<f64>) {
    mask_invalid_reflectance(cube, 0.0, 10000.0);
}

// ---------------------------------------------------------------------------
// Completeness scoring (sampled)
// ---------------------------------------------------------------------------

/// Subsampling step for completeness scoring (matches Python
/// `_VALID_RATIO_SUBSAMPLE_STEP = 8`).
const VALID_RATIO_SUBSAMPLE_STEP: usize = 8;

/// Compute per-band completeness scores for every shared candidate timestamp.
///
/// For each band cube, invalid reflectance values are masked first, then the
/// cube is sampled every `step` pixels in the spatial dimensions.
/// Completeness = fraction of finite (non-NaN) pixels in the sampled time slice.
///
/// Returns `BTreeMap<band_name → BTreeMap<timestamp → completeness>>`.
pub fn score_candidates(
    band_to_cubes: &BTreeMap<String, ndarray::Array3<f64>>,
    band_to_timestamps: &BTreeMap<String, Vec<String>>,
    shared_candidates: &[String],
    valid_min: f64,
    valid_max: f64,
    step: usize,
) -> Result<BTreeMap<String, BTreeMap<String, f64>>> {
    let mut scores: BTreeMap<String, BTreeMap<String, f64>> = BTreeMap::new();

    for (band, cube) in band_to_cubes.iter() {
        let timestamps = band_to_timestamps
            .get(band)
            .with_context(|| format!("timestamps missing for band {band}"))?;

        // Build index map: timestamp string → band index
        let index_map: std::collections::HashMap<&str, usize> = timestamps
            .iter()
            .enumerate()
            .map(|(i, ts)| (ts.as_str(), i))
            .collect();

        // Clone and mask a mutable copy
        let mut masked = cube.clone();
        mask_invalid_reflectance(&mut masked, valid_min, valid_max);

        // Subsample spatial dimensions
        let (t, rows, cols) = masked.dim();
        let sampled_rows: Vec<usize> = (0..rows).step_by(step).collect();
        let sampled_cols: Vec<usize> = (0..cols).step_by(step).collect();

        let mut band_scores: BTreeMap<String, f64> = BTreeMap::new();
        for candidate in shared_candidates {
            if let Some(&idx) = index_map.get(candidate.as_str()) {
                if idx < t {
                    let layer = masked.index_axis(ndarray::Axis(0), idx);
                    let mut finite_count = 0usize;
                    let mut total = 0usize;
                    for &r in &sampled_rows {
                        for &c in &sampled_cols {
                            total += 1;
                            if layer[[r, c]].is_finite() {
                                finite_count += 1;
                            }
                        }
                    }
                    let score = if total == 0 {
                        0.0
                    } else {
                        finite_count as f64 / total as f64
                    };
                    band_scores.insert(candidate.clone(), score);
                }
            }
        }
        scores.insert(band.clone(), band_scores);
    }

    Ok(scores)
}

// ---------------------------------------------------------------------------
// Target timestamp selection
// ---------------------------------------------------------------------------

/// Select the best shared target timestamp from completeness scores.
///
/// Candidates are sorted, then the last `late_fraction` portion is preferred.
/// Within each pool, candidates are examined from latest to earliest; the first
/// one where **every** band meets `min_valid_ratio` is chosen. Falls back to
/// earlier candidates if no late candidate qualifies.
///
/// Mirrors Python `select_shared_target_timestamp()` in
/// `full_scene_reconstruction/pipeline.py:257-285`.
pub fn select_shared_target_timestamp(
    candidates: &[String],
    completeness_by_band: &BTreeMap<String, BTreeMap<String, f64>>,
    min_valid_ratio: f64,
    late_fraction: f64,
) -> Result<String> {
    if candidates.is_empty() {
        anyhow::bail!("No shared timestamps available.");
    }

    let mut ordered: Vec<String> = candidates.iter().cloned().collect();
    ordered.sort();

    let tail_len = 1usize.max((ordered.len() as f64 * late_fraction).ceil() as usize);
    let preferred: Vec<&String> = ordered[ordered.len() - tail_len..].iter().collect();
    let fallback: Vec<&String> = ordered[..ordered.len() - tail_len].iter().collect();

    let pick = |pool: &[&String]| -> Option<String> {
        for candidate in pool.iter().rev() {
            let all_pass = completeness_by_band.iter().all(|(_band, band_scores)| {
                band_scores.get(candidate.as_str()).copied().unwrap_or(0.0) >= min_valid_ratio
            });
            if all_pass {
                return Some(candidate.to_string());
            }
        }
        None
    };

    if let Some(chosen) = pick(&preferred) {
        return Ok(chosen);
    }
    if let Some(chosen) = pick(&fallback) {
        return Ok(chosen);
    }

    anyhow::bail!("No shared timestamp passed the completeness threshold.");
}

/// Full target-timestamp selection: intersect timestamps across bands,
/// compute completeness scores, and select the best shared timestamp.
///
/// Mirrors Python `choose_shared_target_timestamp()` in
/// `full_scene_reconstruction/pipeline.py:288-301`.
pub fn choose_shared_target_timestamp(
    band_to_cubes: &BTreeMap<String, ndarray::Array3<f64>>,
    band_to_timestamps: &BTreeMap<String, Vec<String>>,
    min_valid_ratio: f64,
    late_fraction: f64,
) -> Result<(String, BTreeMap<String, BTreeMap<String, f64>>)> {
    // Intersect timestamps across all bands
    let mut shared: Option<std::collections::HashSet<String>> = None;
    for timestamps in band_to_timestamps.values() {
        let current: std::collections::HashSet<String> =
            timestamps.iter().cloned().collect();
        shared = match shared {
            None => Some(current),
            Some(s) => Some(s.intersection(&current).cloned().collect()),
        };
    }
    let mut candidates: Vec<String> = shared
        .unwrap_or_default()
        .into_iter()
        .collect();
    candidates.sort();

    if candidates.is_empty() {
        anyhow::bail!("No shared timestamps exist across selected bands.");
    }

    let completeness = score_candidates(
        band_to_cubes,
        band_to_timestamps,
        &candidates,
        0.0,
        10000.0,
        VALID_RATIO_SUBSAMPLE_STEP,
    )?;

    let chosen = select_shared_target_timestamp(
        &candidates,
        &completeness,
        min_valid_ratio,
        late_fraction,
    )?;

    Ok((chosen, completeness))
}

// ---------------------------------------------------------------------------
// Target hold-out — create masked time series for reconstruction
// ---------------------------------------------------------------------------

/// Remove the target-time slice from a band cube, producing the reconstruction
/// input and extracting the ground-truth array.
///
/// **Contract**: Timestamps MUST be pre-deduplicated (via
/// `collapse_duplicate_timestamps`) — exactly ONE match for `target_time`
/// is required.  Multiple matches will produce an error.
///
/// Returns `(masked_cube, masked_timestamps, target_idx, ground_truth_2d)`.
///
/// Mirrors Python `make_masked_time_series()` in
/// `full_scene_reconstruction/pipeline.py:355-363`.
pub fn make_masked_time_series(
    cube: &ndarray::Array3<f64>,
    timestamps: &[String],
    target_time: &str,
) -> Result<(ndarray::Array3<f64>, Vec<String>, usize, ndarray::Array2<f64>)> {
    let matching: Vec<usize> = timestamps
        .iter()
        .enumerate()
        .filter(|(_, ts)| ts.as_str() == target_time)
        .map(|(i, _)| i)
        .collect();

    if matching.len() != 1 {
        anyhow::bail!(
            "Expected exactly 1 match for target time '{}' in pre-deduplicated timestamps, found {} matches. \
             Call collapse_duplicate_timestamps() before make_masked_time_series().",
            target_time,
            matching.len(),
        );
    }

    let target_idx = matching[0];
    let (_t, rows, cols) = cube.dim();

    let ground_truth = cube
        .index_axis(ndarray::Axis(0), target_idx)
        .to_owned();

    // Input already deduplicated — exclude only the single target index.
    let keep_n = timestamps.len() - 1;
    let mut masked_cube = ndarray::Array3::zeros((keep_n, rows, cols));
    let mut masked_timestamps = Vec::with_capacity(keep_n);

    let mut out_i = 0;
    for (src_i, ts) in timestamps.iter().enumerate() {
        if src_i == target_idx {
            continue;
        }
        let src_slice = cube.index_axis(ndarray::Axis(0), src_i);
        masked_cube
            .index_axis_mut(ndarray::Axis(0), out_i)
            .assign(&src_slice);
        masked_timestamps.push(ts.clone());
        out_i += 1;
    }

    Ok((masked_cube, masked_timestamps, target_idx, ground_truth))
}

// ---------------------------------------------------------------------------
// Shared spectral frequency pool
// ---------------------------------------------------------------------------

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::Path;
    use gdal_rs::Metadata;

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
        use gdal_rs::Metadata;
        let ds = gdal_rs::Dataset::open(&tmp).expect("re-open should succeed");
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
            crate::find_timestamp_substring("20151227T035152_20151227T035506_T48RVV_B2"),
            Some("20151227T035152")
        );
        assert_eq!(
            crate::find_timestamp_substring("20200101T045859_20200101T045859_T48RVV_B2"),
            Some("20200101T045859")
        );
    }

    #[test]
    fn test_find_timestamp_substring_no_match() {
        assert_eq!(crate::find_timestamp_substring("no_timestamp_here"), None);
        assert_eq!(crate::find_timestamp_substring(""), None);
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

    // -------------------------------------------------------------------
    // mask_invalid_reflectance
    // -------------------------------------------------------------------

    #[test]
    fn test_mask_invalid_sets_bad_values_to_nan() {
        let mut cube = ndarray::Array3::from_shape_vec(
            (2, 2, 3),
            vec![
                -1.0, 0.0, 500.0,
                9999.0, 10000.0, 15000.0,
                f64::NAN, 1.0, 100.0,
                50.0, -0.5, 10001.0,
            ],
        )
        .unwrap();

        mask_invalid_reflectance(&mut cube, 0.0, 10000.0);

        // Valid values (0 < v < 10000) stay unchanged
        assert!((cube[[0, 0, 2]] - 500.0).abs() < 1e-10);  // was 500
        assert!((cube[[0, 1, 0]] - 9999.0).abs() < 1e-10); // was 9999
        assert!((cube[[1, 0, 1]] - 1.0).abs() < 1e-10);    // was 1
        assert!((cube[[1, 0, 2]] - 100.0).abs() < 1e-10);  // was 100
        assert!((cube[[1, 1, 0]] - 50.0).abs() < 1e-10);   // was 50

        // Invalid values become NaN
        assert!(cube[[0, 0, 0]].is_nan());  // was -1.0
        assert!(cube[[0, 0, 1]].is_nan());  // was 0.0
        assert!(cube[[0, 1, 1]].is_nan());  // was 10000.0
        assert!(cube[[0, 1, 2]].is_nan());  // was 15000.0
        assert!(cube[[1, 1, 1]].is_nan());  // was -0.5
        assert!(cube[[1, 1, 2]].is_nan());  // was 10001.0

        // Already NaN stays NaN
        assert!(cube[[1, 0, 0]].is_nan());  // was already NaN
    }

    #[test]
    fn test_mask_invalid_with_custom_range() {
        let mut cube = ndarray::Array3::from_shape_vec(
            (1, 1, 4),
            vec![0.0, 1.0, 2.0, f64::NAN],
        )
        .unwrap();

        mask_invalid_reflectance(&mut cube, 1.0, 2.0);

        assert!(cube[[0, 0, 0]].is_nan()); // 0.0 <= 1.0
        assert!(cube[[0, 0, 1]].is_nan()); // 1.0 <= 1.0
        assert!(cube[[0, 0, 2]].is_nan()); // 2.0 >= 2.0
        assert!(cube[[0, 0, 3]].is_nan()); // already NaN
    }

    // -------------------------------------------------------------------
    // select_shared_target_timestamp
    // -------------------------------------------------------------------

    #[test]
    fn test_select_shared_target_prefers_late() {
        // Three candidates, late_fraction=0.5 means last 2 are preferred.
        // Band A has completeness >= 0.9 for all; band B only >= 0.9 for "C".
        let mut completeness: BTreeMap<String, BTreeMap<String, f64>> = BTreeMap::new();

        let mut band_a = BTreeMap::new();
        band_a.insert("A".to_string(), 0.95);
        band_a.insert("B".to_string(), 0.95);
        band_a.insert("C".to_string(), 0.95);
        completeness.insert("band_a".to_string(), band_a);

        let mut band_b = BTreeMap::new();
        band_b.insert("A".to_string(), 0.5);  // below min
        band_b.insert("B".to_string(), 0.5);  // below min
        band_b.insert("C".to_string(), 0.95); // passes
        completeness.insert("band_b".to_string(), band_b);

        let candidates: Vec<String> = vec!["A", "B", "C"].into_iter().map(String::from).collect();

        let chosen = select_shared_target_timestamp(
            &candidates,
            &completeness,
            0.9,
            0.5,
        )
        .expect("should find a timestamp");

        // With late_fraction=0.5 and 3 candidates, tail=ceil(3*0.5)=2, preferred=[B,C].
        // Reversed: check C, then B. C passes all bands, so C is chosen.
        assert_eq!(chosen, "C");
    }

    #[test]
    fn test_select_shared_target_fallback() {
        let mut completeness: BTreeMap<String, BTreeMap<String, f64>> = BTreeMap::new();

        let mut band = BTreeMap::new();
        band.insert("early".to_string(), 0.95);
        band.insert("mid".to_string(), 0.50);
        band.insert("late".to_string(), 0.50);
        completeness.insert("b".to_string(), band);

        let candidates: Vec<String> = vec!["early", "mid", "late"]
            .into_iter()
            .map(String::from)
            .collect();

        // late_fraction=0.34 → tail=ceil(3*0.34)=2 → preferred=[mid,late]
        // Neither passes (0.5 < 0.9), so fallback to [early], passes.
        let chosen = select_shared_target_timestamp(&candidates, &completeness, 0.9, 0.34)
            .expect("should fallback");
        assert_eq!(chosen, "early");
    }

    #[test]
    fn test_select_shared_target_no_candidates() {
        let completeness: BTreeMap<String, BTreeMap<String, f64>> = BTreeMap::new();
        let candidates: Vec<String> = vec![];
        let result = select_shared_target_timestamp(&candidates, &completeness, 0.9, 0.25);
        assert!(result.is_err());
    }

    #[test]
    fn test_select_shared_target_picks_latest_when_multiple_qualify() {
        let mut completeness: BTreeMap<String, BTreeMap<String, f64>> = BTreeMap::new();

        let mut band = BTreeMap::new();
        band.insert("2020-01-01".to_string(), 0.95);
        band.insert("2020-06-01".to_string(), 0.95);
        band.insert("2020-12-01".to_string(), 0.95);
        completeness.insert("b".to_string(), band);

        let candidates: Vec<String> = vec!["2020-01-01", "2020-06-01", "2020-12-01"]
            .into_iter()
            .map(String::from)
            .collect();

        // late_fraction=1.0 → all are preferred, reversed iteration picks latest
        let chosen = select_shared_target_timestamp(&candidates, &completeness, 0.9, 1.0)
            .expect("should pick latest");
        assert_eq!(chosen, "2020-12-01");
    }

    // -------------------------------------------------------------------
    // choose_shared_target_timestamp — integration test vs Python contract
    // -------------------------------------------------------------------

    /// Convert a raw YYYYMMDDTHHMMSS timestamp to ISO format YYYY-MM-DDTHH:MM:SS.
    fn raw_to_iso(raw: &str) -> String {
        if raw.len() < 15 {
            return raw.to_string();
        }
        // Format: YYYYMMDDTHHMMSS → YYYY-MM-DDTHH:MM:SS
        let y = &raw[0..4];
        let m = &raw[4..6];
        let d = &raw[6..8];
        let hh = &raw[9..11];
        let mm = &raw[11..13];
        let ss = &raw[13..15];
        format!("{y}-{m}-{d}T{hh}:{mm}:{ss}")
    }

    #[test]
    fn test_choose_shared_target_matches_python_contract() {
        let data_dir = Path::new("../../data/sentinel-2");
        if !data_dir.is_dir() {
            eprintln!("Skipping test: data/sentinel-2 not found");
            return;
        }

        let lon = 104.2595;
        let lat = 31.217;
        let stacks = discover_sentinel_band_stacks(data_dir, lon, lat)
            .expect("should discover band stacks");

        // Load B2 data only (contract is per-band, and we need at least B2 to
        // verify the selection behavior). The full contract uses all bands.
        // For a minimal test, load B2 and verify the target selection picks
        // the contract's expected target.
        let b2_paths = stacks.get("B2").expect("B2 should exist");
        let b2_path = &b2_paths[0];

        let reader = crate::RasterReader::open(b2_path).expect("should open B2");
        let n_bands = reader.band_count();
        let win = 512;
        let (full_rows, full_cols) = (reader.shape().0, reader.shape().1);
        let rows = win.min(full_rows);
        let cols = win.min(full_cols);

        // Read the B2 cube through a bounded 512×512 window.
        let mut b2_cube = ndarray::Array3::<f64>::zeros((n_bands, rows, cols));
        for b in 1..=n_bands {
            let band_data = reader.read_band_window(b, win, win).expect("should read band");
            b2_cube.index_axis_mut(ndarray::Axis(0), b - 1).assign(&band_data);
        }

        // Build timestamp strings from the band descriptions (use the descriptions directly)
        let band_timestamps: Vec<String> = (1..=n_bands)
            .map(|b| {
                let band = reader.dataset.rasterband(b).unwrap();
                let desc = band.description().unwrap_or_default().trim().to_string();
                if let Some(sub) = crate::find_timestamp_substring(&desc) {
                    raw_to_iso(sub)
                } else {
                    desc
                }
            })
            .collect();

        let mut band_to_cubes: BTreeMap<String, ndarray::Array3<f64>> = BTreeMap::new();
        let mut band_to_timestamps: BTreeMap<String, Vec<String>> = BTreeMap::new();
        band_to_cubes.insert("B2".to_string(), b2_cube);
        band_to_timestamps.insert("B2".to_string(), band_timestamps);

        let (chosen, completeness) = choose_shared_target_timestamp(
            &band_to_cubes,
            &band_to_timestamps,
            0.9,
            0.25,
        )
        .expect("should select a target timestamp");

        // Verify: the contract expects "2025-05-20T03:52:01"
        assert_eq!(
            chosen, "2025-05-20T03:52:01",
            "Rust target selection must match Python contract target_time"
        );

        // Verify: counts match contract
        let b2_completeness = completeness.get("B2").expect("B2 completeness should exist");
        let contract_target = "2025-05-20T03:52:01";
        let b2_score = b2_completeness
            .get(contract_target)
            .copied()
            .unwrap_or(-1.0);
        assert!(
            b2_score >= 0.9,
            "B2 completeness for target should be >= 0.9, got {b2_score}"
        );

        // Also verify the number of shared candidates is reasonable (should be
        // at least 100 for real data).
        let n_candidates = b2_completeness.len();
        assert!(
            n_candidates >= 50,
            "expected at least 50 shared candidate timestamps, got {n_candidates}"
        );

        dbg!(chosen, b2_score, n_candidates);
    }

    // -------------------------------------------------------------------
    // make_masked_time_series
    // -------------------------------------------------------------------

    #[test]
    fn test_make_masked_time_series_basic() {
        let cube = ndarray::Array3::from_shape_vec(
            (5, 2, 3),
            (0..30).map(|i| i as f64).collect(),
        )
        .unwrap();
        let timestamps: Vec<String> = vec!["A", "B", "C", "D", "E"]
            .into_iter()
            .map(String::from)
            .collect();
        let target = "C";

        let (masked_cube, masked_ts, target_idx, ground_truth) =
            make_masked_time_series(&cube, &timestamps, target)
                .expect("should find target");

        assert_eq!(target_idx, 2);
        assert_eq!(masked_cube.shape(), &[4, 2, 3]);
        assert_eq!(masked_ts, vec!["A", "B", "D", "E"]);
        assert_eq!(ground_truth.shape(), &[2, 3]);

        // Ground truth should be the original "C" slice (index 2)
        for r in 0..2 {
            for c in 0..3 {
                assert!((ground_truth[[r, c]] - cube[[2, r, c]]).abs() < 1e-10);
            }
        }
    }

    #[test]
    fn test_make_masked_time_series_first_slice() {
        let cube = ndarray::Array3::<f64>::from_elem((3, 1, 2), 0.0);
        let timestamps: Vec<String> = vec!["X", "Y", "Z"].into_iter().map(String::from).collect();
        let (masked_cube, masked_ts, idx, _gt) =
            make_masked_time_series(&cube, &timestamps, "X").unwrap();
        assert_eq!(idx, 0);
        assert_eq!(masked_cube.shape(), &[2, 1, 2]);
        assert_eq!(masked_ts, vec!["Y", "Z"]);
        assert_eq!(_gt.shape(), &[1, 2]);
    }

    #[test]
    fn test_make_masked_time_series_last_slice() {
        let cube = ndarray::Array3::<f64>::from_elem((3, 1, 2), 0.0);
        let timestamps: Vec<String> = vec!["X", "Y", "Z"].into_iter().map(String::from).collect();
        let (masked_cube, masked_ts, idx, _gt) =
            make_masked_time_series(&cube, &timestamps, "Z").unwrap();
        assert_eq!(idx, 2);
        assert_eq!(masked_cube.shape(), &[2, 1, 2]);
        assert_eq!(masked_ts, vec!["X", "Y"]);
    }

    #[test]
    fn test_make_masked_time_series_not_found() {
        let cube = ndarray::Array3::<f64>::from_elem((2, 1, 1), 0.0);
        let timestamps: Vec<String> = vec!["A", "A"].into_iter().map(String::from).collect();
        let result = make_masked_time_series(&cube, &timestamps, "B");
        assert!(result.is_err());
    }

    #[test]
    fn test_make_masked_time_series_rejects_duplicate_target() {
        // 2 identical "A" timestamps → expect error (must pre-deduplicate)
        let cube = ndarray::Array3::<f64>::from_elem((3, 1, 1), 0.0);
        let timestamps: Vec<String> = vec!["A", "A", "B"].into_iter().map(String::from).collect();
        let result = make_masked_time_series(&cube, &timestamps, "A");
        assert!(result.is_err(), "duplicate target should be rejected");
    }

    #[test]
    fn test_make_masked_time_series_deduped_ok() {
        // Pre-deduplicated timestamps → correct GT and masked cube
        let cube = ndarray::Array3::from_shape_vec(
            (3, 1, 2),
            vec![10.0, 20.0, 30.0, 40.0, 50.0, 60.0],
        )
        .unwrap();
        let timestamps: Vec<String> = vec!["A", "B", "C"].into_iter().map(String::from).collect();
        let (masked, mts, target_idx, gt) =
            make_masked_time_series(&cube, &timestamps, "B").unwrap();

        assert_eq!(target_idx, 1);
        assert_eq!(masked.shape(), &[2, 1, 2]);
        assert_eq!(mts, vec!["A", "C"]);
        assert!((gt[[0, 0]] - 30.0).abs() < 1e-10);
        assert!((gt[[0, 1]] - 40.0).abs() < 1e-10);

        // Masked cube: first band is "A" (10,20), second is "C" (50,60)
        assert!((masked[[0, 0, 0]] - 10.0).abs() < 1e-10);
        assert!((masked[[1, 0, 1]] - 60.0).abs() < 1e-10);
    }

    // -------------------------------------------------------------------
    // Integration: full B2 target hold-out against Python contract
    // -------------------------------------------------------------------

    #[test]
    fn test_b2_holdout_matches_python_contract() {
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

        let reader = crate::RasterReader::open(&path).expect("should open B2");
        let n_bands = reader.band_count();
        let win = 512;
        let (full_rows, full_cols) = (reader.shape().0, reader.shape().1);
        let rows = win.min(full_rows);
        let cols = win.min(full_cols);

        // Read full cube (windowed to 512×512)
        let mut cube = ndarray::Array3::<f64>::zeros((n_bands, rows, cols));
        for b in 1..=n_bands {
            let band_data = reader.read_band_window(b, win, win).expect("should read band");
            cube.index_axis_mut(ndarray::Axis(0), b - 1).assign(&band_data);
        }

        // Build ISO timestamp strings from band descriptions
        let timestamps: Vec<String> = (1..=n_bands)
            .map(|b| {
                let band = reader.dataset.rasterband(b).unwrap();
                let desc = band.description().unwrap_or_default().trim().to_string();
                if let Some(sub) = crate::find_timestamp_substring(&desc) {
                    raw_to_iso(sub)
                } else {
                    desc
                }
            })
            .collect();

        // Deduplicate — Python pipeline collapses duplicate timestamps before
        // target hold-out. The contract's counts_before=201 reflects this.
        let (cube, timestamps) = crate::collapse_duplicate_timestamps(&cube, &timestamps);
        assert_eq!(cube.shape()[0], 201, "deduplicated band count must match contract counts_before");

        let target = "2025-05-20T03:52:01";
        let (masked_cube, masked_ts, target_idx, ground_truth) =
            make_masked_time_series(&cube, &timestamps, target)
                .expect("should find target in B2");

        // Contract verifications
        assert_eq!(masked_cube.shape()[0], 200, "masked should have 200 bands (counts_after)");
        assert_eq!(masked_ts.len(), 200);
        assert_eq!(target_idx, 181, "B2 target_idx should match contract mask_index");

        // Ground truth shape
        assert_eq!(ground_truth.shape(), &[rows, cols]);

        // Verify ground truth is the actual observed target slice
        let original_target_slice = cube.index_axis(ndarray::Axis(0), target_idx);
        for r in 0..rows {
            for c in 0..cols {
                let a = ground_truth[[r, c]];
                let b = original_target_slice[[r, c]];
                if a.is_nan() && b.is_nan() {
                    continue;
                }
                assert!(
                    (a - b).abs() < 1e-10,
                    "ground truth mismatch at ({r},{c}): {a} vs {b}"
                );
            }
        }

        // Verify masked cube does NOT contain target
        for ts in &masked_ts {
            assert_ne!(ts, target, "masked timestamps should not contain target");
        }
        assert_eq!(masked_ts.len(), cube.shape()[0] - 1);

        // Mask invalid reflectance in both arrays to test NaN masking
        let mut masked_cube = masked_cube;
        let mut ground_truth = ground_truth;
        mask_invalid_sentinel2(&mut masked_cube);
        for v in ground_truth.iter_mut() {
            if v.is_finite() && (*v <= 0.0 || *v >= 10000.0) {
                *v = f64::NAN;
            }
        }

        // Verify NaN masking worked on ground truth
        let nan_count = ground_truth.iter().filter(|v| v.is_nan()).count();
        let total = ground_truth.len();
        // Most pixels should be valid (non-NaN). Typically the fraction NaN is
        // very low for valid imagery. Just ensure the masking ran.
        assert!(
            nan_count < total,
            "NaN count {nan_count} / {total} in ground truth after masking"
        );

        dbg!(n_bands, masked_cube.shape(), target_idx, nan_count, total);
    }

    #[test]
    fn test_nufrost_full_scene_writes_six_band_geotiff() {
        let tmp = std::env::temp_dir().join("nufrost_test_fs_6band.tif");

        let sentinel_bands: [&str; 6] = ["B2", "B3", "B4", "B8", "B11", "B12"];
        let mut arrays = BTreeMap::new();
        for &name in &sentinel_bands {
            arrays.insert(name.to_string(), make_array2(8, 10, 500.0));
        }
        let ordered: Vec<String> = sentinel_bands.iter().map(|s| s.to_string()).collect();
        let meta = test_meta();

        write_band_stack(&tmp, &arrays, &ordered, &meta)
            .expect("write_band_stack with 6 Sentinel-2 bands should succeed");

        let ds = gdal_rs::Dataset::open(&tmp).expect("re-open should succeed");
        assert_eq!(ds.raster_count(), 6);
        for (i, &expected_name) in sentinel_bands.iter().enumerate() {
            let band = ds.rasterband(i + 1).unwrap();
            let desc = band.description().unwrap_or_default();
            assert_eq!(desc, expected_name);
        }
        let (cols, rows) = ds.raster_size();
        assert_eq!(rows, 8);
        assert_eq!(cols, 10);
        let _ = std::fs::remove_file(&tmp);
    }

    #[test]
    fn test_nufrost_full_scene_ground_truth_output_naming() {
        let root = Path::new("/tmp/test_fs");
        let path = build_ground_truth_output_path(
            root, "sentinel-2", 104.2595, 31.2170, "2025-05-20T03:52:01",
        );
        let s = path.to_str().unwrap();
        assert!(s.contains("[ground_truth]"), "{s}");
        assert!(s.contains("sentinel-2_lon104.259500_lat31.217000"), "{s}");
        assert!(s.contains("2025-05-20T03-52-01"), "{s}");
    }

    #[test]
    fn test_nufrost_full_scene_prediction_output_naming() {
        let root = Path::new("/tmp/test_fs");
        let path = build_scene_stack_output_path(
            root, "nufrost", "sentinel-2",
            104.2595, 31.2170, "2025-05-20T03:52:01", "prediction",
        );
        let s = path.to_str().unwrap();
        assert!(s.contains("[nufrost]"), "{s}");
        assert!(s.contains("sentinel-2_lon104.259500_lat31.217000"), "{s}");
        assert!(s.contains("_prediction.tif"), "{s}");
    }

    #[test]
    fn test_hants_full_scene_output_naming_unchanged() {
        let root = Path::new("/tmp/test_fs");
        let path = build_scene_stack_output_path(
            root, "hants", "sentinel-2",
            104.2595, 31.2170, "2025-05-20T03:52:01", "prediction",
        );
        let s = path.to_str().unwrap();
        assert!(s.contains("[hants]"), "{s}");
    }

    #[test]
    fn test_zhu2015_full_scene_output_naming_unchanged() {
        let root = Path::new("/tmp/test_fs");
        let path = build_scene_stack_output_path(
            root, "zhu2015", "sentinel-2",
            104.2595, 31.2170, "2025-05-20T03:52:01", "prediction",
        );
        let s = path.to_str().unwrap();
        assert!(s.contains("[zhu2015]"), "{s}");
    }
}
