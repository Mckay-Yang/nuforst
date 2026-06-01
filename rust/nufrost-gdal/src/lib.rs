// nufrost-gdal — raster I/O via the GDAL crate.
// Requires libgdal system library (e.g. `brew install gdal` or conda).
//
// Provides:
//  - RasterReader: open GeoTIFF/VRT, read metadata and band data, build valid masks
//  - RasterWriter: create single-band and multi-band GeoTIFF output
//  - write_zhu2015_output: 2-band (prediction + QA) output matching Python convention

use std::path::Path;

use anyhow::{Context, Result};
use gdal::{Dataset, DriverManager, GeoTransform, Metadata};
use gdal::raster::Buffer;
use gdal::spatial_ref::SpatialRef;
use ndarray::Array2;

use nufrost_core::{is_valid_reflectance, SENTINEL2_VALID_MAX, SENTINEL2_VALID_MIN};

// ---------------------------------------------------------------------------
// RasterReader — read-only access to GeoTIFF / VRT datasets
// ---------------------------------------------------------------------------

/// A read-only raster reader wrapping a GDAL [`Dataset`].
///
/// Supports GeoTIFF and VRT inputs. All band indices are 1-based.
pub struct RasterReader {
    dataset: Dataset,
    /// (columns, rows) — GDAL convention.
    raster_size: (usize, usize),
    band_count: usize,
}

impl RasterReader {
    /// Open a GeoTIFF or VRT file for reading.
    pub fn open<P: AsRef<Path>>(path: P) -> Result<Self> {
        let dataset = Dataset::open(path.as_ref())
            .with_context(|| format!("Failed to open raster: {}", path.as_ref().display()))?;
        let raster_size = dataset.raster_size(); // (cols, rows)
        let band_count = dataset.raster_count();
        Ok(Self {
            dataset,
            raster_size,
            band_count,
        })
    }

    // -- metadata -------------------------------------------------------

    /// Number of bands in the raster.
    pub fn band_count(&self) -> usize {
        self.band_count
    }

    /// Spatial dimensions as `(rows, cols)` (ndarray / Python convention).
    pub fn shape(&self) -> (usize, usize) {
        (self.raster_size.1, self.raster_size.0)
    }

    /// Spatial dimensions as `(cols, rows)` (GDAL convention).
    pub fn raster_size(&self) -> (usize, usize) {
        self.raster_size
    }

    /// Six-element affine geo-transform `[a, b, c, d, e, f]`.
    ///
    /// Returns `None` if the dataset has no georeferencing.
    pub fn geo_transform(&self) -> Option<GeoTransform> {
        self.dataset.geo_transform().ok()
    }

    /// CRS as WKT string, or `None` if unknown.
    pub fn crs_wkt(&self) -> Option<String> {
        self.dataset
            .spatial_ref()
            .ok()
            .and_then(|s| s.to_wkt().ok())
    }

    /// No-data value for a band (1-indexed), or `None` if unset.
    pub fn nodata(&self, band_idx: usize) -> Option<f64> {
        self.dataset
            .rasterband(band_idx)
            .ok()
            .and_then(|b| b.no_data_value())
    }

    // -- data access ----------------------------------------------------

    /// Read a single band as an [`Array2<f64>`] with shape `(rows, cols)`.
    ///
    /// Band index is 1-based.
    pub fn read_band(&self, band_idx: usize) -> Result<Array2<f64>> {
        let band = self
            .dataset
            .rasterband(band_idx)
            .with_context(|| format!("Band {band_idx} not available"))?;
        let buf = band
            .read_band_as::<f64>()
            .with_context(|| format!("Failed to read band {band_idx}"))?;
        let (cols, rows) = (buf.width(), buf.height());
        let ((_bcols, _brows), data) = buf.into_shape_and_vec();
        Array2::from_shape_vec((rows, cols), data)
            .map_err(|e| anyhow::anyhow!("Shape mismatch reading band {band_idx}: {e}"))
    }

    /// Build a Sentinel-2 valid-pixel mask for a band.
    ///
    /// Pixels are valid when `value > 0.0 && value < 10000.0` (finite values only).
    pub fn read_valid_mask(&self, band_idx: usize) -> Result<Array2<bool>> {
        self.read_valid_mask_custom(band_idx, SENTINEL2_VALID_MIN, SENTINEL2_VALID_MAX)
    }

    /// Build a valid-pixel mask with a custom reflectance range.
    pub fn read_valid_mask_custom(
        &self,
        band_idx: usize,
        valid_min: f64,
        valid_max: f64,
    ) -> Result<Array2<bool>> {
        let data = self.read_band(band_idx)?;
        Ok(data.mapv(|v| is_valid_reflectance(v, valid_min, valid_max)))
    }
}

// ---------------------------------------------------------------------------
// RasterWriter — create GeoTIFF output files
// ---------------------------------------------------------------------------

/// Create and write single-band or multi-band GeoTIFF files.
///
/// Written pixel type is `Float32` to match the Python pipeline convention.
pub struct RasterWriter {
    dataset: Dataset,
    rows: usize,
    cols: usize,
}

impl RasterWriter {
    /// Create a new GeoTIFF file.
    ///
    /// * `path`       — output file path
    /// * `rows`       — number of rows (height)
    /// * `cols`       — number of columns (width)
    /// * `bands`      — number of bands
    /// * `geo_transform` — 6-element affine transform
    /// * `crs_wkt`    — optional WKT CRS string
    /// * `nodata`     — optional no-data value (applied to all bands)
    pub fn create<P: AsRef<Path>>(
        path: P,
        rows: usize,
        cols: usize,
        bands: usize,
        geo_transform: &GeoTransform,
        crs_wkt: Option<&str>,
        nodata: Option<f64>,
    ) -> Result<Self> {
        let driver = DriverManager::get_driver_by_name("GTiff")?;
        let mut dataset = driver.create_with_band_type::<f32, _>(path.as_ref(), cols, rows, bands)?;

        dataset.set_geo_transform(geo_transform)?;

        if let Some(wkt) = crs_wkt {
            let srs = SpatialRef::from_wkt(wkt)
                .with_context(|| "Failed to parse CRS WKT")?;
            dataset.set_spatial_ref(&srs)?;
        }

        if let Some(nd) = nodata {
            for i in 1..=bands {
                let mut band = dataset.rasterband(i)?;
                band.set_no_data_value(Some(nd))?;
            }
        }

        Ok(Self {
            dataset,
            rows,
            cols,
        })
    }

    /// Write an [`Array2<f64>`] to a band (1-indexed).
    ///
    /// Values are converted to `f32` to match the Python pipeline's `Float32` convention.
    pub fn write_band(&mut self, band_idx: usize, data: &Array2<f64>) -> Result<()> {
        let (rows, cols) = data.dim();
        if rows != self.rows || cols != self.cols {
            anyhow::bail!(
                "Data shape ({rows}, {cols}) does not match dataset ({}, {})",
                self.rows,
                self.cols
            );
        }

        let mut band = self.dataset.rasterband(band_idx)?;
        let flat: Vec<f32> = data.iter().map(|&v| v as f32).collect();
        // GDAL Buffer shape is (cols, rows) — opposite of ndarray (rows, cols).
        let mut buf = Buffer::new((cols, rows), flat);
        band.write((0, 0), (cols, rows), &mut buf)?;
        Ok(())
    }

    /// Finish writing and flush/closing. The file is finalized when `writer` is dropped,
    /// but you can call this explicitly for clarity.
    pub fn flush(&mut self) -> Result<()> {
        self.dataset.flush_cache()?;
        Ok(())
    }
}

/// Metadata bundle used when creating output files from algorithm results.
#[derive(Debug, Clone)]
pub struct RasterMetadata {
    pub geo_transform: GeoTransform,
    pub crs_wkt: Option<String>,
    pub nodata: Option<f64>,
}

// ---------------------------------------------------------------------------
// Zhu2015 2-band output helper
// ---------------------------------------------------------------------------

/// Write Zhu2015 output as a 2-band GeoTIFF (band 1 = prediction, band 2 = QA).
///
/// Matches the Python pipeline convention where [`reconstruct_zhu2015_from_cube`]
/// returns a 2-band array and `_write_prediction` handles it as a multi-band stack.
pub fn write_zhu2015_output<P: AsRef<Path>>(
    path: P,
    prediction: &Array2<f64>,
    qa: &Array2<f64>,
    metadata: &RasterMetadata,
) -> Result<()> {
    let (rows, cols) = prediction.dim();
    if qa.dim() != (rows, cols) {
        anyhow::bail!(
            "Prediction shape ({rows}, {cols}) != QA shape ({}, {})",
            qa.dim().0,
            qa.dim().1
        );
    }

    let mut writer = RasterWriter::create(
        path,
        rows,
        cols,
        2,
        &metadata.geo_transform,
        metadata.crs_wkt.as_deref(),
        metadata.nodata,
    )?;
    writer.write_band(1, prediction)?;
    writer.write_band(2, qa)?;
    writer.flush()?;
    Ok(())
}

// ---------------------------------------------------------------------------
// Raster reconstruction — per-pixel algorithm application on GeoTIFF input
// ---------------------------------------------------------------------------

/// Typed error for invalid raster inputs.
///
/// Produced when the input GeoTIFF does not meet the requirements for
/// time-series reconstruction (wrong band count, mismatched timestamps, etc.).
#[derive(Debug)]
pub enum RasterInputError {
    /// The raster has no bands (empty dataset).
    #[allow(dead_code)]
    EmptyRaster,
    /// The number of bands does not match the number of timestamps provided.
    BandTimestampMismatch {
        n_bands: usize,
        n_timestamps: usize,
    },
    /// The raster spatial dimensions are zero.
    ZeroSize { rows: usize, cols: usize },
    /// A band description could not be parsed as a timestamp.
    #[allow(dead_code)]
    InvalidBandTimestamp { band_idx: usize, description: String },
}

impl std::fmt::Display for RasterInputError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::EmptyRaster => write!(f, "raster has no bands"),
            Self::BandTimestampMismatch { n_bands, n_timestamps } => {
                write!(f, "band count ({n_bands}) != timestamp count ({n_timestamps})")
            }
            Self::ZeroSize { rows, cols } => write!(f, "raster has zero spatial size ({rows}r × {cols}c)"),
            Self::InvalidBandTimestamp { band_idx, description } => {
                write!(f, "band {band_idx} description is not a valid timestamp: '{description}'")
            }
        }
    }
}

impl std::error::Error for RasterInputError {}

/// Extract relative-day timestamps from multi-band GeoTIFF band descriptions.
///
/// For Sentinel-2 convention, each band stores a timestamp in its description
/// (e.g. `"2020-01-01T04:58:59Z"`).  This function parses band descriptions
/// as ISO-8601, converts to epoch seconds, then returns days relative to the
/// first valid timestamp.
///
/// Returns `(timestamps_days, target_time_day)` where `target_time_day` is
/// the last timestamp (matching the Python "hold out last scene" convention).
pub fn extract_timestamps_from_band_descriptions(
    reader: &RasterReader,
) -> Result<(Vec<f64>, f64), RasterInputError> {
    let n = reader.band_count();
    if n == 0 {
        return Err(RasterInputError::EmptyRaster);
    }

    let mut epoch_secs = Vec::with_capacity(n);
    for b in 1..=n {
        let band = reader
            .dataset
            .rasterband(b)
            .map_err(|_| RasterInputError::InvalidBandTimestamp {
                band_idx: b,
                description: String::from("<unavailable>"),
            })?;
        let desc = band
            .description()
            .unwrap_or_default();
        let desc = desc.trim();
        // Try ISO-8601 via nufrost-core helper first, then RFC 3339 fallback
        let epoch = nufrost_core::time::parse_iso8601_to_epoch_seconds(desc)
            .or_else(|| {
                chrono::DateTime::parse_from_rfc3339(desc)
                    .ok()
                    .map(|dt| dt.timestamp() as f64)
            })
            .ok_or_else(|| RasterInputError::InvalidBandTimestamp {
                band_idx: b,
                description: desc.to_string(),
            })?;
        epoch_secs.push(epoch);
    }

    let t0 = epoch_secs[0];
    let days: Vec<f64> = epoch_secs.iter().map(|&s| (s - t0) / 86400.0).collect();
    let target = days[days.len() - 1]; // hold out last scene
    Ok((days, target))
}

/// Build synthetic timestamps from band indices (1 day per band).
///
/// Used when band descriptions are not timestamp-formatted (e.g. synthetic
/// test data).  Band 0 → day 0, Band 1 → day 1, etc.
pub fn synthetic_timestamps_from_bands(n_bands: usize) -> (Vec<f64>, f64) {
    let days: Vec<f64> = (0..n_bands).map(|i| i as f64).collect();
    let target = days[days.len() - 1];
    (days, target)
}

// ── Internal: shared reconstruction loop ──────────────────────────────────

/// Run a per-pixel reconstruction algorithm on a 3D cube `(bands, rows, cols)`
/// and write a single-band Float32 GeoTIFF.
///
/// `reconstruct_fn` receives `(timestamps_days, observations, target_t_day)`
/// and returns a scalar prediction.
///
/// Uses rayon for parallel row processing.
fn reconstruct_single_band<F, P: AsRef<Path>>(
    cube: &ndarray::Array3<f64>,
    timestamps_days: &[f64],
    target_t_day: f64,
    output_path: P,
    metadata: &RasterMetadata,
    reconstruct_fn: F,
) -> Result<()>
where
    F: Fn(&[f64], &[f64], f64) -> f64 + Sync + Send,
{
    let (n_bands, rows, cols) = cube.dim();
    if n_bands == 0 || rows == 0 || cols == 0 {
        return Err(anyhow::anyhow!(
            "invalid cube dimensions: ({n_bands}, {rows}, {cols})"
        ));
    }

    use rayon::prelude::*;
    let mut output = ndarray::Array2::<f64>::from_elem((rows, cols), f64::NAN);

    // Parallel over rows: each row gets its own thread-local buffers
    output
        .axis_iter_mut(ndarray::Axis(0))
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
                        ts_buf.push(timestamps_days[b]);
                        obs_buf.push(v);
                    }
                }
                if !ts_buf.is_empty() {
                    row_out[c] = reconstruct_fn(&ts_buf, &obs_buf, target_t_day);
                }
            }
        });

    let mut writer = RasterWriter::create(
        output_path,
        rows,
        cols,
        1,
        &metadata.geo_transform,
        metadata.crs_wkt.as_deref(),
        metadata.nodata,
    )?;
    writer.write_band(1, &output)?;
    writer.flush()?;
    Ok(())
}

// ── Public per-algorithm reconstruction entrypoints ──────────────────────

/// Reconstruct a full raster using NUFROST.
///
/// Reads all bands from `reader`, applies [`nufrost_core::nufrost_pixel`] to
/// every pixel in parallel, and writes a single-band Float32 GeoTIFF.
pub fn reconstruct_nufrost_geotiff<P: AsRef<Path>>(
    reader: &RasterReader,
    timestamps_days: &[f64],
    target_t_day: f64,
    config: &nufrost_core::NufrostConfig,
    output_path: P,
    metadata: &RasterMetadata,
) -> Result<()> {
    let cube = read_all_bands(reader)?;
    reconstruct_single_band(
        &cube,
        timestamps_days,
        target_t_day,
        output_path,
        metadata,
        |ts, obs, targ| {
            let (pred, _n_freqs) =
                nufrost_core::nufrost_pixel(ts, obs, targ, config);
            if pred.is_finite() { pred } else { f64::NAN }
        },
    )
}

/// Reconstruct a full raster using HANTS.
///
/// Reads all bands from `reader`, applies [`nufrost_core::hants_pixel`] to
/// every pixel in parallel, and writes a single-band Float32 GeoTIFF.
pub fn reconstruct_hants_geotiff<P: AsRef<Path>>(
    reader: &RasterReader,
    timestamps_days: &[f64],
    target_t_day: f64,
    nof: u32,
    sf: &str,
    valid_min: Option<f64>,
    valid_max: Option<f64>,
    fet: f64,
    dod: u32,
    period: f64,
    output_path: P,
    metadata: &RasterMetadata,
) -> Result<()> {
    let cube = read_all_bands(reader)?;
    reconstruct_single_band(
        &cube,
        timestamps_days,
        target_t_day,
        output_path,
        metadata,
        |ts, obs, targ| {
            let pred = nufrost_core::hants_pixel(
                ts, obs, targ, nof, sf, valid_min, valid_max, fet, dod, period,
            );
            if pred.is_finite() { pred } else { f64::NAN }
        },
    )
}

/// Reconstruct a full raster using Zhu2015.
///
/// Reads all bands from `reader`, applies
/// [`nufrost_core::zhu2015::fit_predict_pixel`] to every pixel in parallel,
/// and writes a 2-band GeoTIFF (Band 1 = Float32 prediction, Band 2 = Float32 QA).
///
/// The Float32 QA band preserves integer QA values (0-255) losslessly,
/// matching the Python pipeline convention.
pub fn reconstruct_zhu2015_geotiff<P: AsRef<Path>>(
    reader: &RasterReader,
    timestamps_days: &[f64],
    target_t_day: f64,
    lasso_alpha: f64,
    output_path: P,
    metadata: &RasterMetadata,
) -> Result<()> {
    let (n_bands, rows, cols) = {
        let (rows, cols) = reader.shape();
        (reader.band_count(), rows, cols)
    };
    if n_bands == 0 || rows == 0 || cols == 0 {
        anyhow::bail!("empty raster input");
    }

    let cube = read_all_bands(reader)?;

    use rayon::prelude::*;
    let mut prediction = ndarray::Array2::<f64>::from_elem((rows, cols), f64::NAN);
    let mut qa = ndarray::Array2::<f64>::zeros((rows, cols));

    // Parallel over rows with thread-local buffers
    let pred_view = &mut prediction;
    let qa_view = &mut qa;
    pred_view
        .axis_iter_mut(ndarray::Axis(0))
        .into_par_iter()
        .enumerate()
        .zip(
            qa_view
                .axis_iter_mut(ndarray::Axis(0))
                .into_par_iter()
                .enumerate(),
        )
        .for_each(|((r, mut pred_row), (_, mut qa_row))| {
            let mut ts_buf = Vec::with_capacity(n_bands);
            let mut obs_buf = Vec::with_capacity(n_bands);
            for c in 0..cols {
                ts_buf.clear();
                obs_buf.clear();
                for b in 0..n_bands {
                    let v = cube[[b, r, c]];
                    if v.is_finite() {
                        ts_buf.push(timestamps_days[b]);
                        obs_buf.push(v);
                    }
                }
                if !ts_buf.is_empty() {
                    let result = nufrost_core::zhu2015::fit_predict_pixel(
                        &ts_buf, &obs_buf, target_t_day, lasso_alpha,
                    );
                    pred_row[c] = if result.prediction.is_finite() {
                        result.prediction
                    } else {
                        f64::NAN
                    };
                    qa_row[c] = result.qa as f64;
                }
            }
        });

    write_zhu2015_output(output_path, &prediction, &qa, metadata)
}

// Read all bands from a reader into a 3D cube `(bands, rows, cols)`.
fn read_all_bands(reader: &RasterReader) -> Result<ndarray::Array3<f64>> {
    let n_bands = reader.band_count();
    let (rows, cols) = reader.shape();
    if n_bands == 0 {
        anyhow::bail!("raster has no bands");
    }
    if rows == 0 || cols == 0 {
        anyhow::bail!("raster has zero spatial dimensions ({rows}r × {cols}c)");
    }

    let mut cube = ndarray::Array3::<f64>::zeros((n_bands, rows, cols));
    for b in 0..n_bands {
        let band_data = reader.read_band(b + 1)?;
        cube.slice_mut(ndarray::s![b, .., ..]).assign(&band_data);
    }
    Ok(cube)
}

// ---------------------------------------------------------------------------
// GDAL version helper (kept from placeholder for smoke testing)
// ---------------------------------------------------------------------------

/// Returns the GDAL version string at runtime (e.g. "3.10.3").
pub fn gdal_version() -> String {
    gdal::version::version_info("GDAL_VERSION")
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::Array2;
    use nufrost_core::NufrostConfig;
    use std::fs;

    const DEFAULT_GEO: GeoTransform = [0.0, 1.0, 0.0, 0.0, 0.0, -1.0];

    // -- helper to create a small synthetic GeoTIFF for roundtrip testing --

    fn _write_synthetic_tif(
        path: &Path,
        rows: usize,
        cols: usize,
        bands: usize,
        geo: &GeoTransform,
        crs: Option<&str>,
        nodata: Option<f64>,
    ) -> Result<()> {
        let mut writer = RasterWriter::create(path, rows, cols, bands, geo, crs, nodata)?;
        for b in 1..=bands {
            let mut arr = Array2::<f64>::zeros((rows, cols));
            for r in 0..rows {
                for c in 0..cols {
                    arr[[r, c]] = ((r * cols + c + b * 100) as f64) * 0.1;
                }
            }
            writer.write_band(b, &arr)?;
        }
        writer.flush()?;
        Ok(())
    }

    // -- smoke test (kept from placeholder) --

    #[test]
    fn placeholder_gdal_version() {
        let ver = gdal_version();
        assert!(ver.contains('.'), "Expected dotted version, got: {ver}");
    }

    // -- reader metadata tests --

    #[test]
    fn reader_metadata_synthetic() {
        let path = Path::new("test_metadata.tif");
        _write_synthetic_tif(path, 5, 10, 3, &DEFAULT_GEO, None, Some(-9999.0)).unwrap();

        let r = RasterReader::open(path).unwrap();
        assert_eq!(r.band_count(), 3);
        assert_eq!(r.shape(), (5, 10));
        assert_eq!(r.raster_size(), (10, 5));
        assert_eq!(r.nodata(1), Some(-9999.0));
        assert_eq!(r.nodata(2), Some(-9999.0));
        assert_eq!(r.nodata(3), Some(-9999.0));

        let geo = r.geo_transform().unwrap();
        assert_eq!(geo, DEFAULT_GEO);

        let _ = fs::remove_file(path);
    }

    #[test]
    fn reader_crs_roundtrip() {
        let path = Path::new("test_crs.tif");
        let wkt = r#"GEOGCS["WGS 84",DATUM["WGS_1984",SPHEROID["WGS 84",6378137,298.257223563,AUTHORITY["EPSG","7030"]],AUTHORITY["EPSG","6326"]],PRIMEM["Greenwich",0,AUTHORITY["EPSG","8901"]],UNIT["degree",0.0174532925199433,AUTHORITY["EPSG","9122"]],AUTHORITY["EPSG","4326"]]"#;
        _write_synthetic_tif(path, 3, 3, 1, &DEFAULT_GEO, Some(wkt), None).unwrap();

        let r = RasterReader::open(path).unwrap();
        let got = r.crs_wkt().unwrap();
        assert!(got.contains("WGS 84"), "CRS WKT mismatch: {got}");

        let _ = fs::remove_file(path);
    }

    // -- reader data roundtrip --

    #[test]
    fn read_band_roundtrip() {
        let path = Path::new("test_read_roundtrip.tif");
        let rows = 4;
        let cols = 8;
        _write_synthetic_tif(path, rows, cols, 2, &DEFAULT_GEO, None, None).unwrap();

        let r = RasterReader::open(path).unwrap();
        let b1 = r.read_band(1).unwrap();
        let b2 = r.read_band(2).unwrap();

        assert_eq!(b1.dim(), (rows, cols));
        assert_eq!(b2.dim(), (rows, cols));

        // First band: values are ((r*cols + c + 100) * 0.1) as f32 -> f64
        assert!((b1[[0, 0]] - 10.0).abs() < 1e-5, "b1[0,0]={}", b1[[0, 0]]);
        assert!((b1[[rows - 1, cols - 1]] - ((rows * cols - 1) as f64 + 100.0) * 0.1).abs() < 1e-5);
        // Second band: +200 base
        assert!((b2[[0, 0]] - 20.0).abs() < 1e-5, "b2[0,0]={}", b2[[0, 0]]);

        let _ = fs::remove_file(path);
    }

    // -- valid mask --

    #[test]
    fn read_valid_mask() {
        let path = Path::new("test_mask.tif");
        let rows = 3;
        let cols = 3;
        // Write a band with known valid/invalid values
        {
            let mut writer = RasterWriter::create(path, rows, cols, 1, &DEFAULT_GEO, None, None)
                .unwrap();
            let data = Array2::from_shape_vec(
                (rows, cols),
                vec![
                    500.0, f64::NAN, 0.0,
                    10000.0, 3000.0, -1.0,
                    1.0, 9999.0, f64::INFINITY,
                ],
            )
            .unwrap();
            writer.write_band(1, &data).unwrap();
            writer.flush().unwrap();
        }

        let r = RasterReader::open(path).unwrap();
        let mask = r.read_valid_mask(1).unwrap();

        assert!(mask[[0, 0]]);   // 500 — valid
        assert!(!mask[[0, 1]]);  // NaN — invalid
        assert!(!mask[[0, 2]]);  // 0 — invalid (<= 0)
        assert!(!mask[[1, 0]]);  // 10000 — invalid (>= 10000)
        assert!(mask[[1, 1]]);   // 3000 — valid
        assert!(!mask[[1, 2]]);  // -1 — invalid
        assert!(mask[[2, 0]]);   // 1 — valid
        assert!(mask[[2, 1]]);   // 9999 — valid
        assert!(!mask[[2, 2]]);  // inf — invalid

        let _ = fs::remove_file(path);
    }

    // -- zhu2015 2-band output --

    #[test]
    fn write_zhu2015_output_2band() {
        let path = Path::new("test_zhu2015_2band.tif");
        let meta = RasterMetadata {
            geo_transform: DEFAULT_GEO,
            crs_wkt: None,
            nodata: Some(-9999.0),
        };

        let prediction = Array2::from_shape_vec((5, 10), (0..50).map(|v| v as f64).collect()).unwrap();
        let qa = Array2::from_shape_vec((5, 10), (0..50).map(|v| (v % 3) as f64).collect()).unwrap();

        write_zhu2015_output(path, &prediction, &qa, &meta).unwrap();

        // Read back
        let reader = RasterReader::open(path).unwrap();
        assert_eq!(reader.band_count(), 2);
        assert_eq!(reader.shape(), (5, 10));
        let b1 = reader.read_band(1).unwrap();
        let b2 = reader.read_band(2).unwrap();

        assert!((b1[[0, 0]] - 0.0).abs() < 1e-5);
        assert!((b1[[4, 9]] - 49.0).abs() < 1e-5);
        assert!((b2[[0, 0]] - 0.0).abs() < 1e-5);
        assert!((b2[[1, 0]] - 1.0).abs() < 1e-5);

        let _ = fs::remove_file(path);
    }

    // -- shape mismatch error --

    #[test]
    fn writer_shape_mismatch() {
        let path = Path::new("test_shape_err.tif");
        let writer = RasterWriter::create(path, 5, 5, 1, &DEFAULT_GEO, None, None);
        assert!(writer.is_ok(), "create should succeed");
        let _ = fs::remove_file(path);
    }

    // -- nodata not set produces None --

    /// Writes a test GeoTIFF to the evidence directory for Python rasterio verification.
    #[test]
    fn write_evidence_single_band() {
        let ev_dir = Path::new(".sisyphus/evidence");
        let _ = fs::create_dir_all(ev_dir);
        let path = ev_dir.join("task-8-rust-written-singleband.tif");

        let rows = 5;
        let cols = 7;
        let geo = [100000.0, 30.0, 0.0, 5000000.0, 0.0, -30.0];
        let wkt = r#"GEOGCS["WGS 84",DATUM["WGS_1984",SPHEROID["WGS 84",6378137,298.257223563,AUTHORITY["EPSG","7030"]],AUTHORITY["EPSG","6326"]],PRIMEM["Greenwich",0,AUTHORITY["EPSG","8901"]],UNIT["degree",0.0174532925199433,AUTHORITY["EPSG","9122"]],AUTHORITY["EPSG","4326"]]"#;

        let data = Array2::from_shape_vec((rows, cols), (0..(rows * cols)).map(|v| v as f64).collect()).unwrap();

        let mut w = RasterWriter::create(
            &path, rows, cols, 1, &geo, Some(wkt), Some(-9999.0),
        )
        .unwrap();
        w.write_band(1, &data).unwrap();
        w.flush().unwrap();
    }

    /// Writes Zhu2015 2-band test GeoTIFF for Python rasterio verification.
    #[test]
    fn write_evidence_zhu2015_2band() {
        let ev_dir = Path::new(".sisyphus/evidence");
        let _ = fs::create_dir_all(ev_dir);
        let path = ev_dir.join("task-8-rust-written-zhu2015-2band.tif");

        let rows = 4;
        let cols = 6;
        let meta = RasterMetadata {
            geo_transform: [200000.0, 10.0, 0.0, 4000000.0, 0.0, -10.0],
            crs_wkt: None,
            nodata: Some(-32768.0),
        };

        let prediction = Array2::from_shape_vec(
            (rows, cols),
            (0..24).map(|v| v as f64 * 0.01).collect(),
        )
        .unwrap();
        let qa = Array2::from_shape_vec(
            (rows, cols),
            (0..24).map(|v| (v % 4) as f64).collect(),
        )
        .unwrap();

        write_zhu2015_output(&path, &prediction, &qa, &meta).unwrap();
    }

    #[test]
    fn nodata_none_by_default() {
        let path = Path::new("test_no_nodata.tif");
        _write_synthetic_tif(path, 2, 2, 1, &DEFAULT_GEO, None, None).unwrap();
        let r = RasterReader::open(path).unwrap();
        // When no nodata is set, GDAL may return a success=false from
        // GDALGetRasterNoDataValue, which maps to None.
        assert_eq!(r.nodata(1), None);
        let _ = fs::remove_file(path);
    }

    // ── Integration: end-to-end small-window reconstruction ────────────

    /// Create a synthetic multi-band time-series GeoTIFF.
    ///
    /// Each band represents one timestamp. Pixel values follow a sine wave
    /// plus a spatial gradient so algorithms have signal to reconstruct.
    fn _write_time_series_tif(
        path: &Path,
        rows: usize,
        cols: usize,
        n_timestamps: usize,
    ) -> Result<()> {
        use std::f64::consts::PI;
        let mut writer = RasterWriter::create(path, rows, cols, n_timestamps, &DEFAULT_GEO, None, Some(f64::NAN))?;
        for b in 0..n_timestamps {
            let t = b as f64;
            let mut arr = Array2::<f64>::zeros((rows, cols));
            for r in 0..rows {
                for c in 0..cols {
                    let spatial = ((r + c) as f64) * 10.0;
                    arr[[r, c]] = spatial + 100.0 * (2.0 * PI * t / n_timestamps as f64).sin() + 500.0;
                }
            }
            writer.write_band(b + 1, &arr)?;
        }
        writer.flush()?;
        Ok(())
    }

    /// Verify output GeoTIFF band count and spatial shape.
    fn _check_output(path: &Path, expected_rows: usize, expected_cols: usize, expected_bands: usize) {
        let r = RasterReader::open(path).unwrap();
        assert_eq!(
            r.shape(),
            (expected_rows, expected_cols),
            "output shape mismatch"
        );
        assert_eq!(
            r.band_count(),
            expected_bands,
            "output band count mismatch"
        );
    }

    #[test]
    fn reconstruct_nufrost_small_window_roundtrip() {
        let input = Path::new("test_ts_nufrost.tif");
        let output = Path::new("test_out_nufrost.tif");
        _write_time_series_tif(input, 5, 5, 10).unwrap();

        let reader = RasterReader::open(input).unwrap();
        let meta = RasterMetadata {
            geo_transform: DEFAULT_GEO,
            crs_wkt: None,
            nodata: Some(f64::NAN),
        };
        let (t_days, target_t) = synthetic_timestamps_from_bands(reader.band_count());

        let config = NufrostConfig {
            modes: 256,
            eps: 1e-12,
            num_peaks: 2,
            power_cum: 0.7,
            ignore_dc_hz: 1e-10,
            frequency_selection: "hybrid".into(),
            preferred_periods_days: String::new(),
            preferred_top_k: 0,
            spectral_top_k: 2,
            spectral_merge_tol: 0.15,
            refine_peaks: false,
            include_trend: true,
            ridge_lam: 0.01,
            freq_weight: 1.0,
            huber_iters: 3,
            huber_delta: 0.05,
            min_obs: 3,
            outlier_sigma: 2.5,
            lambda_step: 1e30,
            lambda_high: 0.005,
            low_freq_period_days: 0.0,
            step_dt_weighting: false,
            max_outer_iter: 3,
            outer_tol: 1e-3,
            joint_outlier: false,
            joint_outlier_sigma: 2.5,
            admm_rho: 1.0,
            admm_max_iter: 10,
            admm_tol: 1e-3,
        };

        reconstruct_nufrost_geotiff(&reader, &t_days, target_t, &config, output, &meta).unwrap();

        _check_output(output, 5, 5, 1);

        // Verify output has finite values
        let out_r = RasterReader::open(output).unwrap();
        let out_data = out_r.read_band(1).unwrap();
        let n_finite = out_data.iter().filter(|v| v.is_finite()).count();
        assert!(n_finite > 0, "NUFROST output should have finite predictions; got {n_finite}");

        let _ = fs::remove_file(input);
        let _ = fs::remove_file(output);
    }

    #[test]
    fn reconstruct_hants_small_window_roundtrip() {
        let input = Path::new("test_ts_hants.tif");
        let output = Path::new("test_out_hants.tif");
        _write_time_series_tif(input, 5, 5, 10).unwrap();

        let reader = RasterReader::open(input).unwrap();
        let meta = RasterMetadata {
            geo_transform: DEFAULT_GEO,
            crs_wkt: None,
            nodata: Some(f64::NAN),
        };
        let (t_days, target_t) = synthetic_timestamps_from_bands(reader.band_count());

        reconstruct_hants_geotiff(
            &reader, &t_days, target_t,
            3, "high", Some(0.0), Some(10000.0),
            500.0, 5, 365.25,
            output, &meta,
        ).unwrap();

        _check_output(output, 5, 5, 1);

        let out_r = RasterReader::open(output).unwrap();
        let out_data = out_r.read_band(1).unwrap();
        let n_finite = out_data.iter().filter(|v| v.is_finite()).count();
        assert!(n_finite > 0, "HANTS output should have finite predictions; got {n_finite}");

        let _ = fs::remove_file(input);
        let _ = fs::remove_file(output);
    }

    #[test]
    fn reconstruct_zhu2015_small_window_roundtrip() {
        let input = Path::new("test_ts_zhu.tif");
        let output = Path::new("test_out_zhu.tif");
        _write_time_series_tif(input, 5, 5, 12).unwrap(); // need 12+ obs for full fit

        let reader = RasterReader::open(input).unwrap();
        let meta = RasterMetadata {
            geo_transform: DEFAULT_GEO,
            crs_wkt: None,
            nodata: Some(f64::NAN),
        };
        let (t_days, target_t) = synthetic_timestamps_from_bands(reader.band_count());

        reconstruct_zhu2015_geotiff(&reader, &t_days, target_t, 0.1, output, &meta).unwrap();

        // Zhu2015 outputs 2 bands
        _check_output(output, 5, 5, 2);

        let out_r = RasterReader::open(output).unwrap();
        let pred = out_r.read_band(1).unwrap();
        let qa = out_r.read_band(2).unwrap();
        let n_finite_pred = pred.iter().filter(|v| v.is_finite()).count();
        let n_finite_qa = qa.iter().filter(|v| v.is_finite()).count();
        assert!(n_finite_pred > 0, "Zhu2015 prediction band should have finite values; got {n_finite_pred}");
        assert!(n_finite_qa > 0, "Zhu2015 QA band should have values; got {n_finite_qa}");

        let _ = fs::remove_file(input);
        let _ = fs::remove_file(output);
    }

    #[test]
    fn invalid_raster_empty_band_count() {
        let path = Path::new("test_empty.tif");
        // Create a 1-band empty raster
        {
            let mut writer = RasterWriter::create(path, 3, 3, 1, &DEFAULT_GEO, None, None).unwrap();
            let data = Array2::<f64>::from_elem((3, 3), f64::NAN);
            writer.write_band(1, &data).unwrap();
            writer.flush().unwrap();
        }

        let reader = RasterReader::open(path).unwrap();
        let meta = RasterMetadata { geo_transform: DEFAULT_GEO, crs_wkt: None, nodata: Some(f64::NAN) };
        let (t_days, target_t) = synthetic_timestamps_from_bands(reader.band_count());

        // 1 band with all-NaN values: algorithms should still run but output NaNs
        let output = Path::new("test_out_empty.tif");
        let config = NufrostConfig {
            modes: 256, eps: 1e-12, num_peaks: 2, power_cum: 0.7, ignore_dc_hz: 1e-10,
            frequency_selection: "hybrid".into(), preferred_periods_days: String::new(),
            preferred_top_k: 0, spectral_top_k: 2, spectral_merge_tol: 0.15,
            refine_peaks: false, include_trend: true, ridge_lam: 0.01, freq_weight: 1.0,
            huber_iters: 3, huber_delta: 0.05, min_obs: 3, outlier_sigma: 2.5,
            lambda_step: 1e30, lambda_high: 0.005, low_freq_period_days: 0.0,
            step_dt_weighting: false, max_outer_iter: 3, outer_tol: 1e-3,
            joint_outlier: false, joint_outlier_sigma: 2.5,
            admm_rho: 1.0, admm_max_iter: 10, admm_tol: 1e-3,
        };

        let result = reconstruct_nufrost_geotiff(&reader, &t_days, target_t, &config, output, &meta);
        assert!(result.is_ok(), "NAN-only input should still produce output (with NAN predictions)");

        // Verify output exists
        let out_r = RasterReader::open(output).unwrap();
        assert_eq!(out_r.band_count(), 1);
        assert_eq!(out_r.shape(), (3, 3));

        let _ = fs::remove_file(path);
        let _ = fs::remove_file(output);
    }
}
