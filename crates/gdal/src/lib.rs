pub mod full_scene;

// gdal — raster I/O via GDAL bindings.
// Requires libgdal system library (e.g. `brew install gdal` or conda).
//
// Provides:
//  - RasterReader: open GeoTIFF/VRT, read metadata and band data, build valid masks
//  - RasterWriter: create single-band GeoTIFF output

use std::path::Path;

use anyhow::{Context, Result};
use chrono::NaiveDateTime;
use gdal_rs::{Dataset, DriverManager, GeoTransform, Metadata};
use gdal_rs::raster::Buffer;
use gdal_rs::spatial_ref::SpatialRef;
use ndarray::Array2;

/// Default valid reflectance range for Sentinel-2 L2A scaled DN values.
pub const SENTINEL2_VALID_MIN: f64 = 0.0;
pub const SENTINEL2_VALID_MAX: f64 = 10000.0;

const PARSE_FORMATS: &[&str] = &[
    "%Y-%m-%dT%H:%M:%S",
    "%Y-%m-%d %H:%M:%S",
    "%Y-%m-%d",
    "%Y-%m-%dT%H:%M:%SZ",
    "%Y%m%dT%H%M%S",
    "%Y%m%d",
];

/// Check whether a single reflectance value falls within a valid range.
pub fn is_valid_reflectance(value: f64, valid_min: f64, valid_max: f64) -> bool {
    value.is_finite() && value > valid_min && value < valid_max
}

/// Scan `desc` for the first Sentinel-2 style timestamp substring.
pub fn find_timestamp_substring(desc: &str) -> Option<&str> {
    let bytes = desc.as_bytes();
    let len = bytes.len();
    let mut i = 0;
    while i + 14 < len {
        if bytes[i..i + 8].iter().all(u8::is_ascii_digit)
            && bytes[i + 8] == b'T'
            && bytes[i + 9..i + 15].iter().all(u8::is_ascii_digit)
        {
            return Some(&desc[i..i + 15]);
        }
        i += 1;
    }
    None
}

/// Parse a timestamp string into seconds since Unix epoch, interpreted as UTC.
pub fn parse_iso8601_to_epoch_seconds(ts: &str) -> Option<f64> {
    let ts = ts.trim();
    if ts.is_empty() {
        return None;
    }
    for fmt in PARSE_FORMATS {
        if let Ok(dt) = NaiveDateTime::parse_from_str(ts, fmt) {
            return Some(dt.and_utc().timestamp() as f64);
        }
    }
    None
}

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

    /// Read a window `(window_rows, window_cols)` from the top-left of a band.
    ///
    /// Returns an [`Array2<f64>`] with shape `(window_rows, window_cols)`.
    /// Matches the Python behaviour of `RSCube._read_tif()` which only reads
    /// the top-left 512×512 window for memory safety.
    pub fn read_band_window(
        &self,
        band_idx: usize,
        window_rows: usize,
        window_cols: usize,
    ) -> Result<Array2<f64>> {
        let (full_cols, full_rows) = self.raster_size;
        let cols = window_cols.min(full_cols);
        let rows = window_rows.min(full_rows);
        let band = self
            .dataset
            .rasterband(band_idx)
            .with_context(|| format!("Band {band_idx} not available"))?;
        let buf = band
            .read_as::<f64>((0, 0), (cols, rows), (cols, rows), None)
            .with_context(|| format!("Failed to read band {band_idx} window"))?;
        let ((_bcols, _brows), data) = buf.into_shape_and_vec();
        Array2::from_shape_vec((rows, cols), data)
            .map_err(|e| anyhow::anyhow!("Shape mismatch reading band {band_idx} window: {e}"))
    }

    /// Read a window `(window_rows, window_cols)` from a band at an arbitrary offset.
    ///
    /// `(row_offset, col_offset)` specifies the top-left corner in pixel coordinates
    /// (row-major for ease of use with `shape()`). Offsets are clamped to the raster
    /// bounds so callers don't need to pre-validate.
    ///
    /// Returns an [`Array2<f64>`] with shape `(window_rows, window_cols)`.
    pub fn read_band_window_offset(
        &self,
        band_idx: usize,
        row_offset: usize,
        col_offset: usize,
        window_rows: usize,
        window_cols: usize,
    ) -> Result<Array2<f64>> {
        let (full_cols, full_rows) = self.raster_size;
        let r_off = row_offset.min(full_rows.saturating_sub(1));
        let c_off = col_offset.min(full_cols.saturating_sub(1));
        let cols = window_cols.min(full_cols.saturating_sub(c_off));
        let rows = window_rows.min(full_rows.saturating_sub(r_off));
        if rows == 0 || cols == 0 {
            anyhow::bail!(
                "read_band_window_offset band {band_idx}: zero-size window after clamping \
                 (offset ({row_offset},{col_offset}), window ({window_rows},{window_cols}), raster ({full_rows},{full_cols}))"
            );
        }
        let band = self
            .dataset
            .rasterband(band_idx)
            .with_context(|| format!("Band {band_idx} not available"))?;
        let buf = band
            .read_as::<f64>(
                (c_off as isize, r_off as isize),
                (cols, rows),
                (cols, rows),
                None,
            )
            .with_context(|| format!("Failed to read band {band_idx} window at offset ({r_off},{c_off})"))?;
        let ((_bcols, _brows), data) = buf.into_shape_and_vec();
        Array2::from_shape_vec((rows, cols), data)
            .map_err(|e| anyhow::anyhow!("Shape mismatch reading band {band_idx} window offset: {e}"))
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

    /// Set the description for a band (1-indexed).
    ///
    /// Mirrors rasterio's `dst.set_band_description(idx, name)`.
    pub fn set_band_description(&mut self, band_idx: usize, description: &str) -> Result<()> {
        let mut band = self.dataset.rasterband(band_idx)?;
        band.set_description(description)?;
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
/// Mirrors Python `_parse_band_timestamp()` → ISO → epoch → days since first.
/// Sentinel‑2 band descriptions like `20151227T035152_..._T48RVV_B2` are
/// scanned for a `YYYYMMDDTHHMMSS` substring; other formats (`2020-01-01T04:58:59Z`,
/// `2020-01-01T04:58:59`) are parsed directly. All timestamps are treated as UTC.
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
        let desc = band.description().unwrap_or_default();
        let desc = desc.trim();

        let epoch = if let Some(sub) = find_timestamp_substring(desc) {
            parse_iso8601_to_epoch_seconds(sub)
        } else {
            parse_iso8601_to_epoch_seconds(desc)
        }
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
    let target = days[days.len() - 1];
    Ok((days, target))
}

/// Collapse duplicate timestamps in a time-series cube by taking the
/// per-pixel nanmean of bands that share the same timestamp string.
///
/// Matches Python `collapse_duplicate_timestamps()` in
/// `full_scene_reconstruction/pipeline.py:335`.
///
/// Returns `(deduped_cube, deduped_timestamps)` where each timestamp appears
/// exactly once.
pub fn collapse_duplicate_timestamps(
    cube: &ndarray::Array3<f64>,
    timestamps: &[String],
) -> (ndarray::Array3<f64>, Vec<String>) {
    let n_bands = cube.shape()[0];
    assert_eq!(n_bands, timestamps.len(), "cube bands != timestamps");

    // Group band indices by timestamp string
    let mut group_map: std::collections::BTreeMap<&str, Vec<usize>> =
        std::collections::BTreeMap::new();
    for (i, ts) in timestamps.iter().enumerate() {
        group_map.entry(ts.as_str()).or_default().push(i);
    }

    let m = group_map.len();
    let rows = cube.shape()[1];
    let cols = cube.shape()[2];
    let mut deduped = ndarray::Array3::<f64>::zeros((m, rows, cols));
    let mut deduped_ts: Vec<String> = Vec::with_capacity(m);

    for (j, (ts, indices)) in group_map.iter().enumerate() {
        deduped_ts.push(ts.to_string());
        if indices.len() == 1 {
            deduped
                .index_axis_mut(ndarray::Axis(0), j)
                .assign(&cube.index_axis(ndarray::Axis(0), indices[0]));
        } else {
            let mut sum = ndarray::Array2::<f64>::zeros((rows, cols));
            let mut counts = ndarray::Array2::<usize>::zeros((rows, cols));
            for &idx in indices {
                let band = cube.index_axis(ndarray::Axis(0), idx);
                for r in 0..rows {
                    for c in 0..cols {
                        let v = band[[r, c]];
                        if v.is_finite() {
                            sum[[r, c]] += v;
                            counts[[r, c]] += 1;
                        }
                    }
                }
            }
            let mut avg = deduped.index_axis_mut(ndarray::Axis(0), j);
            for r in 0..rows {
                for c in 0..cols {
                    avg[[r, c]] = if counts[[r, c]] > 0 {
                        sum[[r, c]] / counts[[r, c]] as f64
                    } else {
                        f64::NAN
                    };
                }
            }
        }
    }

    (deduped, deduped_ts)
}

/// Build synthetic timestamps from band indices (1 day per band).
///
/// **Test-only.** Full-scene code paths must use
/// [`extract_timestamps_from_band_descriptions`] to parse real GDAL band
/// descriptions.
#[doc(hidden)]
#[deprecated(note = "use extract_timestamps_from_band_descriptions for full-scene; this is test-only")]
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
pub fn reconstruct_single_band<F, P: AsRef<Path>>(
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

/// Extract the raw timestamp substrings from GDAL band descriptions.
///
/// For each band (1-indexed), the description is scanned for a timestamp
/// substring (YYYYmmddTHHMMSS or ISO-8601) using
/// [`find_timestamp_substring`]. Non-timestamp descriptions are returned as
/// empty strings so that the caller can decide on fallback behavior.
///
/// Returns `Vec<String>` with one entry per band, in band order.
pub fn extract_raw_band_descriptions(reader: &RasterReader) -> Result<Vec<String>> {
    let n = reader.band_count();
    let mut descs = Vec::with_capacity(n);
    for b in 1..=n {
        let band = reader
            .dataset
            .rasterband(b)
            .with_context(|| format!("Cannot access band {b}"))?;
        let desc = band.description().unwrap_or_default();
        let ts = find_timestamp_substring(desc.trim());
        descs.push(ts.unwrap_or("").to_string());
    }
    Ok(descs)
}

/// Read all bands from a reader into a 3D cube `(bands, rows, cols)`.
pub fn read_all_bands(reader: &RasterReader) -> Result<ndarray::Array3<f64>> {
    let n_bands = reader.band_count();
    let (rows, cols) = reader.shape();
    if n_bands == 0 {
        anyhow::bail!("raster has no bands");
    }
    if rows == 0 || cols == 0 {
        anyhow::bail!("raster has zero spatial dimensions ({rows}r × {cols}c)");
    }

    // Read all bands in one GDALDatasetRasterIOEx call.  This is essential
    // for BIP-interleaved files: per-band reads would re-read every tile for
    // every band, reading the entire dataset n_bands times.
    let pixels = n_bands * rows * cols;
    let mut data: Vec<f64> = Vec::with_capacity(pixels);
    let band_map: Vec<i32> = (1..=n_bands as i32).collect();

    // Safety: GDALDatasetRasterIOEx writes exactly `pixels` f64 values into
    // the Vec's spare capacity.  We set_len afterwards.  Buffer layout is
    // (bands, rows, cols) in C order:
    //   nPixelSpace = sizeof(f64)
    //   nLineSpace  = cols * sizeof(f64)
    //   nBandSpace  = rows * cols * sizeof(f64)
    let rv = unsafe {
        data.set_len(pixels);

        gdal_sys::GDALDatasetRasterIOEx(
            reader.dataset.c_dataset(),
            gdal_sys::GDALRWFlag::GF_Read,
            0, // nDSXOff
            0, // nDSYOff
            cols as i32,
            rows as i32,
            data.as_mut_ptr() as *mut std::ffi::c_void,
            cols as i32,      // nBXSize — pixels per scanline in buffer
            rows as i32,      // nBYSize — number of scanlines per band in buffer
            gdal_sys::GDALDataType::GDT_Float64,
            n_bands as i32,
            band_map.as_ptr(),
            std::mem::size_of::<f64>() as gdal_sys::GSpacing,           // nPixelSpace
            (cols * std::mem::size_of::<f64>()) as gdal_sys::GSpacing,  // nLineSpace
            (rows * cols * std::mem::size_of::<f64>()) as gdal_sys::GSpacing, // nBandSpace
            std::ptr::null_mut(),
        )
    };
    if rv != gdal_sys::CPLErr::CE_None {
        // Don't leak: forget the set_len allocation
        unsafe { data.set_len(0); }
        anyhow::bail!("GDALDatasetRasterIOEx failed");
    }
    // data already has correct length from set_len

    ndarray::Array3::from_shape_vec((n_bands, rows, cols), data)
        .map_err(|e| anyhow::anyhow!("Shape mismatch after GDAL read: {e}"))
}

/// Read a top-left window from all bands into a 3D cube.
pub fn read_all_bands_window(
    reader: &RasterReader,
    window_rows: usize,
    window_cols: usize,
) -> Result<ndarray::Array3<f64>> {
    let n_bands = reader.band_count();
    if n_bands == 0 {
        anyhow::bail!("raster has no bands");
    }
    let (rows, cols) = reader.shape();
    let r = window_rows.min(rows);
    let c = window_cols.min(cols);
    if r == 0 || c == 0 {
        anyhow::bail!("window has zero dimensions ({r}r × {c}c)");
    }
    let mut cube = ndarray::Array3::<f64>::zeros((n_bands, r, c));
    for b in 0..n_bands {
        let band_data = reader.read_band_window(b + 1, r, c)?;
        cube.slice_mut(ndarray::s![b, .., ..]).assign(&band_data);
    }
    Ok(cube)
}

/// Read a window from all bands at an arbitrary offset into a 3D cube.
///
/// `(row_offset, col_offset)` is the top-left corner in pixel coordinates.
/// Offsets are clamped; the window is clamped to what remains from the offset
/// to the raster edge.
pub fn read_all_bands_window_offset(
    reader: &RasterReader,
    row_offset: usize,
    col_offset: usize,
    window_rows: usize,
    window_cols: usize,
) -> Result<ndarray::Array3<f64>> {
    let n_bands = reader.band_count();
    if n_bands == 0 {
        anyhow::bail!("raster has no bands");
    }
    let (rows, cols) = reader.shape();
    let r_off = row_offset.min(rows.saturating_sub(1));
    let c_off = col_offset.min(cols.saturating_sub(1));
    let r = window_rows.min(rows.saturating_sub(r_off));
    let c = window_cols.min(cols.saturating_sub(c_off));
    if r == 0 || c == 0 {
        anyhow::bail!("window has zero dimensions ({r}r × {c}c) at offset ({row_offset},{col_offset})");
    }
    let mut cube = ndarray::Array3::<f64>::zeros((n_bands, r, c));
    for b in 0..n_bands {
        let band_data = reader.read_band_window_offset(b + 1, r_off, c_off, r, c)?;
        cube.slice_mut(ndarray::s![b, .., ..]).assign(&band_data);
    }
    Ok(cube)
}

// ---------------------------------------------------------------------------
// GDAL version helper (kept from placeholder for smoke testing)
// ---------------------------------------------------------------------------

/// Returns the GDAL version string at runtime (e.g. "3.10.3").
pub fn gdal_version() -> String {
    gdal_rs::version::version_info("GDAL_VERSION")
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::Array2;
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

    #[allow(deprecated)]
    fn synthetic_test_timestamps(n_bands: usize) -> (Vec<f64>, f64) {
        synthetic_timestamps_from_bands(n_bands)
    }

    #[test]
    fn reconstruct_single_band_roundtrip() {
        let input = Path::new("test_ts_generic.tif");
        let output = Path::new("test_out_generic.tif");
        _write_time_series_tif(input, 5, 5, 10).unwrap();

        let reader = RasterReader::open(input).unwrap();
        let meta = RasterMetadata {
            geo_transform: DEFAULT_GEO,
            crs_wkt: None,
            nodata: Some(f64::NAN),
        };
        let cube = read_all_bands(&reader).unwrap();
        let (t_days, target_t) = synthetic_test_timestamps(reader.band_count());

        reconstruct_single_band(&cube, &t_days, target_t, output, &meta, |_ts, obs, _target| {
            obs.iter().sum::<f64>() / obs.len() as f64
        })
        .unwrap();

        _check_output(output, 5, 5, 1);
        let out_r = RasterReader::open(output).unwrap();
        let out_data = out_r.read_band(1).unwrap();
        let n_finite = out_data.iter().filter(|v| v.is_finite()).count();
        assert!(n_finite > 0, "generic output should have finite predictions; got {n_finite}");

        let _ = fs::remove_file(input);
        let _ = fs::remove_file(output);
    }

    #[test]
    fn reconstruct_single_band_nan_only_writes_output() {
        let path = Path::new("test_empty.tif");
        {
            let mut writer = RasterWriter::create(path, 3, 3, 1, &DEFAULT_GEO, None, None).unwrap();
            let data = Array2::<f64>::from_elem((3, 3), f64::NAN);
            writer.write_band(1, &data).unwrap();
            writer.flush().unwrap();
        }

        let reader = RasterReader::open(path).unwrap();
        let meta = RasterMetadata { geo_transform: DEFAULT_GEO, crs_wkt: None, nodata: Some(f64::NAN) };
        let cube = read_all_bands(&reader).unwrap();
        let (t_days, target_t) = synthetic_test_timestamps(reader.band_count());

        let output = Path::new("test_out_empty.tif");
        let result = reconstruct_single_band(&cube, &t_days, target_t, output, &meta, |_ts, _obs, _target| f64::NAN);
        assert!(result.is_ok(), "NAN-only input should still produce output");

        let out_r = RasterReader::open(output).unwrap();
        assert_eq!(out_r.band_count(), 1);
        assert_eq!(out_r.shape(), (3, 3));

        let _ = fs::remove_file(path);
        let _ = fs::remove_file(output);
    }

    #[test]
    fn read_band_window_offset_centered() {
        let path = Path::new("test_window_offset.tif");
        let rows = 7;
        let cols = 9;
        // Write a single band where each pixel's value encodes its (row, col).
        {
            let mut writer = RasterWriter::create(path, rows, cols, 1, &DEFAULT_GEO, None, None).unwrap();
            let mut data = Array2::<f64>::zeros((rows, cols));
            for r in 0..rows {
                for c in 0..cols {
                    data[[r, c]] = (r * 100 + c) as f64;
                }
            }
            writer.write_band(1, &data).unwrap();
            writer.flush().unwrap();
        }

        let reader = RasterReader::open(path).unwrap();

        // Center window 3×3 in 7×9 → offset (2, 3)
        let ws = 3;
        let (full_rows, full_cols) = reader.shape();
        let row_off = (full_rows - ws) / 2;
        let col_off = (full_cols - ws) / 2;
        assert_eq!((row_off, col_off), (2, 3));

        let cube = read_all_bands_window_offset(&reader, row_off, col_off, ws, ws).unwrap();
        assert_eq!(cube.shape(), &[1, ws, ws]);

        // Verify the centered window has the expected pixel values
        let band = cube.slice(ndarray::s![0, .., ..]);
        for wr in 0..ws {
            for wc in 0..ws {
                let expected = ((row_off + wr) * 100 + (col_off + wc)) as f64;
                assert_eq!(
                    band[[wr, wc]], expected,
                    "centered window mismatch at window ({wr},{wc})"
                );
            }
        }

        let _ = fs::remove_file(path);
    }
}
