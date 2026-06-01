// nufrost-gdal — raster I/O via the GDAL crate.
// Requires libgdal system library (e.g. `brew install gdal` or conda).
//
// Provides:
//  - RasterReader: open GeoTIFF/VRT, read metadata and band data, build valid masks
//  - RasterWriter: create single-band and multi-band GeoTIFF output
//  - write_zhu2015_output: 2-band (prediction + QA) output matching Python convention

use std::path::Path;

use anyhow::{Context, Result};
use gdal::raster::Buffer;
use gdal::spatial_ref::SpatialRef;
use gdal::{Dataset, DriverManager, GeoTransform};
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
}
