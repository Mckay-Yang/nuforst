use std::collections::BTreeMap;
use std::fs::{self, File};
use std::io::{BufWriter, Write};
use std::path::{Path, PathBuf};

use anyhow::{bail, Context, Result};
use memmap2::Mmap;
use ndarray::Array3;
use serde::{Deserialize, Serialize};
use serde_json::Value;

use crate::full_scene::{self, discover_sentinel_band_stacks, sorted_band_names};
use crate::{
    extract_raw_band_descriptions, read_all_bands, read_all_bands_window_offset, RasterMetadata,
    RasterReader,
};

const SCENE_CACHE_VERSION: u32 = 1;
const SCENE_CACHE_LAYOUT: &str = "band_time_row_col";
const SCENE_CACHE_DTYPE: &str = "f32_le";

#[derive(Debug, Clone, Serialize, Deserialize)]
struct SceneCacheMeta {
    version: u32,
    source_name: String,
    lon: f64,
    lat: f64,
    rows: usize,
    cols: usize,
    layout: String,
    dtype: String,
    bands: Vec<String>,
    band_meta: Vec<SceneCacheBandMeta>,
    total_values: usize,
    geo_transform: [f64; 6],
    crs_wkt: Option<String>,
    nodata: Option<f64>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct SceneCacheBandMeta {
    name: String,
    timestamps: Vec<String>,
    time_len: usize,
    offset_values: usize,
}

#[derive(Debug)]
pub struct LoadedScene {
    pub ordered_bands: Vec<String>,
    pub band_cubes: BTreeMap<String, Array3<f64>>,
    pub band_timestamps: BTreeMap<String, Vec<String>>,
    pub meta: RasterMetadata,
    pub cache_dir: Option<PathBuf>,
}

pub fn default_scene_cache_dir(
    cache_root: &Path,
    source_name: &str,
    lon: f64,
    lat: f64,
) -> PathBuf {
    cache_root
        .join(safe_cache_component(source_name))
        .join(full_scene::location_output_token(lon, lat))
}

pub fn load_or_build_scene_cache(
    data_root: &Path,
    cache_root: &Path,
    source_name: &str,
    lon: f64,
    lat: f64,
) -> Result<LoadedScene> {
    let cache_dir = default_scene_cache_dir(cache_root, source_name, lon, lat);
    if is_valid_scene_cache(&cache_dir) {
        eprintln!("Loading scene from cache: {}", cache_dir.display());
        return load_scene_cache(&cache_dir);
    }

    eprintln!("Scene cache missing; building {}", cache_dir.display());
    build_scene_cache(data_root, cache_root, source_name, lon, lat, None)?;
    load_scene_cache(&cache_dir)
}

pub fn build_scene_cache(
    data_root: &Path,
    cache_root: &Path,
    source_name: &str,
    lon: f64,
    lat: f64,
    explicit_output: Option<&Path>,
) -> Result<PathBuf> {
    let cache_dir = explicit_output
        .map(Path::to_path_buf)
        .unwrap_or_else(|| default_scene_cache_dir(cache_root, source_name, lon, lat));
    fs::create_dir_all(&cache_dir)
        .with_context(|| format!("Cannot create cache dir {}", cache_dir.display()))?;

    let band_stacks = resolve_full_scene_band_stacks(data_root, source_name, lon, lat)?;
    let ordered_bands: Vec<String> = sorted_band_names(&band_stacks)
        .into_iter()
        .map(str::to_string)
        .collect();

    let bin_path = cache_dir.join("cube.f32.bin");
    let meta_path = cache_dir.join("meta.json");
    let bin_tmp = cache_dir.join("cube.f32.bin.tmp");
    let meta_tmp = cache_dir.join("meta.json.tmp");

    let mut writer = BufWriter::with_capacity(
        16 * 1024 * 1024,
        File::create(&bin_tmp)
            .with_context(|| format!("Cannot create cache data file {}", bin_tmp.display()))?,
    );

    let mut band_meta = Vec::new();
    let mut total_values = 0usize;
    let mut rows_cols: Option<(usize, usize)> = None;
    let mut scene_meta: Option<RasterMetadata> = None;

    for band_name in &ordered_bands {
        let chunk_paths = band_stacks
            .get(band_name)
            .with_context(|| format!("Missing stack paths for band {band_name}"))?;
        eprintln!(
            "Caching band {band_name} from {} chunk(s)...",
            chunk_paths.len()
        );
        let (cube, timestamps, meta) = merge_band_chunks(band_name, chunk_paths, None)?;
        let (time_len, rows, cols) = cube.dim();
        if let Some((expected_rows, expected_cols)) = rows_cols {
            if rows != expected_rows || cols != expected_cols {
                bail!(
                    "Band {band_name} shape mismatch: expected {expected_rows}x{expected_cols}, got {rows}x{cols}"
                );
            }
        } else {
            rows_cols = Some((rows, cols));
            scene_meta = Some(meta);
        }

        if timestamps.len() != time_len {
            bail!(
                "Band {band_name} timestamp count {} does not match time slices {time_len}",
                timestamps.len()
            );
        }

        let offset_values = total_values;
        let slice = cube
            .as_slice()
            .context("Scene cache requires contiguous ndarray storage")?;
        write_f32_values(&mut writer, slice)?;
        total_values += slice.len();
        band_meta.push(SceneCacheBandMeta {
            name: band_name.clone(),
            timestamps,
            time_len,
            offset_values,
        });
        eprintln!("  Cached band {band_name}: {:?}", cube.dim());
    }
    writer.flush()?;
    drop(writer);

    let (rows, cols) = rows_cols.context("No bands were cached")?;
    let scene_meta = scene_meta.context("No raster metadata available for cache")?;
    let meta = SceneCacheMeta {
        version: SCENE_CACHE_VERSION,
        source_name: source_name.to_string(),
        lon,
        lat,
        rows,
        cols,
        layout: SCENE_CACHE_LAYOUT.to_string(),
        dtype: SCENE_CACHE_DTYPE.to_string(),
        bands: ordered_bands,
        band_meta,
        total_values,
        geo_transform: scene_meta.geo_transform,
        crs_wkt: scene_meta.crs_wkt,
        nodata: scene_meta.nodata,
    };
    fs::write(&meta_tmp, serde_json::to_vec_pretty(&meta)?)?;
    fs::rename(&bin_tmp, &bin_path)?;
    fs::rename(&meta_tmp, &meta_path)?;

    eprintln!("Scene cache written to {}", cache_dir.display());
    Ok(cache_dir)
}

pub fn load_scene_cache(cache_dir: &Path) -> Result<LoadedScene> {
    let meta_path = cache_dir.join("meta.json");
    let bin_path = cache_dir.join("cube.f32.bin");
    let meta: SceneCacheMeta = serde_json::from_slice(
        &fs::read(&meta_path)
            .with_context(|| format!("Cannot read scene cache meta {}", meta_path.display()))?,
    )?;
    if meta.version != SCENE_CACHE_VERSION {
        bail!(
            "Unsupported scene cache version {}; expected {}",
            meta.version,
            SCENE_CACHE_VERSION
        );
    }
    if meta.layout != SCENE_CACHE_LAYOUT || meta.dtype != SCENE_CACHE_DTYPE {
        bail!(
            "Unsupported scene cache layout/dtype: {}/{}",
            meta.layout,
            meta.dtype
        );
    }

    let file = File::open(&bin_path)
        .with_context(|| format!("Cannot open scene cache data {}", bin_path.display()))?;
    let expected_bytes = meta
        .total_values
        .checked_mul(4)
        .context("Scene cache byte size overflow")?;
    let actual_bytes = file.metadata()?.len() as usize;
    if actual_bytes != expected_bytes {
        bail!(
            "Scene cache data size mismatch: expected {expected_bytes} bytes, got {actual_bytes}"
        );
    }
    let mmap = unsafe { Mmap::map(&file)? };

    let mut band_cubes = BTreeMap::new();
    let mut band_timestamps = BTreeMap::new();
    for band in &meta.band_meta {
        let n_values = band
            .time_len
            .checked_mul(meta.rows)
            .and_then(|v| v.checked_mul(meta.cols))
            .context("Scene cache band size overflow")?;
        let start = band
            .offset_values
            .checked_mul(4)
            .context("Scene cache offset overflow")?;
        let end = start
            .checked_add(n_values * 4)
            .context("Scene cache offset overflow")?;
        if end > mmap.len() {
            bail!("Scene cache band {} extends beyond data file", band.name);
        }

        let mut cube = Array3::<f64>::zeros((band.time_len, meta.rows, meta.cols));
        let out = cube
            .as_slice_mut()
            .context("Scene cache requires contiguous ndarray storage")?;
        for (idx, dst) in out.iter_mut().enumerate() {
            let byte_idx = start + idx * 4;
            let raw = [
                mmap[byte_idx],
                mmap[byte_idx + 1],
                mmap[byte_idx + 2],
                mmap[byte_idx + 3],
            ];
            *dst = f32::from_le_bytes(raw) as f64;
        }
        eprintln!("Loaded cached band {}: {:?}", band.name, cube.dim());
        band_cubes.insert(band.name.clone(), cube);
        band_timestamps.insert(band.name.clone(), band.timestamps.clone());
    }

    Ok(LoadedScene {
        ordered_bands: meta.bands,
        band_cubes,
        band_timestamps,
        meta: RasterMetadata {
            geo_transform: meta.geo_transform,
            crs_wkt: meta.crs_wkt,
            nodata: meta.nodata,
        },
        cache_dir: Some(cache_dir.to_path_buf()),
    })
}

pub fn load_cached_target_timestamp(
    cache_dir: &Path,
    min_valid_ratio: f64,
    late_fraction: f64,
) -> Result<Option<String>> {
    let path = target_cache_path(cache_dir);
    if !path.is_file() {
        return Ok(None);
    }
    let value: Value = serde_json::from_slice(
        &fs::read(&path)
            .with_context(|| format!("Cannot read target timestamp cache {}", path.display()))?,
    )?;
    let key = target_cache_key(min_valid_ratio, late_fraction);
    Ok(value
        .get("targets")
        .and_then(|targets| targets.get(&key))
        .and_then(|entry| entry.get("target_time"))
        .and_then(Value::as_str)
        .map(str::to_string))
}

pub fn store_cached_target_timestamp(
    cache_dir: &Path,
    min_valid_ratio: f64,
    late_fraction: f64,
    target_time: &str,
) -> Result<()> {
    let path = target_cache_path(cache_dir);
    let mut value: Value =
        if path.is_file() {
            serde_json::from_slice(&fs::read(&path).with_context(|| {
                format!("Cannot read target timestamp cache {}", path.display())
            })?)?
        } else {
            serde_json::json!({
                "version": 1,
                "targets": {}
            })
        };
    if !value.get("targets").is_some_and(Value::is_object) {
        value["targets"] = serde_json::json!({});
    }
    let key = target_cache_key(min_valid_ratio, late_fraction);
    value["targets"][key] = serde_json::json!({
        "target_time": target_time,
        "min_valid_ratio": min_valid_ratio,
        "late_fraction": late_fraction,
    });
    fs::write(&path, serde_json::to_vec_pretty(&value)?)
        .with_context(|| format!("Cannot write target timestamp cache {}", path.display()))?;
    Ok(())
}

pub fn load_scene_from_geotiffs_window(
    data_root: &Path,
    source_name: &str,
    lon: f64,
    lat: f64,
    window_size: usize,
    window_lon: Option<f64>,
    window_lat: Option<f64>,
) -> Result<LoadedScene> {
    let band_stacks = resolve_full_scene_band_stacks(data_root, source_name, lon, lat)?;
    let ordered_bands: Vec<String> = sorted_band_names(&band_stacks)
        .into_iter()
        .map(str::to_string)
        .collect();
    let mut band_cubes = BTreeMap::new();
    let mut band_timestamps = BTreeMap::new();
    let mut scene_meta = None;
    let window = Some((window_size, window_lon, window_lat));

    for band_name in &ordered_bands {
        let chunk_paths = band_stacks
            .get(band_name)
            .with_context(|| format!("Missing stack paths for band {band_name}"))?;
        eprintln!(
            "Loading band {band_name} from {} chunk(s)...",
            chunk_paths.len()
        );
        let (cube, timestamps, meta) = merge_band_chunks(band_name, chunk_paths, window)?;
        if scene_meta.is_none() {
            scene_meta = Some(meta);
        }
        eprintln!("  Loaded band {band_name}: {:?}", cube.dim());
        band_cubes.insert(band_name.clone(), cube);
        band_timestamps.insert(band_name.clone(), timestamps);
    }

    Ok(LoadedScene {
        ordered_bands,
        band_cubes,
        band_timestamps,
        meta: scene_meta.context("No raster metadata loaded")?,
        cache_dir: None,
    })
}

pub fn full_scene_window_offset(
    reader: &RasterReader,
    window_size: usize,
    window_lon: Option<f64>,
    window_lat: Option<f64>,
) -> Result<(usize, usize)> {
    let (rows, cols) = reader.shape();
    let ws = window_size.min(rows).min(cols);
    if let (Some(lon), Some(lat)) = (window_lon, window_lat) {
        let gt = reader
            .geo_transform()
            .context("--window-lon/--window-lat require a georeferenced raster")?;
        let det = gt[1] * gt[5] - gt[2] * gt[4];
        if det.abs() <= 1e-15 {
            bail!("Cannot invert raster geotransform for --window-lon/--window-lat");
        }
        let dx = lon - gt[0];
        let dy = lat - gt[3];
        let col = (gt[5] * dx - gt[2] * dy) / det;
        let row = (-gt[4] * dx + gt[1] * dy) / det;
        let center_col = col.floor() as isize;
        let center_row = row.floor() as isize;
        let half = (ws / 2) as isize;
        let max_row_off = rows.saturating_sub(ws) as isize;
        let max_col_off = cols.saturating_sub(ws) as isize;
        let row_off = (center_row - half).clamp(0, max_row_off) as usize;
        let col_off = (center_col - half).clamp(0, max_col_off) as usize;
        Ok((row_off, col_off))
    } else {
        let row_off = if rows > ws { (rows - ws) / 2 } else { 0 };
        let col_off = if cols > ws { (cols - ws) / 2 } else { 0 };
        Ok((row_off, col_off))
    }
}

fn is_valid_scene_cache(cache_dir: &Path) -> bool {
    cache_dir.join("meta.json").is_file() && cache_dir.join("cube.f32.bin").is_file()
}

fn target_cache_path(cache_dir: &Path) -> PathBuf {
    cache_dir.join("target_timestamps.json")
}

fn target_cache_key(min_valid_ratio: f64, late_fraction: f64) -> String {
    format!(
        "min_valid_ratio={:.6};late_fraction={:.6}",
        min_valid_ratio, late_fraction
    )
}

fn safe_cache_component(value: &str) -> String {
    value
        .chars()
        .map(|ch| {
            if ch.is_ascii_alphanumeric() || ch == '-' || ch == '_' {
                ch
            } else {
                '_'
            }
        })
        .collect()
}

fn resolve_full_scene_band_stacks(
    data_root: &Path,
    source_name: &str,
    lon: f64,
    lat: f64,
) -> Result<BTreeMap<String, Vec<PathBuf>>> {
    let source_dir = data_root.join(source_name);
    let mut band_stacks = discover_sentinel_band_stacks(&source_dir, lon, lat)?;
    if band_stacks.is_empty() {
        bail!(
            "No band stacks found for lon={}, lat={} in {}",
            lon,
            lat,
            source_dir.display()
        );
    }

    let vrt_dir = data_root.join("cache").join("local").join("vrts");
    let loc_token6 = full_scene::location_output_token(lon, lat);
    for (band_name, paths) in band_stacks.iter_mut() {
        if paths.len() <= 1 {
            continue;
        }

        let vrt_path = vrt_dir.join(format!("sentinel_{band_name}_{loc_token6}.vrt"));
        if vrt_path.exists() {
            eprintln!(
                "Band {band_name}: using cached mosaic VRT {} for {} source chunks",
                vrt_path.display(),
                paths.len()
            );
            *paths = vec![vrt_path];
        } else {
            bail!(
                "Band {band_name} has {} source chunks but no cached mosaic VRT at {}. \
                 Build or restore the VRT before full-scene reconstruction.",
                paths.len(),
                vrt_path.display()
            );
        }
    }

    Ok(band_stacks)
}

fn merge_band_chunks(
    band_name: &str,
    chunk_paths: &[PathBuf],
    window: Option<(usize, Option<f64>, Option<f64>)>,
) -> Result<(Array3<f64>, Vec<String>, RasterMetadata)> {
    let mut cube_parts: Vec<Array3<f64>> = Vec::new();
    let mut ts_parts = Vec::new();
    let mut meta = None;

    for (chunk_idx, chunk_path) in chunk_paths.iter().enumerate() {
        eprintln!(
            "  Reading {band_name} chunk {}/{}: {}",
            chunk_idx + 1,
            chunk_paths.len(),
            chunk_path.display()
        );
        let reader = RasterReader::open(chunk_path)?;
        if meta.is_none() {
            meta = Some(metadata_from_reader(&reader));
        }
        let cube = if let Some((ws, window_lon, window_lat)) = window {
            let (row_off, col_off) = full_scene_window_offset(&reader, ws, window_lon, window_lat)?;
            read_all_bands_window_offset(&reader, row_off, col_off, ws, ws)?
        } else {
            read_all_bands(&reader)?
        };
        let descs = extract_raw_band_descriptions(&reader)?;
        cube_parts.push(cube);
        ts_parts.extend(descs);
    }

    let cube_merged = if cube_parts.len() == 1 {
        cube_parts.remove(0)
    } else {
        let views: Vec<_> = cube_parts.iter().map(|c| c.view()).collect();
        ndarray::concatenate(ndarray::Axis(0), &views)?
    };

    let meta = meta.with_context(|| format!("No chunks found for band {band_name}"))?;
    Ok((cube_merged, ts_parts, meta))
}

fn metadata_from_reader(reader: &RasterReader) -> RasterMetadata {
    RasterMetadata {
        geo_transform: reader
            .geo_transform()
            .unwrap_or([0.0, 1.0, 0.0, 0.0, 0.0, -1.0]),
        crs_wkt: reader.crs_wkt(),
        nodata: reader.nodata(1),
    }
}

fn write_f32_values<W: Write>(writer: &mut W, values: &[f64]) -> Result<()> {
    const BUF_LIMIT: usize = 16 * 1024 * 1024;
    let mut bytes = Vec::with_capacity(BUF_LIMIT + 4);
    for &value in values {
        bytes.extend_from_slice(&(value as f32).to_le_bytes());
        if bytes.len() >= BUF_LIMIT {
            writer.write_all(&bytes)?;
            bytes.clear();
        }
    }
    if !bytes.is_empty() {
        writer.write_all(&bytes)?;
    }
    Ok(())
}
