use std::collections::BTreeMap;
use std::fs::{self, File};
use std::io::{BufWriter, Write};
use std::path::{Path, PathBuf};
use std::process::Command;

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
    #[serde(default)]
    source_fingerprint: Option<String>,
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
    let source_component = safe_cache_component(source_name);
    let root = if cache_root
        .parent()
        .and_then(Path::file_name)
        .and_then(|name| name.to_str())
        == Some(source_component.as_str())
    {
        cache_root.to_path_buf()
    } else {
        cache_root.join(source_component)
    };
    root.join(full_scene::location_output_token(lon, lat))
}

fn source_data_dir(data_root: &Path, source_name: &str) -> PathBuf {
    let preferred = data_root.join("raw").join(source_name).join("16-sites");
    if preferred.is_dir() {
        preferred
    } else {
        data_root.join(source_name)
    }
}

fn scene_vrt_dir(data_root: &Path, source_name: &str) -> PathBuf {
    data_root
        .join("cache")
        .join(source_name)
        .join("16-sites")
        .join("vrt")
}

pub fn load_or_build_scene_cache(
    data_root: &Path,
    cache_root: &Path,
    source_name: &str,
    lon: f64,
    lat: f64,
) -> Result<LoadedScene> {
    let cache_dir = default_scene_cache_dir(cache_root, source_name, lon, lat);
    let raw_band_stacks = discover_full_scene_band_stacks(data_root, source_name, lon, lat)?;
    let source_fingerprint = source_fingerprint_for_band_stacks(&raw_band_stacks)?;
    if is_valid_scene_cache(&cache_dir, &source_fingerprint)? {
        eprintln!("Loading scene from cache: {}", cache_dir.display());
        return load_scene_cache(&cache_dir);
    }

    if cache_dir.exists() {
        eprintln!(
            "Scene cache stale or incomplete; rebuilding {}",
            cache_dir.display()
        );
        fs::remove_dir_all(&cache_dir)
            .with_context(|| format!("Cannot remove stale cache dir {}", cache_dir.display()))?;
    } else {
        eprintln!("Scene cache missing; building {}", cache_dir.display());
    }
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

    let raw_band_stacks = discover_full_scene_band_stacks(data_root, source_name, lon, lat)?;
    let source_fingerprint = source_fingerprint_for_band_stacks(&raw_band_stacks)?;
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
        source_fingerprint: Some(source_fingerprint),
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

fn is_valid_scene_cache(cache_dir: &Path, expected_source_fingerprint: &str) -> Result<bool> {
    let meta_path = cache_dir.join("meta.json");
    let bin_path = cache_dir.join("cube.f32.bin");
    if !meta_path.is_file() || !bin_path.is_file() {
        return Ok(false);
    }

    let meta: SceneCacheMeta = match serde_json::from_slice(
        &fs::read(&meta_path)
            .with_context(|| format!("Cannot read scene cache meta {}", meta_path.display()))?,
    ) {
        Ok(meta) => meta,
        Err(_) => return Ok(false),
    };
    Ok(meta.version == SCENE_CACHE_VERSION
        && meta.layout == SCENE_CACHE_LAYOUT
        && meta.dtype == SCENE_CACHE_DTYPE
        && meta.source_fingerprint.as_deref() == Some(expected_source_fingerprint))
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
    let mut band_stacks = discover_full_scene_band_stacks(data_root, source_name, lon, lat)?;
    let vrt_dir = scene_vrt_dir(data_root, source_name);
    let loc_token6 = full_scene::location_output_token(lon, lat);
    for (band_name, paths) in band_stacks.iter_mut() {
        if paths.len() <= 1 {
            continue;
        }

        let vrt_path = vrt_dir.join(format!("sentinel_{band_name}_{loc_token6}.vrt"));
        ensure_mosaic_vrt(&vrt_path, paths)?;
        eprintln!(
            "Band {band_name}: using mosaic VRT {} for {} source chunks",
            vrt_path.display(),
            paths.len()
        );
        *paths = vec![vrt_path];
    }

    Ok(band_stacks)
}

fn discover_full_scene_band_stacks(
    data_root: &Path,
    source_name: &str,
    lon: f64,
    lat: f64,
) -> Result<BTreeMap<String, Vec<PathBuf>>> {
    let source_dir = source_data_dir(data_root, source_name);
    let band_stacks = discover_sentinel_band_stacks(&source_dir, lon, lat)?;
    if band_stacks.is_empty() {
        bail!(
            "No band stacks found for lon={}, lat={} in {}",
            lon,
            lat,
            source_dir.display()
        );
    }
    Ok(band_stacks)
}

fn ensure_mosaic_vrt(vrt_path: &Path, source_paths: &[PathBuf]) -> Result<()> {
    if source_paths.len() <= 1 {
        bail!(
            "Cannot build mosaic VRT {} from fewer than two source chunks",
            vrt_path.display()
        );
    }
    if !vrt_path.exists() {
        let parent = vrt_path
            .parent()
            .with_context(|| format!("VRT path has no parent: {}", vrt_path.display()))?;
        fs::create_dir_all(parent)
            .with_context(|| format!("Cannot create VRT directory {}", parent.display()))?;

        let mut cmd = Command::new("gdalbuildvrt");
        cmd.arg("-overwrite").arg(vrt_path);
        for path in source_paths {
            cmd.arg(path);
        }
        eprintln!(
            "Building mosaic VRT {} from {} source chunks...",
            vrt_path.display(),
            source_paths.len()
        );
        let output = cmd
            .output()
            .with_context(|| "Failed to run gdalbuildvrt. Ensure GDAL is available on PATH.")?;
        if !output.status.success() {
            bail!(
                "gdalbuildvrt failed for {} with status {:?}\nstdout:\n{}\nstderr:\n{}",
                vrt_path.display(),
                output.status.code(),
                String::from_utf8_lossy(&output.stdout),
                String::from_utf8_lossy(&output.stderr)
            );
        }
    }
    let first_source = source_paths
        .first()
        .with_context(|| format!("No source chunks for VRT {}", vrt_path.display()))?;
    let reader = RasterReader::open(first_source)?;
    let descriptions = extract_raw_band_descriptions(&reader)?;
    write_vrt_band_descriptions(vrt_path, &descriptions)?;
    Ok(())
}

fn write_vrt_band_descriptions(vrt_path: &Path, descriptions: &[String]) -> Result<()> {
    let mut xml = fs::read_to_string(vrt_path)
        .with_context(|| format!("Cannot read generated VRT {}", vrt_path.display()))?;
    let mut search_from = 0usize;
    for desc in descriptions {
        let Some(rel_start) = xml[search_from..].find("<VRTRasterBand") else {
            break;
        };
        let tag_start = search_from + rel_start;
        let Some(rel_tag_end) = xml[tag_start..].find('>') else {
            break;
        };
        let insert_at = tag_start + rel_tag_end + 1;
        let next_band = xml[insert_at..]
            .find("<VRTRasterBand")
            .map(|offset| insert_at + offset)
            .unwrap_or(xml.len());
        if !desc.is_empty() && !xml[insert_at..next_band].contains("<Description>") {
            let escaped = escape_xml_text(desc);
            let insertion = format!("\n    <Description>{escaped}</Description>");
            xml.insert_str(insert_at, &insertion);
            search_from = insert_at + insertion.len();
        } else {
            search_from = insert_at;
        }
    }
    fs::write(vrt_path, xml)
        .with_context(|| format!("Cannot write generated VRT {}", vrt_path.display()))?;
    Ok(())
}

fn escape_xml_text(value: &str) -> String {
    let mut escaped = String::with_capacity(value.len());
    for ch in value.chars() {
        match ch {
            '&' => escaped.push_str("&amp;"),
            '<' => escaped.push_str("&lt;"),
            '>' => escaped.push_str("&gt;"),
            '"' => escaped.push_str("&quot;"),
            '\'' => escaped.push_str("&apos;"),
            _ => escaped.push(ch),
        }
    }
    escaped
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

fn source_fingerprint_for_band_stacks(
    band_stacks: &BTreeMap<String, Vec<PathBuf>>,
) -> Result<String> {
    let mut manifest = String::new();
    for (band, paths) in band_stacks {
        manifest.push_str("band=");
        manifest.push_str(band);
        manifest.push('\n');
        for path in paths {
            let meta = fs::metadata(path)
                .with_context(|| format!("Cannot stat source file {}", path.display()))?;
            let modified = meta
                .modified()
                .ok()
                .and_then(|t| t.duration_since(std::time::UNIX_EPOCH).ok())
                .map(|d| d.as_nanos())
                .unwrap_or(0);
            manifest.push_str("path=");
            manifest.push_str(&path.display().to_string());
            manifest.push('\n');
            manifest.push_str("len=");
            manifest.push_str(&meta.len().to_string());
            manifest.push('\n');
            manifest.push_str("mtime_ns=");
            manifest.push_str(&modified.to_string());
            manifest.push('\n');
        }
    }
    Ok(md5_hex(manifest.as_bytes()))
}

fn md5_hex(input: &[u8]) -> String {
    const S: [u32; 64] = [
        7, 12, 17, 22, 7, 12, 17, 22, 7, 12, 17, 22, 7, 12, 17, 22, 5, 9, 14, 20, 5, 9, 14, 20, 5,
        9, 14, 20, 5, 9, 14, 20, 4, 11, 16, 23, 4, 11, 16, 23, 4, 11, 16, 23, 4, 11, 16, 23, 6, 10,
        15, 21, 6, 10, 15, 21, 6, 10, 15, 21, 6, 10, 15, 21,
    ];
    const K: [u32; 64] = [
        0xd76aa478, 0xe8c7b756, 0x242070db, 0xc1bdceee, 0xf57c0faf, 0x4787c62a, 0xa8304613,
        0xfd469501, 0x698098d8, 0x8b44f7af, 0xffff5bb1, 0x895cd7be, 0x6b901122, 0xfd987193,
        0xa679438e, 0x49b40821, 0xf61e2562, 0xc040b340, 0x265e5a51, 0xe9b6c7aa, 0xd62f105d,
        0x02441453, 0xd8a1e681, 0xe7d3fbc8, 0x21e1cde6, 0xc33707d6, 0xf4d50d87, 0x455a14ed,
        0xa9e3e905, 0xfcefa3f8, 0x676f02d9, 0x8d2a4c8a, 0xfffa3942, 0x8771f681, 0x6d9d6122,
        0xfde5380c, 0xa4beea44, 0x4bdecfa9, 0xf6bb4b60, 0xbebfbc70, 0x289b7ec6, 0xeaa127fa,
        0xd4ef3085, 0x04881d05, 0xd9d4d039, 0xe6db99e5, 0x1fa27cf8, 0xc4ac5665, 0xf4292244,
        0x432aff97, 0xab9423a7, 0xfc93a039, 0x655b59c3, 0x8f0ccc92, 0xffeff47d, 0x85845dd1,
        0x6fa87e4f, 0xfe2ce6e0, 0xa3014314, 0x4e0811a1, 0xf7537e82, 0xbd3af235, 0x2ad7d2bb,
        0xeb86d391,
    ];

    let mut msg = input.to_vec();
    let bit_len = (msg.len() as u64).wrapping_mul(8);
    msg.push(0x80);
    while msg.len() % 64 != 56 {
        msg.push(0);
    }
    msg.extend_from_slice(&bit_len.to_le_bytes());

    let mut a0: u32 = 0x67452301;
    let mut b0: u32 = 0xefcdab89;
    let mut c0: u32 = 0x98badcfe;
    let mut d0: u32 = 0x10325476;

    for chunk in msg.chunks_exact(64) {
        let mut m = [0u32; 16];
        for (i, word) in m.iter_mut().enumerate() {
            let start = i * 4;
            *word = u32::from_le_bytes([
                chunk[start],
                chunk[start + 1],
                chunk[start + 2],
                chunk[start + 3],
            ]);
        }

        let mut a = a0;
        let mut b = b0;
        let mut c = c0;
        let mut d = d0;

        for i in 0..64 {
            let (f, g) = if i < 16 {
                ((b & c) | ((!b) & d), i)
            } else if i < 32 {
                ((d & b) | ((!d) & c), (5 * i + 1) % 16)
            } else if i < 48 {
                (b ^ c ^ d, (3 * i + 5) % 16)
            } else {
                (c ^ (b | (!d)), (7 * i) % 16)
            };
            let temp = d;
            d = c;
            c = b;
            b = b.wrapping_add(
                a.wrapping_add(f)
                    .wrapping_add(K[i])
                    .wrapping_add(m[g])
                    .rotate_left(S[i]),
            );
            a = temp;
        }

        a0 = a0.wrapping_add(a);
        b0 = b0.wrapping_add(b);
        c0 = c0.wrapping_add(c);
        d0 = d0.wrapping_add(d);
    }

    let mut digest = [0u8; 16];
    digest[0..4].copy_from_slice(&a0.to_le_bytes());
    digest[4..8].copy_from_slice(&b0.to_le_bytes());
    digest[8..12].copy_from_slice(&c0.to_le_bytes());
    digest[12..16].copy_from_slice(&d0.to_le_bytes());
    digest.iter().map(|b| format!("{b:02x}")).collect()
}

#[cfg(test)]
mod tests {
    use super::md5_hex;

    #[test]
    fn md5_matches_standard_vectors() {
        assert_eq!(md5_hex(b""), "d41d8cd98f00b204e9800998ecf8427e");
        assert_eq!(md5_hex(b"abc"), "900150983cd24fb0d6963f7d28e17f72");
        assert_eq!(
            md5_hex(b"The quick brown fox jumps over the lazy dog"),
            "9e107d9d372bb6826bd81d3542a419d6"
        );
    }
}
