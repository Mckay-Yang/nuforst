use std::fs::{self, File};
use std::io::{BufWriter, Write};
use std::path::{Path, PathBuf};

use anyhow::{bail, Context, Result};
use memmap2::Mmap;
use serde::{Deserialize, Serialize};

use crate::parse_iso8601_to_epoch_seconds;

const SAMPLE_CACHE_VERSION: u32 = 1;
const SAMPLE_CACHE_DTYPE: &str = "f32_le";
const SAMPLE_CACHE_LAYOUT: &str = "sample_time_band";
const SAMPLE_CACHE_MASK_LAYOUT: &str = "sample_time";
const SAMPLE_CACHE_INDEX_LAYOUT: &str = "sample_scene_row_col_n_times";

#[derive(Debug, Clone)]
pub struct SampleCacheBuildOptions {
    pub scene_cache_root: PathBuf,
    pub source_name: String,
    pub output_dir: PathBuf,
    pub n_samples: usize,
    pub min_joint_valid: usize,
    pub seed: u64,
    pub max_attempts_multiplier: usize,
    pub limit_scenes: Option<usize>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SampleCacheMeta {
    pub version: u32,
    pub source_name: String,
    pub dtype: String,
    pub layout: String,
    pub mask_layout: String,
    pub index_layout: String,
    pub n_samples: usize,
    pub max_times: usize,
    pub n_bands: usize,
    pub bands: Vec<String>,
    pub sample_file: String,
    pub mask_file: String,
    pub scene_time_file: String,
    pub index_file: String,
    pub index_columns: Vec<String>,
    pub min_joint_valid: usize,
    pub seed: u64,
    pub scenes: Vec<SampleCacheSceneMeta>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SampleCacheSceneMeta {
    pub scene_id: usize,
    pub source_name: String,
    pub lon: f64,
    pub lat: f64,
    pub rows: usize,
    pub cols: usize,
    pub n_times: usize,
    pub time_offset: usize,
    pub n_sampled: usize,
    pub cache_dir: String,
}

#[derive(Debug, Clone, Deserialize)]
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
}

#[derive(Debug, Clone, Deserialize)]
struct SceneCacheBandMeta {
    name: String,
    timestamps: Vec<String>,
    time_len: usize,
    offset_values: usize,
}

struct SceneSource {
    cache_dir: PathBuf,
    meta: SceneCacheMeta,
    mmap: Mmap,
}

struct Lcg64 {
    state: u64,
}

impl Lcg64 {
    fn new(seed: u64) -> Self {
        Self {
            state: if seed == 0 {
                0x9e37_79b9_7f4a_7c15
            } else {
                seed
            },
        }
    }

    fn next_u64(&mut self) -> u64 {
        self.state = self
            .state
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        self.state
    }

    fn gen_range(&mut self, upper: usize) -> usize {
        if upper <= 1 {
            0
        } else {
            (self.next_u64() % upper as u64) as usize
        }
    }
}

pub fn build_sample_cache(options: &SampleCacheBuildOptions) -> Result<SampleCacheMeta> {
    if options.n_samples == 0 {
        bail!("n_samples must be > 0");
    }
    if options.max_attempts_multiplier == 0 {
        bail!("max_attempts_multiplier must be > 0");
    }

    let source_root = options.scene_cache_root.join(&options.source_name);
    let mut scenes = discover_scene_sources(&source_root, &options.source_name)?;
    if let Some(limit) = options.limit_scenes {
        scenes.truncate(limit);
    }
    if scenes.is_empty() {
        bail!("No scene caches found in {}", source_root.display());
    }

    let bands = scenes[0].meta.bands.clone();
    let n_bands = bands.len();
    let max_times = scenes
        .iter()
        .map(|scene| scene.meta.band_meta[0].time_len)
        .max()
        .context("No scene time axes found")?;
    if max_times == 0 || n_bands == 0 {
        bail!("Scene caches must contain at least one time step and one band");
    }

    let mut eligible_pixels = Vec::with_capacity(scenes.len());
    let mut total_eligible = 0usize;
    for scene in &scenes {
        let pixels = collect_eligible_pixels(scene, options.min_joint_valid)?;
        eprintln!(
            "Scene lon={:.6}, lat={:.6}: {} eligible pixels with >= {} joint-valid dates",
            scene.meta.lon,
            scene.meta.lat,
            pixels.len(),
            options.min_joint_valid
        );
        total_eligible = total_eligible
            .checked_add(pixels.len())
            .context("eligible pixel count overflow")?;
        eligible_pixels.push(pixels);
    }
    if total_eligible == 0 {
        bail!(
            "No eligible pixels found with --min-joint-valid {}",
            options.min_joint_valid
        );
    }

    fs::create_dir_all(&options.output_dir)
        .with_context(|| format!("Cannot create {}", options.output_dir.display()))?;
    let sample_path = options.output_dir.join("samples.f32.bin");
    let mask_path = options.output_dir.join("mask.u8.bin");
    let time_path = options.output_dir.join("scene_times.f64.bin");
    let index_path = options.output_dir.join("index.u64.bin");
    let meta_path = options.output_dir.join("meta.json");

    let mut sample_writer = BufWriter::with_capacity(16 * 1024 * 1024, File::create(&sample_path)?);
    let mut mask_writer = BufWriter::with_capacity(4 * 1024 * 1024, File::create(&mask_path)?);
    let mut index_writer = BufWriter::with_capacity(4 * 1024 * 1024, File::create(&index_path)?);

    let mut time_writer = BufWriter::with_capacity(1024 * 1024, File::create(&time_path)?);
    let mut scene_meta = Vec::with_capacity(scenes.len());
    let mut time_offset = 0usize;
    for (scene_id, scene) in scenes.iter().enumerate() {
        let timestamps = &scene.meta.band_meta[0].timestamps;
        for ts in timestamps {
            let epoch = parse_iso8601_to_epoch_seconds(ts)
                .with_context(|| format!("Cannot parse timestamp {ts}"))?;
            time_writer.write_all(&epoch.to_le_bytes())?;
        }
        scene_meta.push(SampleCacheSceneMeta {
            scene_id,
            source_name: scene.meta.source_name.clone(),
            lon: scene.meta.lon,
            lat: scene.meta.lat,
            rows: scene.meta.rows,
            cols: scene.meta.cols,
            n_times: timestamps.len(),
            time_offset,
            n_sampled: 0,
            cache_dir: scene.cache_dir.display().to_string(),
        });
        time_offset += timestamps.len();
    }
    time_writer.flush()?;

    let mut rng = Lcg64::new(options.seed);
    let mut accepted = 0usize;

    while accepted < options.n_samples {
        let mut global_idx = rng.gen_range(total_eligible);
        let mut scene_id = 0usize;
        for (idx, pixels) in eligible_pixels.iter().enumerate() {
            if global_idx < pixels.len() {
                scene_id = idx;
                break;
            }
            global_idx -= pixels.len();
        }
        let scene = &scenes[scene_id];
        let pixel_offset = eligible_pixels[scene_id][global_idx] as usize;
        let row = pixel_offset / scene.meta.cols;
        let col = pixel_offset % scene.meta.cols;

        write_sample(
            scene,
            row,
            col,
            max_times,
            &mut sample_writer,
            &mut mask_writer,
        )?;
        write_u64(&mut index_writer, scene_id as u64)?;
        write_u64(&mut index_writer, row as u64)?;
        write_u64(&mut index_writer, col as u64)?;
        write_u64(&mut index_writer, scene.meta.band_meta[0].time_len as u64)?;
        scene_meta[scene_id].n_sampled += 1;
        accepted += 1;

        if accepted % 100_000 == 0 || accepted == options.n_samples {
            eprintln!(
                "sample cache progress: {accepted}/{} accepted from {total_eligible} eligible pixels",
                options.n_samples
            );
        }
    }

    sample_writer.flush()?;
    mask_writer.flush()?;
    index_writer.flush()?;

    let meta = SampleCacheMeta {
        version: SAMPLE_CACHE_VERSION,
        source_name: options.source_name.clone(),
        dtype: SAMPLE_CACHE_DTYPE.to_string(),
        layout: SAMPLE_CACHE_LAYOUT.to_string(),
        mask_layout: SAMPLE_CACHE_MASK_LAYOUT.to_string(),
        index_layout: SAMPLE_CACHE_INDEX_LAYOUT.to_string(),
        n_samples: options.n_samples,
        max_times,
        n_bands,
        bands,
        sample_file: "samples.f32.bin".to_string(),
        mask_file: "mask.u8.bin".to_string(),
        scene_time_file: "scene_times.f64.bin".to_string(),
        index_file: "index.u64.bin".to_string(),
        index_columns: vec![
            "scene_id".to_string(),
            "row".to_string(),
            "col".to_string(),
            "n_times".to_string(),
        ],
        min_joint_valid: options.min_joint_valid,
        seed: options.seed,
        scenes: scene_meta,
    };
    fs::write(&meta_path, serde_json::to_vec_pretty(&meta)?)?;

    Ok(meta)
}

fn discover_scene_sources(source_root: &Path, source_name: &str) -> Result<Vec<SceneSource>> {
    let mut out = Vec::new();
    if !source_root.is_dir() {
        return Ok(out);
    }
    let mut dirs: Vec<PathBuf> = fs::read_dir(source_root)?
        .filter_map(|entry| entry.ok().map(|e| e.path()))
        .filter(|path| path.is_dir())
        .collect();
    dirs.sort();

    for dir in dirs {
        let meta_path = dir.join("meta.json");
        let bin_path = dir.join("cube.f32.bin");
        if !meta_path.is_file() || !bin_path.is_file() {
            continue;
        }
        let meta: SceneCacheMeta = serde_json::from_slice(
            &fs::read(&meta_path)
                .with_context(|| format!("Cannot read scene meta {}", meta_path.display()))?,
        )?;
        if meta.version != 1
            || meta.source_name != source_name
            || meta.layout != "band_time_row_col"
            || meta.dtype != "f32_le"
        {
            continue;
        }
        if let Err(err) = validate_scene_meta(&meta, &dir) {
            eprintln!("Skipping scene cache {}: {err:#}", dir.display());
            continue;
        }
        let file =
            File::open(&bin_path).with_context(|| format!("Cannot open {}", bin_path.display()))?;
        let expected_bytes = meta
            .total_values
            .checked_mul(4)
            .context("scene cache size overflow")?;
        let actual_bytes = file.metadata()?.len() as usize;
        if actual_bytes != expected_bytes {
            bail!(
                "Scene cache {} has wrong size: expected {expected_bytes}, got {actual_bytes}",
                bin_path.display()
            );
        }
        let mmap = unsafe { Mmap::map(&file)? };
        out.push(SceneSource {
            cache_dir: dir,
            meta,
            mmap,
        });
    }
    Ok(out)
}

fn validate_scene_meta(meta: &SceneCacheMeta, dir: &Path) -> Result<()> {
    if meta.bands.is_empty() || meta.band_meta.is_empty() {
        bail!("Scene cache {} has no bands", dir.display());
    }
    if meta.bands.len() != meta.band_meta.len() {
        bail!(
            "Scene cache {} has bands/meta length mismatch: {} vs {}",
            dir.display(),
            meta.bands.len(),
            meta.band_meta.len()
        );
    }
    let first_timestamps = &meta.band_meta[0].timestamps;
    for (band_name, band_meta) in meta.bands.iter().zip(meta.band_meta.iter()) {
        if &band_meta.name != band_name {
            bail!(
                "Scene cache {} band order mismatch: {} vs {}",
                dir.display(),
                band_name,
                band_meta.name
            );
        }
        if band_meta.time_len != first_timestamps.len()
            || band_meta.timestamps.as_slice() != first_timestamps.as_slice()
        {
            bail!(
                "Scene cache {} has non-aligned timestamps for band {}",
                dir.display(),
                band_meta.name
            );
        }
    }
    Ok(())
}

fn collect_eligible_pixels(scene: &SceneSource, min_joint_valid: usize) -> Result<Vec<u32>> {
    let n_pixels = scene
        .meta
        .rows
        .checked_mul(scene.meta.cols)
        .context("pixel count overflow")?;
    if n_pixels > u32::MAX as usize {
        bail!("Scene has too many pixels for u32 sample index: {n_pixels}");
    }
    let n_times = scene.meta.band_meta[0].time_len;
    let mut counts = vec![0u16; n_pixels];
    let mut date_valid = vec![1u8; n_pixels];

    for t in 0..n_times {
        date_valid.fill(1);
        for band in &scene.meta.band_meta {
            let start = band_time_byte_offset(scene, band, t)?;
            let end = start
                .checked_add(n_pixels * 4)
                .context("time-slice byte range overflow")?;
            if end > scene.mmap.len() {
                bail!("Scene cache time slice extends beyond mmap");
            }
            let bytes = &scene.mmap[start..end];
            for pixel in 0..n_pixels {
                if date_valid[pixel] == 0 {
                    continue;
                }
                let byte_idx = pixel * 4;
                let value = f32::from_le_bytes([
                    bytes[byte_idx],
                    bytes[byte_idx + 1],
                    bytes[byte_idx + 2],
                    bytes[byte_idx + 3],
                ]);
                if !value.is_finite() {
                    date_valid[pixel] = 0;
                }
            }
        }
        for (count, &valid) in counts.iter_mut().zip(date_valid.iter()) {
            if valid != 0 {
                *count = count.saturating_add(1);
            }
        }
    }

    let mut out = Vec::new();
    for (pixel_offset, &count) in counts.iter().enumerate() {
        if count as usize >= min_joint_valid {
            out.push(pixel_offset as u32);
        }
    }
    Ok(out)
}

fn write_sample<W1: Write, W2: Write>(
    scene: &SceneSource,
    row: usize,
    col: usize,
    max_times: usize,
    sample_writer: &mut W1,
    mask_writer: &mut W2,
) -> Result<()> {
    let n_times = scene.meta.band_meta[0].time_len;
    let n_bands = scene.meta.band_meta.len();
    for t in 0..max_times {
        if t < n_times {
            let mut values = Vec::with_capacity(n_bands);
            let mut valid = true;
            for band in &scene.meta.band_meta {
                let value = read_scene_value(scene, band, t, row, col)?;
                if !value.is_finite() {
                    valid = false;
                }
                values.push(value);
            }
            mask_writer.write_all(&[u8::from(valid)])?;
            for value in values {
                sample_writer.write_all(&value.to_le_bytes())?;
            }
        } else {
            mask_writer.write_all(&[0])?;
            for _ in 0..n_bands {
                sample_writer.write_all(&f32::NAN.to_le_bytes())?;
            }
        }
    }
    Ok(())
}

fn read_scene_value(
    scene: &SceneSource,
    band: &SceneCacheBandMeta,
    t: usize,
    row: usize,
    col: usize,
) -> Result<f32> {
    let pixel_offset = row
        .checked_mul(scene.meta.cols)
        .and_then(|v| v.checked_add(col))
        .context("pixel offset overflow")?;
    let time_offset = t
        .checked_mul(scene.meta.rows)
        .and_then(|v| v.checked_mul(scene.meta.cols))
        .context("time offset overflow")?;
    let value_idx = band
        .offset_values
        .checked_add(time_offset)
        .and_then(|v| v.checked_add(pixel_offset))
        .context("value offset overflow")?;
    let byte_idx = value_idx.checked_mul(4).context("byte offset overflow")?;
    if byte_idx + 4 > scene.mmap.len() {
        bail!("Scene cache read beyond mmap");
    }
    Ok(f32::from_le_bytes([
        scene.mmap[byte_idx],
        scene.mmap[byte_idx + 1],
        scene.mmap[byte_idx + 2],
        scene.mmap[byte_idx + 3],
    ]))
}

fn band_time_byte_offset(
    scene: &SceneSource,
    band: &SceneCacheBandMeta,
    t: usize,
) -> Result<usize> {
    let time_offset = t
        .checked_mul(scene.meta.rows)
        .and_then(|v| v.checked_mul(scene.meta.cols))
        .context("time offset overflow")?;
    let value_idx = band
        .offset_values
        .checked_add(time_offset)
        .context("value offset overflow")?;
    value_idx.checked_mul(4).context("byte offset overflow")
}

fn write_u64<W: Write>(writer: &mut W, value: u64) -> Result<()> {
    writer.write_all(&value.to_le_bytes())?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn lcg_is_deterministic() {
        let mut a = Lcg64::new(42);
        let mut b = Lcg64::new(42);
        for _ in 0..16 {
            assert_eq!(a.next_u64(), b.next_u64());
        }
    }

    #[test]
    fn lcg_range_stays_inside_upper_bound() {
        let mut rng = Lcg64::new(7);
        for _ in 0..100 {
            assert!(rng.gen_range(3) < 3);
        }
    }
}
