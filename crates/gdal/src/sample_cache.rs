use std::fs::{self, File};
use std::io::{BufWriter, Write};
use std::path::{Path, PathBuf};
use std::time::Instant;

use anyhow::{bail, Context, Result};
use memmap2::Mmap;
use rayon::prelude::*;
use serde::{Deserialize, Serialize};

use crate::parse_iso8601_to_epoch_seconds;

const SAMPLE_CACHE_VERSION: u32 = 1;
const SAMPLE_CACHE_DTYPE: &str = "f32_le";
const SAMPLE_CACHE_LAYOUT: &str = "sample_time_band";
const SAMPLE_CACHE_MASK_LAYOUT: &str = "sample_time";
const SAMPLE_CACHE_INDEX_LAYOUT: &str = "sample_scene_row_col_n_times";
const SAMPLE_CACHE_WRITE_BATCH_SIZE: usize = 32_768;

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

#[derive(Debug, Clone, Copy)]
struct SampleTask {
    scene_id: usize,
    row: usize,
    col: usize,
    pixel_offset: usize,
    n_times: usize,
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
    let build_started = Instant::now();
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

    eprintln!(
        "sample cache build: source={}, scene_root={}, scenes={}, output={}, n_samples={}, min_joint_valid={}, seed={}",
        options.source_name,
        source_root.display(),
        scenes.len(),
        options.output_dir.display(),
        options.n_samples,
        options.min_joint_valid,
        options.seed
    );

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

    let eligible_started = Instant::now();
    eprintln!(
        "sample cache eligible scan: using rayon scene-level parallelism, current_threads={}",
        rayon::current_num_threads()
    );
    let eligible_pixels: Vec<Vec<u32>> = scenes
        .par_iter()
        .enumerate()
        .map(|(scene_idx, scene)| -> Result<Vec<u32>> {
            let scene_started = Instant::now();
            let n_pixels = scene
                .meta
                .rows
                .checked_mul(scene.meta.cols)
                .context("pixel count overflow")?;
            eprintln!(
                "sample cache eligible scan {}/{}: lon={:.6}, lat={:.6}, size={}x{}, times={}, pixels={}",
                scene_idx + 1,
                scenes.len(),
                scene.meta.lon,
                scene.meta.lat,
                scene.meta.rows,
                scene.meta.cols,
                scene.meta.band_meta[0].time_len,
                n_pixels
            );
            let pixels = collect_eligible_pixels(scene, options.min_joint_valid)?;
            let eligible_ratio = if n_pixels == 0 {
                0.0
            } else {
                pixels.len() as f64 * 100.0 / n_pixels as f64
            };
            eprintln!(
                "sample cache eligible scan {}/{} done: lon={:.6}, lat={:.6}, eligible={} ({eligible_ratio:.2}%), threshold={}, elapsed={}",
                scene_idx + 1,
                scenes.len(),
                scene.meta.lon,
                scene.meta.lat,
                pixels.len(),
                options.min_joint_valid,
                format_duration(scene_started.elapsed().as_secs_f64())
            );
            Ok(pixels)
        })
        .collect::<Result<Vec<_>>>()?;
    let total_eligible = eligible_pixels.iter().try_fold(0usize, |acc, pixels| {
        acc.checked_add(pixels.len())
            .context("eligible pixel count overflow")
    })?;
    eprintln!(
        "sample cache eligible scan complete: total_eligible={}, elapsed={}",
        total_eligible,
        format_duration(eligible_started.elapsed().as_secs_f64())
    );
    if total_eligible == 0 {
        bail!(
            "No eligible pixels found with --min-joint-valid {}",
            options.min_joint_valid
        );
    }

    let parent_dir = options
        .output_dir
        .parent()
        .unwrap_or_else(|| Path::new("."));
    fs::create_dir_all(parent_dir)
        .with_context(|| format!("Cannot create {}", parent_dir.display()))?;
    let output_name = options
        .output_dir
        .file_name()
        .and_then(|name| name.to_str())
        .unwrap_or("sample-cache");
    let build_dir = parent_dir.join(format!(".{output_name}.tmp-{}", std::process::id()));
    if build_dir.exists() {
        fs::remove_dir_all(&build_dir)
            .with_context(|| format!("Cannot remove stale temp dir {}", build_dir.display()))?;
    }
    fs::create_dir_all(&build_dir)
        .with_context(|| format!("Cannot create {}", build_dir.display()))?;
    eprintln!(
        "sample cache output: temp={}, final={}",
        build_dir.display(),
        options.output_dir.display()
    );

    let sample_path = build_dir.join("samples.f32.bin");
    let mask_path = build_dir.join("mask.u8.bin");
    let time_path = build_dir.join("scene_times.f64.bin");
    let index_path = build_dir.join("index.u64.bin");
    let meta_path = build_dir.join("meta.json");

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
    eprintln!(
        "sample cache time axis: wrote {} timestamps across {} scenes",
        time_offset,
        scenes.len()
    );

    let mut rng = Lcg64::new(options.seed);
    let mut accepted = 0usize;
    let sample_bytes_per_record = max_times
        .checked_mul(n_bands)
        .and_then(|v| v.checked_mul(4))
        .context("sample record byte size overflow")?;
    let mask_bytes_per_record = max_times;
    let estimated_sample_bytes = sample_bytes_per_record as u64 * options.n_samples as u64;
    let estimated_mask_bytes = mask_bytes_per_record as u64 * options.n_samples as u64;
    let estimated_index_bytes = 4u64 * 8 * options.n_samples as u64;
    eprintln!(
        "sample cache layout: {} samples x {} times x {} bands; estimated samples={}, mask={}, index={}",
        options.n_samples,
        max_times,
        n_bands,
        format_bytes(estimated_sample_bytes),
        format_bytes(estimated_mask_bytes),
        format_bytes(estimated_index_bytes)
    );

    let task_started = Instant::now();
    eprintln!(
        "sample cache sampling: generating {} deterministic sample tasks",
        options.n_samples
    );
    let mut all_tasks = Vec::with_capacity(options.n_samples);
    for _ in 0..options.n_samples {
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
        all_tasks.push(SampleTask {
            scene_id,
            row,
            col,
            pixel_offset,
            n_times: scene.meta.band_meta[0].time_len,
        });
    }
    eprintln!(
        "sample cache sampling: generated {} tasks in {}; sorting by scene/pixel",
        all_tasks.len(),
        format_duration(task_started.elapsed().as_secs_f64())
    );
    all_tasks.sort_unstable_by_key(|task| (task.scene_id, task.pixel_offset));
    eprintln!(
        "sample cache sampling: sorted tasks in {}",
        format_duration(task_started.elapsed().as_secs_f64())
    );

    let mut sample_block = vec![0u8; SAMPLE_CACHE_WRITE_BATCH_SIZE * sample_bytes_per_record];
    let mut mask_block = vec![0u8; SAMPLE_CACHE_WRITE_BATCH_SIZE * mask_bytes_per_record];
    let mut index_block = vec![0u8; SAMPLE_CACHE_WRITE_BATCH_SIZE * 4 * 8];
    let write_started = Instant::now();

    for tasks in all_tasks.chunks(SAMPLE_CACHE_WRITE_BATCH_SIZE) {
        let batch_started = Instant::now();
        let batch_len = tasks.len();

        let sample_len = batch_len
            .checked_mul(sample_bytes_per_record)
            .context("sample block byte size overflow")?;
        let mask_len = batch_len
            .checked_mul(mask_bytes_per_record)
            .context("mask block byte size overflow")?;
        let index_len = batch_len
            .checked_mul(4 * 8)
            .context("index block byte size overflow")?;
        let nan = f32::NAN.to_le_bytes();
        sample_block[..sample_len]
            .par_chunks_mut(4)
            .for_each(|chunk| chunk.copy_from_slice(&nan));
        mask_block[..mask_len].fill(0);

        let mut group_start = 0usize;
        while group_start < tasks.len() {
            let scene_id = tasks[group_start].scene_id;
            let mut group_end = group_start + 1;
            while group_end < tasks.len() && tasks[group_end].scene_id == scene_id {
                group_end += 1;
            }
            fill_scene_sample_group(
                &scenes[scene_id],
                &tasks[group_start..group_end],
                group_start,
                max_times,
                n_bands,
                sample_bytes_per_record,
                mask_bytes_per_record,
                &mut sample_block[..sample_len],
                &mut mask_block[..mask_len],
            )?;
            group_start = group_end;
        }

        for (task_idx, task) in tasks.iter().enumerate() {
            let start = task_idx * 4 * 8;
            put_u64_le(&mut index_block[start..start + 8], task.scene_id as u64);
            put_u64_le(&mut index_block[start + 8..start + 16], task.row as u64);
            put_u64_le(&mut index_block[start + 16..start + 24], task.col as u64);
            put_u64_le(
                &mut index_block[start + 24..start + 32],
                task.n_times as u64,
            );
            scene_meta[task.scene_id].n_sampled += 1;
        }

        sample_writer.write_all(&sample_block[..sample_len])?;
        mask_writer.write_all(&mask_block[..mask_len])?;
        index_writer.write_all(&index_block[..index_len])?;
        accepted += batch_len;

        if accepted % SAMPLE_CACHE_WRITE_BATCH_SIZE == 0 || accepted == options.n_samples {
            sample_writer.flush()?;
            mask_writer.flush()?;
            index_writer.flush()?;
            let elapsed = write_started.elapsed().as_secs_f64();
            let batch_elapsed = batch_started.elapsed().as_secs_f64();
            let rate = accepted as f64 / elapsed.max(1.0e-9);
            let remaining = (options.n_samples - accepted) as f64 / rate.max(1.0e-9);
            let pct = accepted as f64 * 100.0 / options.n_samples as f64;
            let sample_bytes = sample_bytes_per_record as u64 * accepted as u64;
            let mask_bytes = mask_bytes_per_record as u64 * accepted as u64;
            let index_bytes = 4u64 * 8 * accepted as u64;
            eprintln!(
                "sample cache write progress: {accepted}/{} ({pct:.2}%), batch={}, batch_elapsed={}, rate={rate:.1} samples/s, elapsed={}, eta={}, samples={}, mask={}, index={}, eligible_pool={total_eligible}",
                options.n_samples,
                batch_len,
                format_duration(batch_elapsed),
                format_duration(elapsed),
                format_duration(remaining),
                format_bytes(sample_bytes),
                format_bytes(mask_bytes),
                format_bytes(index_bytes)
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
    eprintln!("sample cache metadata written: {}", meta_path.display());

    if options.output_dir.exists() {
        eprintln!(
            "sample cache replace: removing existing {}",
            options.output_dir.display()
        );
        fs::remove_dir_all(&options.output_dir)
            .with_context(|| format!("Cannot replace {}", options.output_dir.display()))?;
    }
    eprintln!(
        "sample cache replace: moving {} -> {}",
        build_dir.display(),
        options.output_dir.display()
    );
    fs::rename(&build_dir, &options.output_dir).with_context(|| {
        format!(
            "Cannot move completed sample cache {} to {}",
            build_dir.display(),
            options.output_dir.display()
        )
    })?;
    eprintln!(
        "sample cache build complete: output={}, elapsed={}",
        options.output_dir.display(),
        format_duration(build_started.elapsed().as_secs_f64())
    );

    Ok(meta)
}

fn format_bytes(bytes: u64) -> String {
    const KIB: f64 = 1024.0;
    const MIB: f64 = KIB * 1024.0;
    const GIB: f64 = MIB * 1024.0;
    let bytes_f = bytes as f64;
    if bytes_f >= GIB {
        format!("{:.2} GiB", bytes_f / GIB)
    } else if bytes_f >= MIB {
        format!("{:.2} MiB", bytes_f / MIB)
    } else if bytes_f >= KIB {
        format!("{:.2} KiB", bytes_f / KIB)
    } else {
        format!("{bytes} B")
    }
}

fn format_duration(seconds: f64) -> String {
    if !seconds.is_finite() || seconds < 0.0 {
        return "unknown".to_string();
    }
    if seconds >= 3600.0 {
        format!("{:.2} h", seconds / 3600.0)
    } else if seconds >= 60.0 {
        format!("{:.1} min", seconds / 60.0)
    } else {
        format!("{seconds:.1} s")
    }
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

struct BlockLayout {
    group_start: usize,
    n_bands: usize,
    sample_bytes_per_record: usize,
}

fn fill_scene_sample_group(
    scene: &SceneSource,
    tasks: &[SampleTask],
    group_start: usize,
    max_times: usize,
    n_bands: usize,
    sample_bytes_per_record: usize,
    mask_bytes_per_record: usize,
    sample_block: &mut [u8],
    mask_block: &mut [u8],
) -> Result<()> {
    let layout = BlockLayout {
        group_start,
        n_bands,
        sample_bytes_per_record,
    };
    let n_times = scene.meta.band_meta[0].time_len;
    let n_pixels = scene
        .meta
        .rows
        .checked_mul(scene.meta.cols)
        .context("pixel count overflow")?;
    if scene.meta.band_meta.len() != n_bands {
        bail!("sample cache scene band count mismatch");
    }
    let expected_sample_len = tasks
        .len()
        .checked_add(group_start)
        .and_then(|v| v.checked_mul(sample_bytes_per_record))
        .context("sample block size overflow")?;
    let expected_mask_len = tasks
        .len()
        .checked_add(group_start)
        .and_then(|v| v.checked_mul(mask_bytes_per_record))
        .context("mask block size overflow")?;
    let expected_sample_record_len = max_times
        .checked_mul(n_bands)
        .and_then(|v| v.checked_mul(4))
        .context("sample buffer size overflow")?;
    if expected_sample_record_len != sample_bytes_per_record
        || max_times != mask_bytes_per_record
        || sample_block.len() < expected_sample_len
        || mask_block.len() < expected_mask_len
    {
        bail!("sample cache write block has unexpected lengths");
    }

    let mut valid = vec![1u8; tasks.len()];
    for t in 0..n_times {
        valid.fill(1);
        for (band_idx, band) in scene.meta.band_meta.iter().enumerate() {
            fill_scene_sample_group_band(
                scene,
                band,
                tasks,
                &layout,
                n_pixels,
                t,
                band_idx,
                &mut valid,
                sample_block,
            )?;
        }
        for (local_idx, &is_valid) in valid.iter().enumerate() {
            let sample_idx = group_start + local_idx;
            let mask_idx = sample_idx
                .checked_mul(mask_bytes_per_record)
                .and_then(|v| v.checked_add(t))
                .context("mask byte offset overflow")?;
            mask_block[mask_idx] = is_valid;
        }
    }
    Ok(())
}

fn fill_scene_sample_group_band(
    scene: &SceneSource,
    band: &SceneCacheBandMeta,
    tasks: &[SampleTask],
    layout: &BlockLayout,
    n_pixels: usize,
    t: usize,
    band_idx: usize,
    valid: &mut [u8],
    sample_block: &mut [u8],
) -> Result<()> {
    let time_offset = t
        .checked_mul(n_pixels)
        .context("time pixel offset overflow")?;
    let value_start = band
        .offset_values
        .checked_add(time_offset)
        .context("value offset overflow")?;
    let byte_start = value_start.checked_mul(4).context("byte offset overflow")?;
    let byte_len = n_pixels
        .checked_mul(4)
        .context("time-slice byte size overflow")?;
    let byte_end = byte_start
        .checked_add(byte_len)
        .context("time-slice byte range overflow")?;
    if byte_end > scene.mmap.len() {
        bail!("Scene cache sample time slice extends beyond mmap");
    }
    let time_slice = &scene.mmap[byte_start..byte_end];
    for (local_idx, task) in tasks.iter().enumerate() {
        let src = task
            .pixel_offset
            .checked_mul(4)
            .context("pixel byte offset overflow")?;
        let bytes = &time_slice[src..src + 4];
        let value = f32::from_le_bytes([bytes[0], bytes[1], bytes[2], bytes[3]]);
        if !value.is_finite() {
            valid[local_idx] = 0;
        }
        let sample_idx = layout.group_start + local_idx;
        let dst = sample_idx
            .checked_mul(layout.sample_bytes_per_record)
            .and_then(|v| {
                v.checked_add(
                    t.checked_mul(layout.n_bands)
                        .and_then(|x| x.checked_add(band_idx))
                        .and_then(|x| x.checked_mul(4))?,
                )
            })
            .context("sample byte offset overflow")?;
        sample_block[dst..dst + 4].copy_from_slice(bytes);
    }
    Ok(())
}

fn put_u64_le(dst: &mut [u8], value: u64) {
    dst.copy_from_slice(&value.to_le_bytes());
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
