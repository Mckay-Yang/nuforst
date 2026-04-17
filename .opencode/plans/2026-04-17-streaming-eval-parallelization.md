# Streaming Eval Parallelization Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add joblib parallelization to the `_from_source` evaluation functions so that the local evaluation pipeline runs ~8x faster, and add incremental result saving so partial progress survives interruptions.

**Architecture:** The two `_from_source` functions (`evaluate_algorithms_from_source` and `evaluate_timeseries_from_source`) currently iterate sequentially over pixels, reading each pixel's time series from disk and running all three algorithms. We parallelize them using the same joblib `Parallel`/`delayed` pattern already used in their `_on_cube` counterparts. For incremental saving, we add a `callback` parameter that the notebook can use to append partial results to CSV after every batch of pixels.

**Tech Stack:** Python, joblib, numpy, pandas, tqdm

---

## File Structure

| File | Action | Responsibility |
|------|--------|---------------|
| `src/evaluation.py:309-361` | Modify | Parallelize `evaluate_algorithms_from_source` |
| `src/evaluation.py:364-413` | Modify | Parallelize `evaluate_timeseries_from_source` + add incremental save |
| `src/evaluation.py:739-755` | No change | `_process_random_point` already pure function, safe for parallel |
| `src/evaluation.py:634-706` | No change | `_process_pixel_ts` already pure function, safe for parallel |
| `tests/test_evaluation.py` | Modify | Add parallel-mode smoke tests |
| `notebooks/local_evals.ipynb` | Modify | Add incremental save callbacks |

---

## Task 1: Parallelize `evaluate_algorithms_from_source`

**Files:**
- Modify: `src/evaluation.py:309-361`
- Modify: `tests/test_evaluation.py`

The current implementation (lines 335-341) iterates sequentially over pixels, reading each pixel's time series and calling `_process_random_point`. The `_on_cube` counterpart (lines 440-471) already uses joblib `Parallel`/`delayed` with tqdm progress. We replicate that pattern here.

**Key challenge:** `source.read_pixel_series()` requires the rasterio dataset to be open. With joblib's `loky` backend, child processes cannot share the parent's file handle. Solution: pre-read all pixel time series into a dict before dispatching parallel jobs, same pattern as `_on_cube` which pre-loads the entire cube.

- [ ] **Step 1: Write failing test**

Add to `tests/test_evaluation.py`:

```python
def test_evaluate_algorithms_from_source_parallel(single_tile_path: str, cache_dir) -> None:
    args = build_args({"cache_dir": cache_dir, "force_refresh": False, "min_obs": 6, "n_jobs": 2})
    with open_evaluation_source([single_tile_path], args) as prepared:
        sampled_points = sample_random_points_from_source(
            prepared["source"], prepared["t_days"], min_obs=args.min_obs, num_points=20, seed=123
        )
        df = evaluate_algorithms_from_source(
            prepared["source"],
            prepared["t_sec"],
            prepared["t_days"],
            args,
            sampled_points=sampled_points,
            n_jobs=2,
        )
    assert set(df["Algorithm"]) == {"NuFrost", "Zhu2015", "HANTS"}
    assert np.isfinite(df["RMSE"]).all()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_evaluation.py::test_evaluate_algorithms_from_source_parallel -v`
Expected: PASS (current sequential code works for n_jobs=2 — the n_jobs parameter is currently ignored). This test will be used to verify the parallel version produces identical results.

- [ ] **Step 3: Implement parallelization**

Replace the body of `evaluate_algorithms_from_source` (lines 327-361) with:

```python
    print(f"Selected {len(sampled_points)} valid points for evaluation.")
    rc_to_t_idx = defaultdict(list)
    for t_idx, r, c in sampled_points:
        rc_to_t_idx[(int(r), int(c))].append(int(t_idx))

    start_time = time.time()
    n_jobs_resolved = _resolve_n_jobs(n_jobs)
    print(f"--> Running predictions with {n_jobs_resolved} parallel workers...")

    ordered_points = []
    tasks = []
    for (r, c), t_idxs in rc_to_t_idx.items():
        y_ts = source.read_pixel_series(r, c).copy()
        for t in t_idxs:
            y_ts[t] = np.nan
        for target_t_idx in t_idxs:
            tasks.append(delayed(_process_random_point)(target_t_idx, r, c, y_ts, t_sec, t_days, args))
            ordered_points.append((target_t_idx, r, c))

    point_results = []
    if JOBLIB_AVAILABLE and n_jobs_resolved > 1:
        if TQDM_AVAILABLE:
            try:
                gen = Parallel(n_jobs=n_jobs_resolved, return_as="generator")(tasks)
                point_results = list(tqdm(gen, total=len(tasks), desc="Evaluating Points"))
            except TypeError:
                point_results = Parallel(n_jobs=n_jobs_resolved)(tasks)
        else:
            point_results = Parallel(n_jobs=n_jobs_resolved)(tasks)
    else:
        iterator = tasks
        if TQDM_AVAILABLE:
            iterator = tqdm(iterator, total=len(tasks), desc="Evaluating Points")
        for task in iterator:
            func, func_args, func_kwargs = task
            point_results.append(func(*func_args, **func_kwargs))

    print(f"--> Prediction finished in {time.time() - start_time:.2f}s.")

    true_all = []
    pred_all_nufrost = []
    pred_all_zhu = []
    pred_all_hants = []
    for (t_idx, r, c), res in zip(ordered_points, point_results):
        true_all.append(source.read_pixel_series(r, c)[t_idx])
        pred_all_nufrost.append(res["nufrost"])
        pred_all_zhu.append(res["zhu"])
        pred_all_hants.append(res["hants"])

    metrics_nufrost = compute_metrics(np.array(true_all), np.array(pred_all_nufrost))
    metrics_zhu = compute_metrics(np.array(true_all), np.array(pred_all_zhu))
    metrics_hants = compute_metrics(np.array(true_all), np.array(pred_all_hants))
    return pd.DataFrame([
        {"Algorithm": "NuFrost", "RMSE": metrics_nufrost["RMSE"], "MAE": metrics_nufrost["MAE"], "R": metrics_nufrost["R"], "OutlierRatio": metrics_nufrost.get("OutlierRatio", np.nan)},
        {"Algorithm": "Zhu2015", "RMSE": metrics_zhu["RMSE"], "MAE": metrics_zhu["MAE"], "R": metrics_zhu["R"], "OutlierRatio": metrics_zhu.get("OutlierRatio", np.nan)},
        {"Algorithm": "HANTS", "RMSE": metrics_hants["RMSE"], "MAE": metrics_hants["MAE"], "R": metrics_hants["R"], "OutlierRatio": metrics_hants.get("OutlierRatio", np.nan)},
    ])
```

- [ ] **Step 4: Run all evaluation tests**

Run: `pytest tests/test_evaluation.py -v`
Expected: All 5 tests pass, including the new parallel test.

- [ ] **Step 5: Commit**

```bash
git add src/evaluation.py tests/test_evaluation.py
git commit -m "feat: parallelize evaluate_algorithms_from_source with joblib

- Add joblib Parallel/delayed pattern matching evaluate_algorithms_on_cube
- Pre-read pixel time series before dispatching to avoid shared file handles
- Add tqdm progress bar support
- Add n_jobs=2 smoke test
- Fall back to sequential when n_jobs=1 or joblib unavailable"
```

---

## Task 2: Parallelize `evaluate_timeseries_from_source` + Add Incremental Save

**Files:**
- Modify: `src/evaluation.py:364-413`
- Modify: `tests/test_evaluation.py`

The current implementation (lines 385-388) iterates sequentially over gap pixels. The `_on_cube` counterpart (lines 526-541) already uses joblib. We replicate that pattern and add an `on_batch_done` callback for incremental CSV saving.

**Key challenge:** Same as Task 1 — `source.read_pixel_series()` cannot be called from child processes. We pre-read all pixel time series into a list before dispatching.

**Incremental save design:** Add an optional `on_batch_done` callback parameter. After every `batch_size` pixels complete, the function calls `on_batch_done(batch_results)` where `batch_results` is a list of `_process_pixel_ts` return dicts. The notebook can use this to append partial results to CSV.

- [ ] **Step 1: Write failing test**

Add to `tests/test_evaluation.py`:

```python
def test_evaluate_timeseries_from_source_parallel(single_tile_path: str, cache_dir) -> None:
    args = build_args({"cache_dir": cache_dir, "force_refresh": False, "min_obs": 6, "n_jobs": 2})
    with open_evaluation_source([single_tile_path], args) as prepared:
        sampled_pixels = sample_gap_pixels_from_source(
            prepared["source"], prepared["t_days"], min_obs=args.min_obs, num_samples=10, seed=123
        )
        df = evaluate_timeseries_from_source(
            prepared["source"],
            prepared["t_sec"],
            prepared["t_days"],
            args,
            simulate_gap_days=30,
            sampled_pixels=sampled_pixels,
            n_jobs=2,
        )
    assert set(df["Algorithm"]) == {"NuFrost", "Zhu2015", "HANTS"}
    assert np.isfinite(df["MAE"]).all()
```

- [ ] **Step 2: Run test to verify current behavior**

Run: `pytest tests/test_evaluation.py::test_evaluate_timeseries_from_source_parallel -v`
Expected: PASS (current sequential code handles n_jobs=2 by ignoring it).

- [ ] **Step 3: Implement parallelization with incremental save**

Replace the body of `evaluate_timeseries_from_source` (lines 364-413) with:

```python
def evaluate_timeseries_from_source(
    source: TimeSeriesRasterSource,
    t_sec: np.ndarray,
    t_days: np.ndarray,
    args: Args,
    simulate_gap_days: int,
    num_samples: Optional[int] = None,
    sampled_pixels: Optional[np.ndarray] = None,
    n_jobs: int = -1,
    batch_size: int = 500,
    on_batch_done=None,
) -> pd.DataFrame:
    if sampled_pixels is None:
        if num_samples is None:
            raise ValueError("Either num_samples or sampled_pixels must be provided.")
        sampled_pixels = sample_gap_pixels_from_source(source, t_days, args.min_obs, num_samples)

    if len(sampled_pixels) == 0:
        print("Not enough valid pixels to evaluate.")
        return pd.DataFrame()

    print(f"Selected {len(sampled_pixels)} valid pixels for time-series evaluation.")
    n_jobs_resolved = _resolve_n_jobs(n_jobs)
    T = len(t_days)
    start_time = time.time()

    pixel_results = []
    if JOBLIB_AVAILABLE and n_jobs_resolved > 1:
        print(f"Running predictions with {n_jobs_resolved} parallel workers...")
        n_pixels = len(sampled_pixels)
        for batch_start in range(0, n_pixels, batch_size):
            batch_end = min(batch_start + batch_size, n_pixels)
            batch_pixels = sampled_pixels[batch_start:batch_end]
            tasks = [
                delayed(_process_pixel_ts)(
                    int(r), int(c),
                    source.read_pixel_series(int(r), int(c)).copy(),
                    t_days, t_sec, T, simulate_gap_days, args
                )
                for r, c in batch_pixels
            ]
            if TQDM_AVAILABLE:
                try:
                    gen = Parallel(n_jobs=n_jobs_resolved, return_as="generator")(tasks)
                    batch_results = list(tqdm(gen, total=len(tasks), desc=f"Pixels {batch_start}-{batch_end}"))
                except TypeError:
                    batch_results = Parallel(n_jobs=n_jobs_resolved)(tasks)
            else:
                batch_results = Parallel(n_jobs=n_jobs_resolved)(tasks)
            pixel_results.extend(batch_results)
            if on_batch_done is not None:
                on_batch_done(batch_results, batch_start, batch_end)
    else:
        iterator = sampled_pixels
        if TQDM_AVAILABLE:
            iterator = tqdm(iterator, total=len(sampled_pixels), desc="Evaluating Pixels")
        batch_results = []
        for i, (r, c) in enumerate(iterator):
            y_ts = source.read_pixel_series(int(r), int(c)).copy()
            res = _process_pixel_ts(int(r), int(c), y_ts, t_days, t_sec, T, simulate_gap_days, args)
            pixel_results.append(res)
            batch_results.append(res)
            if on_batch_done is not None and len(batch_results) >= batch_size:
                on_batch_done(batch_results, i + 1 - len(batch_results), i + 1)
                batch_results = []
        if on_batch_done is not None and batch_results:
            on_batch_done(batch_results, len(sampled_pixels) - len(batch_results), len(sampled_pixels))

    true_all_nufrost, pred_all_nufrost = [], []
    true_all_zhu, pred_all_zhu = [], []
    true_all_hants, pred_all_hants = [], []
    pixel_stats = []
    for res in pixel_results:
        if res is None:
            continue
        pixel_stats.append(res["stat"])
        true_all_nufrost.extend(res["true"])
        pred_all_nufrost.extend(res["pred_nufrost"])
        true_all_zhu.extend(res["true"])
        pred_all_zhu.extend(res["pred_zhu"])
        true_all_hants.extend(res["true"])
        pred_all_hants.extend(res["pred_hants"])
    print(f"--> Time-series evaluation finished in {time.time() - start_time:.2f}s.")
    if not pixel_stats:
        return pd.DataFrame()
    metrics_n = compute_metrics(np.array(true_all_nufrost), np.array(pred_all_nufrost))
    metrics_z = compute_metrics(np.array(true_all_zhu), np.array(pred_all_zhu))
    metrics_h = compute_metrics(np.array(true_all_hants), np.array(pred_all_hants))
    return pd.DataFrame([
        {"Algorithm": "NuFrost", "RMSE": metrics_n["RMSE"], "MAE": metrics_n["MAE"], "R": metrics_n["R"], "OutlierRatio": metrics_n["OutlierRatio"]},
        {"Algorithm": "Zhu2015", "RMSE": metrics_z["RMSE"], "MAE": metrics_z["MAE"], "R": metrics_z["R"], "OutlierRatio": metrics_z["OutlierRatio"]},
        {"Algorithm": "HANTS", "RMSE": metrics_h["RMSE"], "MAE": metrics_h["MAE"], "R": metrics_h["R"], "OutlierRatio": metrics_h["OutlierRatio"]},
    ])
```

- [ ] **Step 4: Run all evaluation tests**

Run: `pytest tests/test_evaluation.py -v`
Expected: All 6 tests pass (5 existing + 1 new).

- [ ] **Step 5: Commit**

```bash
git add src/evaluation.py tests/test_evaluation.py
git commit -m "feat: parallelize evaluate_timeseries_from_source with joblib and batched incremental save

- Add joblib Parallel/delayed pattern matching evaluate_timeseries_on_cube
- Process pixels in configurable batches (default 500) for incremental saving
- Add on_batch_done callback for notebook-level incremental CSV persistence
- Add n_jobs=2 smoke test
- Fall back to sequential with batched callbacks when n_jobs=1"
```

---

## Task 3: Wire Incremental Save into Notebook

**Files:**
- Modify: `notebooks/local_evals.ipynb`

The notebook currently calls `evaluate_timeseries_from_source` and `evaluate_algorithms_from_source` without incremental saving — results are only persisted when the full function returns and `append_rows` is called. We add an `on_batch_done` callback for the gap evaluation steps that writes partial per-pixel metrics to a secondary CSV, so progress survives interruption.

**Design:** For `evaluate_timeseries_from_source` calls, we add an `on_batch_done` callback that computes per-batch metrics and appends them to a "checkpoint" CSV. On restart, the `load_done_keys` mechanism already skips completed (Image, Scenario, Variant) tuples, so we only need the checkpoint CSV as a safety net for the currently-running step. The final `append_rows` call still writes the authoritative results.

For simplicity, we log batch progress to a separate checkpoint file (e.g., `data/output/hls_ablation_checkpoint.csv`) and clean it up after the authoritative `append_rows` succeeds.

- [ ] **Step 1: Add checkpoint helper functions to the notebook**

In the notebook's helper cell (the same cell that defines `append_rows`, `load_done_keys`, etc.), add:

```python
def _make_batch_checkpoint_writer(csv_path, loc_id, scenario, variant, gap_length=None, gap_index_target=None, loc_meta=None):
    checkpoint_path = csv_path.parent / f"{csv_path.stem}_checkpoint.csv"
    def on_batch_done(batch_results, batch_start, batch_end):
        n_valid = sum(1 for r in batch_results if r is not None)
        if n_valid == 0:
            return
        print(f"  [checkpoint] {scenario}/{variant} pixels {batch_start}-{batch_end}: {n_valid}/{len(batch_results)} valid", flush=True)
        row = {
            "Image": loc_id,
            "Scenario": scenario,
            "Variant": variant,
            "BatchStart": batch_start,
            "BatchEnd": batch_end,
            "ValidPixels": n_valid,
            "Timestamp": pd.Timestamp.now().isoformat(),
        }
        if gap_length is not None:
            row["GapLength"] = gap_length
        if gap_index_target is not None:
            row["GapIndexTarget"] = gap_index_target
        if loc_meta:
            row.update(loc_meta)
        append_rows(checkpoint_path, pd.DataFrame([row]))
    return on_batch_done, checkpoint_path
```

- [ ] **Step 2: Wire callback into ablation gap calls**

In the main evaluation loop, for each `evaluate_timeseries_from_source` call, create the callback and pass it:

Replace:
```python
df_gap = src.evaluation.evaluate_timeseries_from_source(source, t_sec, t_days, variant_args, simulate_gap_days=ablation_gap_days, sampled_pixels=gap_pixels_full, n_jobs=N_JOBS)
```

With:
```python
on_batch, ckpt_path = _make_batch_checkpoint_writer(OUTPUT_PATHS["ablation"], loc_id, "gap", variant_name, gap_length=ablation_gap_days, gap_index_target=ABLATION_GAP_INDEX, loc_meta=loc_meta)
df_gap = src.evaluation.evaluate_timeseries_from_source(source, t_sec, t_days, variant_args, simulate_gap_days=ablation_gap_days, sampled_pixels=gap_pixels_full, n_jobs=N_JOBS, on_batch_done=on_batch)
```

Apply the same pattern to:
- Baseline gap call (line ~1868): variant="Baselines"
- Gap sweep call (line ~1902): variant=f"GapSweep_{gap_days}d"
- Repeatability gap call (line ~1935): variant=f"Repeat_seed{repeat_seed}"

After each `append_rows(OUTPUT_PATHS["ablation"], ...)` call succeeds, add:
```python
if ckpt_path.exists():
    ckpt_path.unlink()
```

- [ ] **Step 3: Commit**

```bash
git add notebooks/local_evals.ipynb
git commit -m "feat: add incremental checkpoint saving to local_evals notebook

- Add _make_batch_checkpoint_writer helper that logs batch progress
- Wire on_batch_done callback into all evaluate_timeseries_from_source calls
- Checkpoint CSV is cleaned up after authoritative results are saved
- Prevents data loss on interruption of long-running gap evaluations"
```

---

## Self-Review Checklist

**1. Spec coverage:**
- Parallelize `evaluate_algorithms_from_source` → Task 1
- Parallelize `evaluate_timeseries_from_source` → Task 2
- Incremental save for gap evaluation → Task 2 (callback) + Task 3 (notebook wiring)
- Notebook uses new features → Task 3

**2. Placeholder scan:**
- No TBD/TODO/handwave steps found.
- All code blocks contain complete implementation code.
- Line numbers are approximate and must be verified during implementation.

**3. Type consistency:**
- `on_batch_done` callback signature: `(batch_results: list, batch_start: int, batch_end: int) -> None`
- `batch_size: int = 500` default matches the callback batching logic
- `n_jobs` parameter already exists in both function signatures; behavior changes from ignored to used
- `_process_random_point` and `_process_pixel_ts` are pure functions safe for parallel dispatch
- `source.read_pixel_series()` is called in the parent process before dispatching tasks (pre-read pattern)

**4. Risk assessment:**
- **Memory:** Pre-reading all pixel time series for `evaluate_algorithms_from_source` stores them in `tasks` list via closure. For 20,000 points with ~400 time steps each, this is ~64 MB — acceptable.
- **Batch pre-read for timeseries:** In Task 2, we read pixel series inside the batch loop (before `Parallel`), so memory is bounded by `batch_size` pixels × ~400 time steps = ~1.6 MB per batch — excellent.
- **Shared state:** `source` object is only used in the parent process for reading pixel series; child processes receive numpy arrays — no file handle sharing issues.
- **Backward compatibility:** `batch_size` and `on_batch_done` are optional parameters with defaults; existing callers are unaffected.
