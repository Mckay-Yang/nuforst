# Streaming Evaluation Refactor Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Move the local evaluation workflow off the full-cube NPZ path and into loader-owned streaming reads that still evaluate complete images.

**Architecture:** Keep `RSCube` as the explicit full-cube plus NPZ loader, but add a separate streaming reader in `src/data_loader.py` for metadata, window reads, and per-pixel time-series reads from VRT/TIFF. Refactor `src/evaluation.py` and `notebooks/local_evals.ipynb` to use the streaming reader so notebooks no longer own NPZ avoidance logic.

**Tech Stack:** Python, rasterio/GDAL, NumPy, pandas, pytest, Jupyter notebooks

---

### Task 1: Add loader-owned streaming read interfaces

**Files:**
- Modify: `src/data_loader.py`
- Test: `tests/test_data_loader.py`

- [ ] Add a streaming dataset wrapper that opens VRT/TIFF directly without writing NPZ.
- [ ] Expose metadata access, spatial window iteration, window reads, and pixel-series reads.
- [ ] Keep `RSCube.load()` unchanged as the explicit full-cube NPZ path.
- [ ] Add tests proving streaming reads work and do not create `npz/` artifacts.

### Task 2: Refactor evaluation onto streaming APIs

**Files:**
- Modify: `src/evaluation.py`
- Test: `tests/test_evaluation.py`

- [ ] Add streaming-based source open helpers that return timestamps and time axes without loading full cubes.
- [ ] Add full-image candidate scanning via loader-owned window iteration.
- [ ] Add random-point and gap-pixel sampling on top of streaming scans.
- [ ] Add streaming evaluation entrypoints for random-point and time-series gap experiments.
- [ ] Preserve the existing full-cube functions for backward compatibility where practical.

### Task 3: Switch maintained local notebook to streaming evaluation

**Files:**
- Modify: `notebooks/local_evals.ipynb`

- [ ] Replace `load_evaluation_cube()` usage with streaming source open helpers.
- [ ] Replace cube-based sampling/evaluation calls with streaming evaluation calls.
- [ ] Keep evaluation coverage the same: ablation, sparse sweep, gap sweep, repeatability.
- [ ] Ensure the notebook no longer triggers `NPZ cache miss` for evaluation.

### Task 4: Connect parameter-cache work to loader-owned reads

**Files:**
- Modify: `src/model_params.py`
- Test: `tests/test_model_params.py`

- [ ] Add `from_loader` parameter-fit entrypoints so parameter caching no longer requires a preloaded full cube.
- [ ] Keep the three algorithms supported: `NUFROST`, `HANTS`, and `Zhu2015` with `max_segments=10`.
- [ ] Preserve full-image semantics through blockwise iteration.

### Task 5: Verify no-NPZ local evaluation path

**Files:**
- Modify: `tests/test_evaluation.py`
- Modify: `tests/test_data_loader.py`
- Modify: `tests/test_model_params.py`

- [ ] Add regression tests proving streaming evaluation avoids NPZ creation.
- [ ] Run focused tests for loader, evaluation, and parameter caching.
- [ ] Run the full test suite.
- [ ] Manually verify that `notebooks/local_evals.ipynb` now uses the streaming path by checking the call sites and expected logs.
