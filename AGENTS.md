# NUFROST Agent Notes

## Project-Specific Notes

- This is a notebook-first research repo. Real user workflows start in `notebooks/`, not a CLI or packaged app.
- The public Python entrypoints are re-exported from `src/__init__.py`: `reconstruct_nufrost`, `reconstruct_zhu2015`, `reconstruct_hants`, `RSCube`, and `build_args`.
- Treat `NUFROST`, `Zhu2015`, and `HANTS` as three separate reconstruction algorithms. `Zhu2015` and `HANTS` are comparison baselines, not sub-parts of NUFROST.
- In prose and docs, write the algorithm name as `NUFROST`; keep code identifiers lowercase (`src/nufrost.py`, `reconstruct_nufrost`).

## Config And Data

- Runtime config is centered in `config/config.yaml` and `config/settings.py`.
- `build_args(overrides=...)` merges Python overrides on top of YAML defaults. CLI parsing uses YAML as parser defaults, so precedence is Python overrides > CLI flags > YAML.
- If `target_time` is omitted, `build_args()` falls back to `start_time`.
- `find_image_chunks()` returns a `List[str]`, usually a one-element list containing the final VRT path. Do not treat it as a single string.
- `find_image_chunks()` writes cached VRTs under `<cache_dir>/vrts/`.
- `RSCube._read_tif()` only reads the top-left `512x512` window of each source raster for memory safety.
- Default caches are `data/cache/local` for local runs and `data/cache/colab` for Colab-oriented notebooks.

## Output And Verification

- Output shapes differ by algorithm: `NUFROST` and `HANTS` write single-band GeoTIFFs; `Zhu2015` writes a 2-band GeoTIFF with prediction + QA.
- `environment.yml` is the full working environment for notebooks and Earth Engine tooling; `requirements.txt` is only the minimal algorithm/runtime set.
- There is no lint, formatter, typecheck, pre-commit, task runner, or CI config in the repo.
- The only checked-in test source is `tests/test_data_loader.py`. It hardcodes this machine's absolute `data/test_sample` path and ends with `breakpoint()`, so do not assume `pytest tests/` is a clean unattended smoke test on another machine.
- `.vscode/settings.json` is configured to run `pytest tests`.
- `temp_debug/` contains one-off experiments and validation scripts; do not treat it as stable workflow or authoritative test coverage.
