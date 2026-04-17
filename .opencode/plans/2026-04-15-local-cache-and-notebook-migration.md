# Local Cache And Notebook Migration Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make the unified evaluation workflow runnable from the local checkout by standardizing cache layout under `data/cache/`, fixing local VRT loading, and adding maintained local notebook entrypoints.

**Architecture:** Centralize all cache layout decisions in `src/data_loader.py`, keep configuration defaults project-root-relative, and move the maintained notebook workflow to local entrypoints that call the shared evaluation code. Verify the fix with one real chunk before trusting notebook execution.

**Tech Stack:** Python, rasterio/GDAL, NumPy, pandas, Jupyter notebooks, pytest

---

### Task 1: Audit current cache and path assumptions

**Files:**
- Modify: `docs/superpowers/specs/2026-04-15-local-cache-and-notebook-design.md`
- Modify: `docs/superpowers/plans/2026-04-15-local-cache-and-notebook-migration.md`

- [ ] **Step 1: Confirm path-sensitive files to touch**

Run: `rg -n "/Users/mckay|/content/drive|MyDrive|data/local_cache|data/colab_cache" /Volumes/T7/nuforst`
Expected: matches in notebooks and shared path/cache code.

- [ ] **Step 2: Confirm the current local VRT failure on one real chunk**

Run:

```bash
python - <<'PY'
from config import build_args
from src.data_loader import find_image_chunks
from src.evaluation import load_evaluation_cube

data_dir = 'data/hls'
cache_dir = 'data/cache/local'
chunks = find_image_chunks(data_dir, 91.2734, 29.7904, 'BLUE', cache_dir=cache_dir)
print(chunks)
args = build_args({})
args.cache_dir = cache_dir
args.force_refresh = True
load_evaluation_cube(chunks, args)
PY
```

Expected: current branch reproduces the VRT-backed load failure before the fix.

### Task 2: Add failing tests for the new cache layout and VRT loading

**Files:**
- Modify: `tests/test_data_loader.py`

- [ ] **Step 1: Add a failing cache layout test**

```python
def test_find_image_chunks_writes_vrt_under_cache_vrts(fixtures_input_dir, tmp_path):
    from src.data_loader import find_image_chunks

    cache_root = tmp_path / "cache" / "local"
    chunks = find_image_chunks(
        fixtures_input_dir.as_posix(),
        100.112,
        25.654,
        "B2",
        cache_dir=cache_root,
    )

    assert len(chunks) == 1
    assert "/vrts/" in chunks[0].replace("\\", "/")
```

- [ ] **Step 2: Add a failing NPZ placement test**

```python
def test_rscube_writes_npz_under_cache_npz(fixtures_input_dir, tmp_path):
    from src.data_loader import RSCube

    tif_path = fixtures_input_dir / "COPERNICUS_S2_HARMONIZED_B2_lon100.112_lat25.654-0000001024-0000000000.tif"
    cache_root = tmp_path / "cache" / "local"
    loader = RSCube(tif_path.as_posix(), cache_dir=cache_root, force_refresh=True)
    data = loader.load()

    cache_path = data["cache_path"]
    assert "/npz/" in cache_path.replace("\\", "/")
```

- [ ] **Step 3: Add a failing VRT readability regression test**

```python
def test_rscube_can_load_cube_from_generated_vrt(fixtures_input_dir, tmp_path):
    from src.data_loader import RSCube, find_image_chunks

    cache_root = tmp_path / "cache" / "local"
    chunks = find_image_chunks(
        fixtures_input_dir.as_posix(),
        100.112,
        25.654,
        "B2",
        cache_dir=cache_root,
    )

    loader = RSCube(chunks, cache_dir=cache_root, force_refresh=True)
    data = loader.load()

    assert data["cube"].ndim == 3
    assert data["cube"].shape[1] > 0
    assert data["cube"].shape[2] > 0
```

- [ ] **Step 4: Run the targeted tests and confirm failure**

Run: `pytest tests/test_data_loader.py -q`
Expected: one or more failures showing current cache placement or VRT loading is incorrect.

### Task 3: Centralize cache directory resolution in the data loader

**Files:**
- Modify: `src/data_loader.py`
- Test: `tests/test_data_loader.py`

- [ ] **Step 1: Add minimal cache directory helpers**

```python
def _resolve_cache_root(cache_dir: Optional[Union[str, Path]]) -> Path:
    if cache_dir is None:
        return Path(__file__).resolve().parent.parent / "data" / "cache" / "local"
    return Path(cache_dir)


def _cache_subdir(cache_root: Path, kind: str) -> Path:
    path = cache_root / kind
    path.mkdir(parents=True, exist_ok=True)
    return path
```

- [ ] **Step 2: Update `RSCube.__init__` to store a cache root instead of a flat cache directory**

```python
self.cache_dir = _resolve_cache_root(cache_dir)
self.cache_dir.mkdir(parents=True, exist_ok=True)
```

- [ ] **Step 3: Route NPZ cache files into `npz/`**

```python
def _cache_path(self) -> Path:
    npz_dir = _cache_subdir(self.cache_dir, "npz")
    base_stem = self.tif_paths[0].stem
    if len(base_stem.split('_')) > 1 and (base_stem.split('_')[-1].startswith('part') or (base_stem.split('_')[-1].isdigit() and len(base_stem.split('_')[-1]) == 4)):
        base_stem = '_'.join(base_stem.split('_')[:-1])
    return npz_dir / f"{base_stem}_{self._file_signature()}.npz"
```

- [ ] **Step 4: Route VRT cache files into `vrts/`**

```python
cache_root = _resolve_cache_root(cache_dir) if cache_dir is not None else Path(data_dir).parent / "cache" / "local"
vrt_dir = _cache_subdir(cache_root, "vrts")
```

- [ ] **Step 5: Run targeted tests and confirm only VRT path issues remain if anything fails**

Run: `pytest tests/test_data_loader.py -q`
Expected: cache placement assertions pass; remaining failure, if any, points at VRT source resolution.

### Task 4: Fix VRT source path generation for local loads

**Files:**
- Modify: `src/data_loader.py`
- Test: `tests/test_data_loader.py`

- [ ] **Step 1: Replace brittle VRT source path generation with a GDAL-safe absolute path**

```python
src_filename = ET.SubElement(simple_source, "SourceFilename", relativeToVRT="0")
src_filename.text = str(tile_info["path"].resolve())
```

- [ ] **Step 2: Keep tile path objects as `Path` instances through VRT generation**

```python
tile_infos.append({
    "path": Path(tile_path),
    "width": src.width,
    "height": src.height,
    "block_w": block_w,
    "block_h": block_h,
    "x_origin": float(transform.c),
    "y_origin": float(transform.f),
})
```

- [ ] **Step 3: Re-run the targeted data-loader tests**

Run: `pytest tests/test_data_loader.py -q`
Expected: all targeted tests pass, including VRT-backed cube loading.

### Task 5: Update configuration defaults to the new local cache root

**Files:**
- Modify: `config/config.yaml`
- Modify: `config/settings.py`
- Modify: `tests/test_config.py`

- [ ] **Step 1: Locate the current cache default in config**

Run: `rg -n "cache_dir|local_cache|colab_cache" config tests/test_config.py`
Expected: current cache default references are found in config and tests.

- [ ] **Step 2: Change the default local cache root to `data/cache/local`**

```yaml
cache_dir: data/cache/local
```

- [ ] **Step 3: Update any settings-layer normalization logic if present**

```python
cache_dir = Path(raw_cache_dir) if raw_cache_dir is not None else project_root / "data" / "cache" / "local"
```

- [ ] **Step 4: Update config tests for the new default**

```python
def test_build_args_uses_local_cache_default():
    args = build_args({})
    assert args.cache_dir == "data/cache/local"
```

- [ ] **Step 5: Run config tests**

Run: `pytest tests/test_config.py -q`
Expected: PASS.

### Task 6: Create maintained local notebook entrypoints

**Files:**
- Create: `notebooks/local_evals.ipynb`
- Create: `notebooks/local_result_summary.ipynb`
- Modify: `notebooks/colab_evals.ipynb`
- Modify: `notebooks/colab_result_summary.ipynb`

- [ ] **Step 1: Copy the maintained evaluation flow into a local notebook**

Create `notebooks/local_evals.ipynb` from the unified evaluation notebook, with path cells equivalent to:

```python
from pathlib import Path

PROJECT_DIR = Path.cwd()
IMAGE_DIR = PROJECT_DIR / "data" / "hls"
OUTPUT_DIR = PROJECT_DIR / "data" / "output"
CACHE_DIR = PROJECT_DIR / "data" / "cache" / "local"
```

- [ ] **Step 2: Copy the maintained summary flow into a local notebook**

Create `notebooks/local_result_summary.ipynb` with local project-relative paths and without Drive mount steps.

- [ ] **Step 3: Stop treating the Colab notebooks as maintained entrypoints**

Add a top markdown note in each old maintained Colab notebook:

```markdown
# Deprecated Notebook

This Colab notebook is no longer the maintained workflow.
Use `notebooks/local_evals.ipynb` or `notebooks/local_result_summary.ipynb` for current local runs.
```

- [ ] **Step 4: Clear stale notebook outputs that embed old absolute paths**

Run: `python - <<'PY'
import json
from pathlib import Path
for path in [Path('notebooks/local_evals.ipynb'), Path('notebooks/local_result_summary.ipynb')]:
    data = json.loads(path.read_text())
    for cell in data['cells']:
        if cell.get('cell_type') == 'code':
            cell['execution_count'] = None
            cell['outputs'] = []
    path.write_text(json.dumps(data, ensure_ascii=False, indent=1) + '\n')
PY`
Expected: notebooks are saved cleanly with no stale execution output.

### Task 7: Remove stale absolute-path assumptions from maintained runtime code

**Files:**
- Modify: `notebooks/local_evals.ipynb`
- Modify: `notebooks/local_result_summary.ipynb`
- Modify: `README.md`

- [ ] **Step 1: Search for old absolute paths in maintained local entrypoints**

Run: `rg -n "/Users/mckay|/content/drive|MyDrive|WorkSpaces/nufrost" notebooks/local_*.ipynb README.md`
Expected: no executable path assumptions remain in maintained local notebooks after cleanup.

- [ ] **Step 2: Update runtime docs to point to the new cache layout**

Add or update README text like:

```markdown
- Local cache root: `data/cache/local`
- Colab cache root: `data/cache/colab`
- VRT cache location: `data/cache/<env>/vrts/`
- NPZ cache location: `data/cache/<env>/npz/`
```

- [ ] **Step 3: Re-run the path search and confirm cleanup**

Run: `rg -n "/Users/mckay|/content/drive|MyDrive|WorkSpaces/nufrost" notebooks/local_*.ipynb README.md`
Expected: no matches in maintained local runtime files.

### Task 8: Verify the local workflow end to end

**Files:**
- Modify: `tests/test_evaluation.py`
- Test: `tests/test_data_loader.py`
- Test: `tests/test_config.py`

- [ ] **Step 1: Add a local evaluation smoke test if needed**

```python
def test_load_evaluation_cube_uses_local_cache_layout(fixtures_input_dir, tmp_path):
    from config import build_args
    from src.data_loader import find_image_chunks
    from src.evaluation import load_evaluation_cube

    cache_root = tmp_path / "cache" / "local"
    chunks = find_image_chunks(fixtures_input_dir.as_posix(), 100.112, 25.654, "B2", cache_dir=cache_root)
    args = build_args({})
    args.cache_dir = cache_root.as_posix()
    args.force_refresh = True
    prepared = load_evaluation_cube(chunks, args)

    assert prepared["cube"].ndim == 3
    assert (cache_root / "vrts").exists()
    assert (cache_root / "npz").exists()
```

- [ ] **Step 2: Run focused tests**

Run: `pytest tests/test_data_loader.py tests/test_config.py tests/test_evaluation.py -q`
Expected: PASS.

- [ ] **Step 3: Run the full test suite**

Run: `pytest tests -q`
Expected: PASS.

- [ ] **Step 4: Run one real local chunk smoke script outside fixtures**

Run:

```bash
python - <<'PY'
from config import build_args
from src.data_loader import find_image_chunks
from src.evaluation import load_evaluation_cube

chunks = find_image_chunks('data/hls', 91.2734, 29.7904, 'BLUE', cache_dir='data/cache/local')
args = build_args({})
args.cache_dir = 'data/cache/local'
args.force_refresh = True
prepared = load_evaluation_cube(chunks, args)
print(prepared['cube'].shape)
PY
```

Expected: prints a valid cube shape without missing-file errors.
