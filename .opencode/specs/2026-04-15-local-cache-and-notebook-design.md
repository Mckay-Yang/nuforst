# Local Cache And Notebook Design

## Goal

Make the project runnable from the new local checkout by removing stale absolute-path assumptions, standardizing cache layout under `data/cache/`, and introducing local notebook entrypoints for the unified evaluation workflow.

## Scope

This design covers three linked problems:

1. Local path assumptions no longer match the current checkout root.
2. Cache layout is inconsistent across code paths and mixes VRT and NPZ artifacts.
3. The active evaluation workflow is currently notebook-first and still framed around Colab.

This design does not attempt to modernize every historical notebook. It focuses on the unified evaluation workflow and the shared data-loading/cache code that all entrypoints depend on.

## Decisions

### 1. Canonical cache root

The project-wide cache root becomes `data/cache/`.

Environment-specific subdirectories:

- `data/cache/local/`
- `data/cache/colab/`

Artifact-specific subdirectories inside each environment cache:

- `data/cache/<env>/vrts/`
- `data/cache/<env>/npz/`

No cache artifacts should be written directly into `data/`, `data/local_cache/`, or `data/colab_cache/` after this migration.

### 2. Cache ownership and resolution

`src/data_loader.py` is the single source of truth for cache path resolution.

Expected behavior:

- `find_image_chunks()` always writes VRTs under the resolved `vrts/` directory.
- `RSCube` always writes NPZ files under the resolved `npz/` directory.
- Callers pass either a cache root (`data/cache/local`) or let defaults resolve from configuration.
- Data loader code, not notebooks, decides where `vrts/` and `npz/` live.

This removes duplicated path logic from notebooks and prevents future layout drift.

### 3. Local versus Colab defaults

The codebase keeps both cache environments available, but local becomes the actively maintained default.

Default behavior:

- Local code paths use `data/cache/local`.
- Colab-oriented paths, where retained, use `data/cache/colab`.

Colab notebooks are not deleted in this phase, but they stop being the maintained workflow.

### 4. Notebook strategy

The maintained notebook entrypoints become local notebooks:

- `notebooks/local_evals.ipynb`
- `notebooks/local_result_summary.ipynb`

The unified evaluation flow remains notebook-first, but with local filesystem assumptions:

- derive project root locally
- use local data directories
- use `data/cache/local`
- write outputs under `data/output`

The current `colab_evals.ipynb` is used only as a migration source, not as the maintained runtime notebook.

### 5. Absolute path cleanup

Hardcoded absolute paths under the old Google Drive checkout are treated as invalid technical debt.

Required cleanup:

- Replace executable path assumptions in notebooks and Python code with project-root-relative paths.
- Clear stale notebook outputs that embed old absolute paths where needed.
- Keep historical prose references only if they are non-executable and clearly archival.

### 6. VRT path correctness

The migration must fix current local VRT loading failures.

Observed failure mode:

- `find_image_chunks()` can build a VRT, but subsequent `load_evaluation_cube()` fails because the generated VRT does not resolve source TIFF paths correctly from the new local checkout.

Design requirement:

- VRT generation must produce paths that rasterio/GDAL can reopen reliably from the VRT location on the local filesystem.

This is a release blocker for the local workflow and must be verified with a real local chunk.

## File-level design

### `src/data_loader.py`

Responsibilities after migration:

- Resolve cache roots and subdirectories.
- Build VRTs in the environment-specific `vrts/` directory.
- Write NPZ caches in the environment-specific `npz/` directory.
- Generate VRT source references that are valid from the VRT file location.

Potential additions are small helpers rather than a new subsystem, for example:

- resolve cache directories
- normalize cache root input
- choose relative or absolute VRT source filenames safely

### `config/settings.py` and config defaults

Responsibilities after migration:

- expose the default local cache root as `data/cache/local`
- preserve override behavior through `build_args(overrides=...)`

The existing precedence model stays unchanged.

### `notebooks/local_evals.ipynb`

Responsibilities after migration:

- serve as the maintained unified evaluation notebook for local runs
- use local project/data/cache/output paths
- keep the current evaluation coverage model

### `notebooks/local_result_summary.ipynb`

Responsibilities after migration:

- read CSV outputs from local runs
- provide local summary and plotting support without Colab assumptions

### Historical Colab notebooks

Responsibilities after migration:

- remain in the repo as historical artifacts for now
- are not updated beyond any minimal path cleanup needed to keep the repo coherent

## Error handling

Path and cache handling should fail loudly and specifically.

Expected failures:

- missing input TIFFs
- stale or unreadable VRT references
- missing cache root parent directories when caller passes invalid paths

Expected messaging:

- include the resolved cache path in cache logs
- include the VRT path and failing source path when VRT-backed reads fail

## Testing strategy

### Code-level verification

Add or update tests to verify:

- cache path resolution routes NPZ to `npz/` and VRT to `vrts/`
- `find_image_chunks()` returns a VRT under the expected cache directory
- `RSCube.load()` can read from a VRT generated from real local fixture TIFFs

### Workflow-level verification

Run a local smoke check that:

1. builds a VRT for one real chunk
2. loads the cube successfully
3. confirms resulting cache files appear under `data/cache/local/{vrts,npz}`

### Notebook-level verification

Verify the local notebook JSON is valid and the first runtime cells execute with local paths.

## Success criteria

The migration is complete when all of the following are true:

- local evaluation code no longer depends on the old absolute checkout path
- the active unified evaluation notebook has a maintained local version
- shared code writes caches only under `data/cache/<env>/{vrts,npz}`
- a real local chunk can be converted to VRT and loaded into a cube successfully
- tests and smoke checks confirm the local workflow works end to end
