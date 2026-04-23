from pathlib import Path

from src.data_loader import RSCube, TimeSeriesRasterSource, find_image_chunks


def test_rscube_loads_single_fixture_and_creates_cache(single_tile_path: str, cache_dir: Path) -> None:
    cube = RSCube(single_tile_path, cache_dir=cache_dir, force_refresh=False)
    result = cube.load()

    assert "cube" in result
    assert "timestamps" in result
    assert "band_names" in result
    assert result["cube"].ndim == 3
    assert len(result["timestamps"]) == result["cube"].shape[0]
    assert len(result["band_names"]) == result["cube"].shape[0]
    assert any((cache_dir / "npz").glob("*.npz"))


def test_rscube_hits_npz_cache_on_second_load(single_tile_path: str, cache_dir: Path) -> None:
    cube = RSCube(single_tile_path, cache_dir=cache_dir, force_refresh=False)
    first = cube.load()
    second = cube.load()

    assert first["cube"].shape == second["cube"].shape
    assert first["cache_path"] == second["cache_path"]


def test_find_image_chunks_builds_vrt(fixture_input_dir: Path, cache_dir: Path) -> None:
    chunks = find_image_chunks(fixture_input_dir.as_posix(), lon=100.112, lat=25.654, band="B2", cache_dir=cache_dir)

    assert len(chunks) == 1
    vrt_path = Path(chunks[0])
    assert vrt_path.suffix == ".vrt"
    assert vrt_path.exists()
    assert vrt_path.parent == cache_dir / "vrts"


def test_rscube_writes_npz_under_cache_npz(single_tile_path: str, cache_dir: Path) -> None:
    cube = RSCube(single_tile_path, cache_dir=cache_dir, force_refresh=True)
    result = cube.load()

    cache_path = Path(result["cache_path"])
    assert cache_path.parent == cache_dir / "npz"
    assert cache_path.exists()


def test_rscube_can_load_cube_from_generated_vrt(fixture_input_dir: Path, cache_dir: Path) -> None:
    chunks = find_image_chunks(fixture_input_dir.as_posix(), lon=100.112, lat=25.654, band="B2", cache_dir=cache_dir)

    cube = RSCube(chunks, cache_dir=cache_dir, force_refresh=True)
    result = cube.load()

    assert result["cube"].ndim == 3
    assert result["cube"].shape[1] > 0
    assert result["cube"].shape[2] > 0


def test_streaming_source_reads_vrt_without_creating_npz(fixture_input_dir: Path, cache_dir: Path) -> None:
    chunks = find_image_chunks(fixture_input_dir.as_posix(), lon=100.112, lat=25.654, band="B2", cache_dir=cache_dir)

    with TimeSeriesRasterSource(chunks, cache_dir=cache_dir) as source:
        meta = source.metadata()
        arr = source.read_pixel_series(0, 0)

    assert meta["height"] > 0
    assert meta["width"] > 0
    assert arr.ndim == 1
    assert len(arr) == meta["count"]
    assert not any((cache_dir / "npz").glob("*.npz"))
