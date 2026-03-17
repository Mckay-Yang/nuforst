from src.data_loader import RSCube

def test_load_file() -> None:
    cube = RSCube(
        "/Users/mckay/Library/CloudStorage/GoogleDrive-yangluhao990714@gmail.com/我的云端硬盘/WorkSpaces/nufrost/data/test_sample/input/COPERNICUS_S2_HARMONIZED_B2_lon91.2734_lat29.7904.tif",
        cache_dir="/Users/mckay/Library/CloudStorage/GoogleDrive-yangluhao990714@gmail.com/我的云端硬盘/WorkSpaces/nufrost/data/test_sample/cache",
        force_refresh=False,
    )
    result = cube.load()

    assert "cube" in result
    assert "timestamps" in result
    assert "band_names" in result

    data = result["cube"]
    timestamps = result["timestamps"]
    band_names = result["band_names"]

    assert data.ndim == 3
    assert len(timestamps) == data.shape[0]
    assert len(band_names) == data.shape[0]
    breakpoint()