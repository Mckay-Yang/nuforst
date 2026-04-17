from pathlib import Path

from config import build_args


def test_build_args_applies_overrides_and_path_types() -> None:
    args = build_args(
        {
            "cache_dir": "tests/runtime/cache",
            "output_path": "tests/runtime/out.tif",
            "target_time": None,
            "start_time": "2020-01-02T00:00:00",
            "image": "tests/fixtures/input/example.tif",
            "include_trend": False,
        }
    )

    assert args.cache_dir == Path("tests/runtime/cache")
    assert args.output_path == Path("tests/runtime/out.tif")
    assert args.image == Path("tests/fixtures/input/example.tif")
    assert args.target_time == args.start_time == "2020-01-02T00:00:00"
    assert args.include_trend is False


def test_build_args_preserves_list_images() -> None:
    image_list = ["a.tif", "b.tif"]
    args = build_args({"image": image_list})
    assert args.image == image_list


def test_build_args_uses_local_cache_default() -> None:
    args = build_args({})
    assert args.cache_dir == Path("data/cache/local")
