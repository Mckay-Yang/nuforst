from pathlib import Path

from config import build_args


def test_build_args_applies_overrides_and_path_types() -> None:
    args = build_args("nufrost",
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
    args = build_args("nufrost",{"image": image_list})
    assert args.image == image_list


def test_build_args_uses_local_cache_default() -> None:
    args = build_args("nufrost",{})
    assert args.cache_dir == Path("data/cache/local")


def test_hants_default_fet_matches_sentinel2_dn_scale() -> None:
    args = build_args("hants", {})
    assert args.fet == 500.0
    assert args.sf == "high"


def test_zhu2015_default_lasso_alpha_is_tuned_value() -> None:
    args = build_args("zhu2015", {})
    assert args.lasso_alpha == 0.1
