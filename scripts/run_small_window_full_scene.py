import argparse
from pathlib import Path
from typing import Sequence

from src.full_scene_reconstruction import reconstruct_full_scene_for_location


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run a small-window full-scene reconstruction.")
    parser.add_argument("--source-name", required=True)
    parser.add_argument("--lon", type=float, required=True)
    parser.add_argument("--lat", type=float, required=True)
    parser.add_argument("--output-root", type=Path, default=Path("data/output"))
    parser.add_argument("--data-root", type=Path, default=Path("data"))
    parser.add_argument("--cache-dir", type=Path, default=Path("data/cache/local"))
    parser.add_argument("--window-size", type=int, required=True)
    parser.add_argument("--methods", nargs="+", default=["nufrost", "hants", "zhu2015"])
    parser.add_argument("--n-jobs", type=int, default=-1)
    parser.add_argument("--force-refresh", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    reconstruct_full_scene_for_location(
        source_name=args.source_name,
        lon=args.lon,
        lat=args.lat,
        output_root=args.output_root,
        data_root=args.data_root,
        cache_dir=args.cache_dir,
        methods=tuple(args.methods),
        n_jobs=args.n_jobs,
        force_refresh=args.force_refresh,
        window_size=args.window_size,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
