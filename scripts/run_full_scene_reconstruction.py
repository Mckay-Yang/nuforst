import argparse
from pathlib import Path
import sys
from typing import Sequence

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.full_scene_reconstruction import reconstruct_full_scene_for_all_locations, reconstruct_full_scene_for_location


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run a full-scene reconstruction.")
    parser.add_argument("--source-name", required=True)
    parser.add_argument("--lon", type=float)
    parser.add_argument("--lat", type=float)
    parser.add_argument("--all-coordinates", action="store_true")
    parser.add_argument("--output-root", type=Path, default=Path("data/output"))
    parser.add_argument("--data-root", type=Path, default=Path("data"))
    parser.add_argument("--cache-dir", type=Path, default=Path("data/cache/local"))
    parser.add_argument("--methods", nargs="+", default=["nufrost", "hants", "zhu2015"])
    parser.add_argument("--n-jobs", type=int, default=-1)
    parser.add_argument("--force-refresh", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    common_kwargs = {
        "source_name": args.source_name,
        "output_root": args.output_root,
        "data_root": args.data_root,
        "cache_dir": args.cache_dir,
        "methods": tuple(args.methods),
        "n_jobs": args.n_jobs,
        "force_refresh": args.force_refresh,
    }
    if args.all_coordinates:
        reconstruct_full_scene_for_all_locations(**common_kwargs)
        return 0

    if args.lon is None or args.lat is None:
        raise SystemExit("--lon and --lat are required unless --all-coordinates is set")

    reconstruct_full_scene_for_location(lon=args.lon, lat=args.lat, **common_kwargs)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
