import argparse
from pathlib import Path
import sys
from typing import Sequence

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.local_eval_workflow import run_local_evals_workflow


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run local evaluation experiments without a notebook.")
    parser.add_argument("--source-name", required=True, choices=("sentinel-2", "hls"))
    parser.add_argument("--output-dir", type=Path, default=REPO_ROOT / "data/output")
    parser.add_argument("--cache-dir", type=Path, default=REPO_ROOT / "data/cache/local")
    parser.add_argument("--max-images", type=int)
    parser.add_argument("--n-jobs", type=int, default=-1)
    parser.add_argument("--run-ablation", action="store_true")
    parser.add_argument("--run-sparse", action="store_true")
    parser.add_argument("--run-gap", action="store_true")
    parser.add_argument("--run-repeatability", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    selected = [args.run_ablation, args.run_sparse, args.run_gap, args.run_repeatability]
    if not any(selected):
        args.run_ablation = True
        args.run_sparse = True
        args.run_gap = True
        args.run_repeatability = True

    run_local_evals_workflow(
        source_name=args.source_name,
        project_dir=REPO_ROOT,
        output_dir=args.output_dir,
        cache_dir=args.cache_dir,
        max_images=args.max_images,
        n_jobs=args.n_jobs,
        run_ablation=args.run_ablation,
        run_sparse=args.run_sparse,
        run_gap=args.run_gap,
        run_repeatability=args.run_repeatability,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
