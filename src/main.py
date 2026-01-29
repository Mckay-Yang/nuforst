import rasterio
from pathlib import Path
from typing import Optional, cast
import numpy as np
from .config import build_args, Args
from .data_loader import RSCube
from .reconstruction import revive


def run_reconstruction(args: Args) -> Path:
    """CLI 版本的运行逻辑"""
    from . import reconstruct

    reconstruct(
        image=args.image,
        target_time=args.target_time,
        output_path=args.output_path,
        args=args
    )
    return Path(args.output_path)


def main():
    args = build_args()

    run_reconstruction(args)

if __name__ == "__main__":
    main()
