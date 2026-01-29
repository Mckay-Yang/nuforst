import rasterio
from pathlib import Path
from typing import Optional, cast
import numpy as np
from .config import build_args, Args
from .data_loader import RSCube
from .reconstruction import revive


def run_reconstruction(args: Args) -> Path:
    print(f"[System] Input Image: {args.image}")

    try:
        cube_loader = RSCube(args.image, cache_dir=args.cache_dir, force_refresh=args.force_refresh)
        cube_data = cube_loader.load()
    except FileNotFoundError as e:
        print(f"\n[Error] {e}")
        print("Please check the --image path.")
        raise

    cube = cast(np.ndarray, cube_data["cube"])
    timestamps = cast(np.ndarray, cube_data["timestamps"])
    print(f"[Data] Cube shape: {cube.shape}")
    print(f"[Data] Timestamps (first 5): {timestamps[:5]}")

    # Reconstruct at target time
    print(f"[System] Target Time: {args.target_time}")
    recon = revive(cube, timestamps, args.target_time, args)
    print(f"[Result] Reconstructed map shape: {recon.shape}")

    # Save reconstructed image
    output_path = Path(args.output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    transform = None

    if "transform" in cube_data:
        try:
            transform = rasterio.Affine(*cube_data["transform"])
        except Exception:
            transform = None

    with rasterio.open(
        output_path,
        "w",
        driver="GTiff",
        height=recon.shape[0],
        width=recon.shape[1],
        count=1,
        dtype=recon.dtype,
        crs=cube_data.get("crs_wkt", None),
        transform=transform,
    ) as dst:
        dst.write(recon, 1)
    print(f"[Success] Saved to: {output_path}")
    return output_path


def main():
    args = build_args()

    run_reconstruction(args)

if __name__ == "__main__":
    main()
