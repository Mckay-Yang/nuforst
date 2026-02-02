from .reconstruction import revive
from .zhu2015 import reconstruct_zhu2015
from .hants import reconstruct_hants
from .config import Args, build_args
from .data_loader import RSCube
from pathlib import Path
import numpy as np
import rasterio
from typing import Optional, Union

def reconstruct(
    image: str,
    target_time: str,
    output_path: Optional[str] = None,
    **kwargs
) -> np.ndarray:
    """
    一键重建接口。

    参数:
        image: 输入的多波段 TIF 路径
        target_time: 目标重建时间 (如 '2023-06-15')
        output_path: 输出 TIF 路径 (可选)
        **kwargs: 其他算法参数 (如 n_jobs, ridge, etc.)
    """
    # 1. 构建参数
    overrides = {**kwargs, "image": image, "target_time": target_time}
    if output_path:
        overrides["output_path"] = output_path

    args = build_args(overrides=overrides)

    # 2. 加载数据
    loader = RSCube(args.image, cache_dir=args.cache_dir, force_refresh=args.force_refresh)
    data = loader.load()
    cube = data["cube"]
    timestamps = data["timestamps"]

    # 3. 执行重建
    recon = revive(cube, timestamps, args.target_time, args=args)

    # 4. 保存结果
    if args.output_path:
        out_p = Path(args.output_path)
        out_p.parent.mkdir(parents=True, exist_ok=True)

        transform = None
        if "transform" in data:
            transform = rasterio.Affine(*data["transform"])

        with rasterio.open(
            out_p, "w",
            driver="GTiff",
            height=recon.shape[0],
            width=recon.shape[1],
            count=1,
            dtype=recon.dtype,
            crs=data.get("crs_wkt"),
            transform=transform,
        ) as dst:
            dst.write(recon, 1)
        print(f"[Success] Saved to: {out_p}")

    return recon

__all__ = ["reconstruct", "reconstruct_zhu2015", "reconstruct_hants", "revive", "Args", "RSCube", "build_args"]
