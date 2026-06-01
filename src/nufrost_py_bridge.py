import json
from typing import Dict, Any, Optional
import numpy as np

import nufrost_py


def reconstruct_nufrost_rust(
    cube: np.ndarray,
    timestamps: np.ndarray,
    target_time: float,
    config: Dict[str, Any],
) -> np.ndarray:
    """Run NUFROST reconstruction using Rust backend.

    Args:
        cube: 3-D array of shape (T, H, W) — observations over time.
        timestamps: 1-D array of length T — timestamps in any consistent unit.
        target_time: Target timestamp for prediction, in same unit as timestamps.
        config: NUFROST configuration dict matching NufrostConfig struct.

    Returns:
        2-D array of shape (H, W) with predicted values.
    """
    T, H, W = cube.shape
    if len(timestamps) != T:
        raise ValueError(
            f"timestamps length ({len(timestamps)}) must match cube time dim ({T})"
        )

    # Ensure C-contiguous float64 for Rust interop
    cube_2d = np.ascontiguousarray(
        cube.reshape(T, H * W).astype(np.float64, copy=False)
    )
    ts_1d = timestamps.astype(np.float64, copy=False)
    config_json = json.dumps(config)

    predictions = nufrost_py.nufrost_raster_rust(ts_1d, cube_2d, float(target_time), config_json)
    return np.asarray(predictions, dtype=np.float32).reshape(H, W)


def reconstruct_hants_rust(
    cube: np.ndarray,
    timestamps: np.ndarray,
    target_time: float,
    config: Dict[str, Any],
) -> np.ndarray:
    """Run HANTS reconstruction using Rust backend.

    Args:
        cube: 3-D array of shape (T, H, W).
        timestamps: 1-D array of length T — timestamps in any consistent unit.
        target_time: Target timestamp, same unit as timestamps.
        config: HANTS configuration dict matching HantsConfig struct.

    Returns:
        2-D array of shape (H, W) with predicted values.
    """
    T, H, W = cube.shape
    if len(timestamps) != T:
        raise ValueError(
            f"timestamps length ({len(timestamps)}) must match cube time dim ({T})"
        )

    cube_2d = np.ascontiguousarray(
        cube.reshape(T, H * W).astype(np.float64, copy=False)
    )
    ts_1d = timestamps.astype(np.float64, copy=False)
    config_json = json.dumps(config)

    predictions = nufrost_py.hants_raster_rust(ts_1d, cube_2d, float(target_time), config_json)
    return np.asarray(predictions, dtype=np.float32).reshape(H, W)


def reconstruct_zhu2015_rust(
    cube: np.ndarray,
    timestamps: np.ndarray,
    target_time: float,
    config: Dict[str, Any],
) -> np.ndarray:
    """Run Zhu2015 reconstruction using Rust backend.

    Args:
        cube: 3-D array of shape (T, H, W) — observations over time.
        timestamps: 1-D array of length T — timestamps in any consistent unit.
        target_time: Target timestamp for prediction, in same unit as timestamps.
        config: Zhu2015 configuration dict matching Zhu2015Config struct.

    Returns:
        2-D array of shape (H, W) with predicted values.
    """
    T, H, W = cube.shape
    if len(timestamps) != T:
        raise ValueError(
            f"timestamps length ({len(timestamps)}) must match cube time dim ({T})"
        )

    cube_2d = np.ascontiguousarray(
        cube.reshape(T, H * W).astype(np.float64, copy=False)
    )
    ts_1d = timestamps.astype(np.float64, copy=False)
    config_json = json.dumps(config)

    pred = nufrost_py.zhu2015_raster_rust(ts_1d, cube_2d, float(target_time), config_json)
    return np.asarray(pred, dtype=np.float32).reshape(H, W)

    cube_2d = np.ascontiguousarray(
        cube.reshape(T, H * W).astype(np.float64, copy=False)
    )
    ts_1d = timestamps.astype(np.float64, copy=False)
    config_json = json.dumps(config)

    pred, qa = nufrost_py.zhu2015_raster_rust(ts_1d, cube_2d, float(target_time), config_json)
    return (
        np.asarray(pred, dtype=np.float32).reshape(H, W),
        np.asarray(qa, dtype=np.int32).reshape(H, W),
    )
