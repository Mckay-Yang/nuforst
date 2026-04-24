import hashlib
import json
import rasterio
import numpy as np
from pathlib import Path
from datetime import datetime
from typing import Dict, Iterator, Tuple, Optional, Union, List
import glob
import re
from collections import defaultdict
import subprocess
import os
import xml.etree.ElementTree as ET
from rasterio.windows import Window


GDAL_DTYPE_MAP = {
    "uint8": "Byte",
    "int8": "Byte",
    "uint16": "UInt16",
    "int16": "Int16",
    "uint32": "UInt32",
    "int32": "Int32",
    "uint64": "UInt64",
    "int64": "Int64",
    "float32": "Float32",
    "float64": "Float64",
    "complex64": "CFloat32",
    "complex128": "CFloat64",
}

READ_LAYOUT_VERSION = "full-image-v1"


def _resolve_cache_root(cache_dir: Optional[Union[str, Path]]) -> Path:
    if cache_dir is None:
        return Path(__file__).resolve().parent.parent / "data" / "cache" / "local"
    return Path(cache_dir)


def _cache_subdir(cache_root: Path, kind: str) -> Path:
    path = cache_root / kind
    path.mkdir(parents=True, exist_ok=True)
    return path


def _parse_band_timestamp(name: str) -> str:
    tokens = [name.strip()]

    match = re.search(r'(\d{8}T\d{6})', name)
    if match:
        try:
            return datetime.strptime(match.group(1), "%Y%m%dT%H%M%S").isoformat()
        except ValueError:
            pass

    digits = "".join(ch for ch in name if ch.isdigit())
    if len(digits) >= 8:
        tokens.append(digits[:8])
    if len(digits) >= 14:
        tokens.append(digits[:14])
    fmts = ("%Y%m%d", "%Y-%m-%d", "%Y/%m/%d", "%Y%m%dT%H%M%S", "%Y-%m-%dT%H:%M:%S")
    for tok in tokens:
        for fmt in fmts:
            try:
                return datetime.strptime(tok, fmt).isoformat()
            except ValueError:
                continue
    return name.strip() or "band"


def _gdal_dtype_name(dtype: str) -> str:
    return GDAL_DTYPE_MAP.get(np.dtype(dtype).name, str(dtype))


def _log_cache_event(kind: str, status: str, path: Union[str, Path], detail: Optional[str] = None) -> None:
    msg = f"[DataLoader] {kind} cache {status}: {Path(path)}"
    if detail:
        msg += f" ({detail})"
    print(msg, flush=True)


def _is_stale(target: Path, sources: List[Path]) -> bool:
    if not target.exists():
        return True

    target_mtime = target.stat().st_mtime
    for src in sources:
        if (not src.exists()) or src.stat().st_mtime > target_mtime:
            return True
    return False


def _part_sort_key(part_name: str) -> Tuple[int, int, str]:
    if part_name == "full":
        return (0, 0, part_name)

    match = re.fullmatch(r"part(\d+)", part_name)
    if match:
        return (1, int(match.group(1)), part_name)

    return (2, 0, part_name)


def _write_stacked_vrt(vrt_path: Path, ordered_part_files: List[List[Path]]) -> None:
    if not ordered_part_files:
        raise ValueError("No source TIFF parts were provided.")

    crs = None
    res_x: Optional[float] = None
    res_y: Optional[float] = None
    min_x = float("inf")
    min_y = float("inf")
    max_x = float("-inf")
    max_y = float("-inf")
    part_infos = []

    for part_files in ordered_part_files:
        if not part_files:
            continue

        tile_infos = []
        descriptions = None
        band_count = None
        nodata = None
        data_type = None

        for tile_path in part_files:
            with rasterio.open(tile_path) as src:
                transform = src.transform
                if not np.isclose(transform.b, 0.0) or not np.isclose(transform.d, 0.0):
                    raise ValueError(f"Rotated geotransform is not supported: {tile_path}")

                if crs is None:
                    crs = src.crs
                elif src.crs != crs:
                    raise ValueError(f"CRS mismatch while building VRT: {tile_path}")

                tile_res_x = float(transform.a)
                tile_res_y = float(abs(transform.e))
                if res_x is None or res_y is None:
                    res_x = tile_res_x
                    res_y = tile_res_y
                elif (not np.isclose(tile_res_x, res_x)) or (not np.isclose(tile_res_y, res_y)):
                    raise ValueError(f"Pixel size mismatch while building VRT: {tile_path}")

                if band_count is None:
                    band_count = src.count
                    descriptions = list(src.descriptions)
                    if not descriptions or all(name is None or name == "" for name in descriptions):
                        descriptions = [f"band_{i+1}" for i in range(src.count)]
                    nodata = src.nodatavals[0] if src.nodatavals else None
                    data_type = _gdal_dtype_name(src.dtypes[0])
                elif src.count != band_count:
                    raise ValueError(f"Band count mismatch within part for {tile_path}")

                block_h, block_w = src.block_shapes[0] if src.block_shapes else (src.height, src.width)
                bounds = src.bounds
                min_x = min(min_x, float(bounds.left))
                min_y = min(min_y, float(bounds.bottom))
                max_x = max(max_x, float(bounds.right))
                max_y = max(max_y, float(bounds.top))

                tile_infos.append({
                    "path": tile_path,
                    "width": src.width,
                    "height": src.height,
                    "block_w": block_w,
                    "block_h": block_h,
                    "x_origin": float(transform.c),
                    "y_origin": float(transform.f),
                })

        if band_count is None or data_type is None:
            continue

        part_infos.append({
            "band_count": band_count,
            "descriptions": descriptions,
            "nodata": nodata,
            "data_type": data_type,
            "tiles": tile_infos,
        })

    if not part_infos or crs is None or res_x is None or res_y is None:
        raise ValueError("No valid TIFF metadata found to build the VRT.")

    width = int(round((max_x - min_x) / res_x))
    height = int(round((max_y - min_y) / res_y))
    transform = rasterio.transform.from_origin(min_x, max_y, res_x, res_y)

    root = ET.Element("VRTDataset", rasterXSize=str(width), rasterYSize=str(height))

    srs = ET.SubElement(root, "SRS")
    srs.text = crs.to_wkt()

    geotransform = ET.SubElement(root, "GeoTransform")
    geotransform.text = ", ".join(f"{v:.16g}" for v in transform.to_gdal())

    out_band_idx = 1
    for part_info in part_infos:
        for src_band_idx in range(1, part_info["band_count"] + 1):
            vrt_band = ET.SubElement(
                root,
                "VRTRasterBand",
                dataType=part_info["data_type"],
                band=str(out_band_idx),
            )

            description = part_info["descriptions"][src_band_idx - 1]
            desc_elem = ET.SubElement(vrt_band, "Description")
            desc_elem.text = str(description)

            if part_info["nodata"] is not None:
                nodata_elem = ET.SubElement(vrt_band, "NoDataValue")
                nodata_elem.text = str(part_info["nodata"])

            color_interp = ET.SubElement(vrt_band, "ColorInterp")
            color_interp.text = "Gray"

            for tile_info in part_info["tiles"]:
                col_off_f, row_off_f = (~transform) * (tile_info["x_origin"], tile_info["y_origin"])
                col_off = int(round(col_off_f))
                row_off = int(round(row_off_f))

                simple_source = ET.SubElement(vrt_band, "SimpleSource")
                src_filename = ET.SubElement(simple_source, "SourceFilename", relativeToVRT="0")
                src_filename.text = str(Path(tile_info["path"]).resolve())
                src_band = ET.SubElement(simple_source, "SourceBand")
                src_band.text = str(src_band_idx)

                ET.SubElement(
                    simple_source,
                    "SourceProperties",
                    RasterXSize=str(tile_info["width"]),
                    RasterYSize=str(tile_info["height"]),
                    DataType=part_info["data_type"],
                    BlockXSize=str(tile_info["block_w"]),
                    BlockYSize=str(tile_info["block_h"]),
                )
                ET.SubElement(
                    simple_source,
                    "SrcRect",
                    xOff="0",
                    yOff="0",
                    xSize=str(tile_info["width"]),
                    ySize=str(tile_info["height"]),
                )
                ET.SubElement(
                    simple_source,
                    "DstRect",
                    xOff=str(col_off),
                    yOff=str(row_off),
                    xSize=str(tile_info["width"]),
                    ySize=str(tile_info["height"]),
                )

            out_band_idx += 1

    tree = ET.ElementTree(root)
    ET.indent(tree, space="  ")
    tree.write(vrt_path, encoding="utf-8")

class RSCube:
    """
    Handle spatiotemporal data cube from one or more multi-band GeoTIFFs where each band is a timestamped slice.
    Data are cached as compressed npz to avoid repeatedly reading the large TIFF(s).
    If multiple TIFFs are provided, they are concatenated along the time (band) axis.
    """
    def __init__(
            self,
            tif_path: Union[str, Path, List[Union[str, Path]]],
            cache_dir: Optional[Union[str, Path]] = None,
            force_refresh: bool = False
        ) -> None:
        self.tif_paths = self._resolve_paths(tif_path)
        self.cache_dir = _resolve_cache_root(cache_dir)

        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.force_refresh = force_refresh
        self.meta: Dict[str, object] = {}

    def _resolve_paths(self, tif_path: Union[str, Path, List[Union[str, Path]]]) -> List[Path]:
        if isinstance(tif_path, list):
            return sorted([Path(p) for p in tif_path])

        p_str = str(tif_path)
        if "*" in p_str or "?" in p_str:
            paths = [Path(p) for p in glob.glob(p_str)]
            if not paths:
                raise FileNotFoundError(f"No files matched pattern: {p_str}")
            return sorted(paths)
        else:
            return [Path(tif_path)]

    def _file_signature(self) -> str:
        if not self.tif_paths:
            raise FileNotFoundError("No input images found.")

        # Create a combined signature for all input files
        payloads = []
        for p in self.tif_paths:
            if not p.exists():
                raise FileNotFoundError(f"Input image not found: {p}")
            stat = p.stat()
            payloads.append(f"{p.resolve()}:{stat.st_size}:{stat.st_mtime}")

        combined_payload = "|".join(payloads + [READ_LAYOUT_VERSION])
        return hashlib.md5(combined_payload.encode("utf-8")).hexdigest()[:12]

    def _cache_path(self) -> Path:
        if not self.cache_dir.exists():
            self.cache_dir.mkdir(parents=True, exist_ok=True)

        npz_dir = _cache_subdir(self.cache_dir, "npz")

        # Use the first file's stem as a base name
        base_stem = self.tif_paths[0].stem
        # Optional: remove trailing chunk like '_part1' or '_2015' from the stem for cache name clarity
        parts = base_stem.split('_')
        if len(parts) > 1 and (parts[-1].startswith('part') or (parts[-1].isdigit() and len(parts[-1]) == 4)):
            base_stem = '_'.join(parts[:-1])

        return npz_dir / f"{base_stem}_{self._file_signature()}.npz"

    def _parse_band_timestamp(self, name: str) -> str:
        """Best-effort parse; fall back to the raw band name."""
        return _parse_band_timestamp(name)

    def _read_tif(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        all_arrs = []
        all_band_names = []
        all_timestamps = []

        first_meta = None

        for path in self.tif_paths:
            with rasterio.open(path) as src:
                arr = src.read(masked=True).astype(np.float32)
                if np.ma.isMaskedArray(arr):
                    arr = arr.filled(np.nan)

                band_names = list(src.descriptions)
                if not band_names or all(name is None or name == "" for name in band_names):
                    if str(path).endswith(".vrt"):
                        for f in src.files:
                            if not f.endswith(".vrt"):
                                try:
                                    with rasterio.open(f) as sub_src:
                                        sub_names = list(sub_src.descriptions)
                                        if sub_names and any(name for name in sub_names):
                                            band_names = sub_names
                                            break
                                except Exception:
                                    pass

                if not band_names or all(name is None or name == "" for name in band_names):
                    band_names = [f"band_{i+1}" for i in range(src.count)]
                timestamps = [self._parse_band_timestamp(name) for name in band_names]

                all_arrs.append(arr)
                all_band_names.extend(band_names)
                all_timestamps.extend(timestamps)

                if first_meta is None:
                    first_meta = {
                        "transform": list(src.transform),
                        "crs_wkt": src.crs.to_wkt() if src.crs else None,
                        "height": src.height,
                        "width": src.width,
                    }

        # Crop all arrays to the minimum spatial dimensions to allow concatenation
        if len(all_arrs) > 1:
            min_h = min(arr.shape[1] for arr in all_arrs)
            min_w = min(arr.shape[2] for arr in all_arrs)
            all_arrs = [arr[:, :min_h, :min_w] for arr in all_arrs]
            first_meta["height"] = min_h
            first_meta["width"] = min_w

        # Concatenate along the band/time axis (axis 0)
        combined_arr = np.concatenate(all_arrs, axis=0) if len(all_arrs) > 1 else all_arrs[0]

        # Sort by timestamp to ensure temporal order
        sorted_indices = np.argsort(all_timestamps)
        sorted_timestamps = np.array(all_timestamps, dtype="U32")[sorted_indices]
        sorted_band_names = np.array(all_band_names, dtype="U64")[sorted_indices]
        sorted_arr = combined_arr[sorted_indices]

        self.meta = {
            **first_meta,
            "count": sorted_arr.shape[0],
            "source_files": [str(p) for p in self.tif_paths]
        }

        return sorted_arr, sorted_timestamps, sorted_band_names

    def _save_npz(self, cache_path: Path, cube: np.ndarray, timestamps: np.ndarray, band_names: np.ndarray) -> None:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        stem_prefix = cache_path.name.rsplit("_", 1)[0]
        for sibling in cache_path.parent.glob(f"{stem_prefix}_*.npz"):
            if sibling != cache_path:
                sibling.unlink(missing_ok=True)
        meta = {**self.meta, "cache_path": str(cache_path)}
        np.savez_compressed(cache_path, cube=cube, timestamps=timestamps, band_names=band_names, meta=json.dumps(meta))

    def _load_npz(self, cache_path: Path) -> Dict[str, object]:
        with np.load(cache_path, allow_pickle=False) as z:
            cube = z["cube"]
            timestamps = z["timestamps"]
            band_names = z["band_names"]
            meta = json.loads(z["meta"].item()) if "meta" in z else {}
        self.meta = meta
        return {"cube": cube, "timestamps": timestamps, "band_names": band_names, **meta}

    def load(self) -> Dict[str, object]:
        cache_path = self._cache_path()
        if cache_path.exists() and (not self.force_refresh):
            _log_cache_event("NPZ", "hit", cache_path)
            return self._load_npz(cache_path)

        if cache_path.exists() and self.force_refresh:
            _log_cache_event("NPZ", "refresh", cache_path, detail="force_refresh=True")
        else:
            # Determine file type for logging
            has_vrt = any(str(p).endswith('.vrt') for p in self.tif_paths)
            has_tif = any(str(p).endswith('.tif') for p in self.tif_paths)
            if has_vrt and not has_tif:
                file_type = "VRT"
            elif has_tif and not has_vrt:
                file_type = "TIFF"
            else:
                file_type = "file"
            plural = "s" if len(self.tif_paths) > 1 else ""
            _log_cache_event("NPZ", "miss", cache_path, detail=f"building from {len(self.tif_paths)} {file_type}{plural}")

        cube, timestamps, band_names = self._read_tif()
        self._save_npz(cache_path, cube, timestamps, band_names)
        _log_cache_event("NPZ", "saved", cache_path)
        return {"cube": cube, "timestamps": timestamps, "band_names": band_names, **self.meta, "cache_path": str(cache_path)}


class TimeSeriesRasterSource:
    """Streaming reader for VRT/TIFF time-series stacks that does not create NPZ caches."""

    def __init__(
        self,
        tif_path: Union[str, Path, List[Union[str, Path]]],
        cache_dir: Optional[Union[str, Path]] = None,
    ) -> None:
        self.tif_paths = RSCube(tif_path, cache_dir=cache_dir)._resolve_paths(tif_path)
        if len(self.tif_paths) != 1:
            raise ValueError("TimeSeriesRasterSource expects exactly one VRT/TIFF path.")
        self.path = self.tif_paths[0]
        self.cache_dir = _resolve_cache_root(cache_dir)
        self._src: Optional[rasterio.io.DatasetReader] = None
        self._meta_cache: Optional[Dict[str, object]] = None

    def __enter__(self) -> "TimeSeriesRasterSource":
        self.open()
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()

    def open(self) -> None:
        if self._src is None:
            self._src = rasterio.open(self.path)

    def close(self) -> None:
        if self._src is not None:
            self._src.close()
            self._src = None

    @property
    def src(self):
        if self._src is None:
            self.open()
        assert self._src is not None
        return self._src

    def metadata(self) -> Dict[str, object]:
        if self._meta_cache is not None:
            return self._meta_cache

        src = self.src
        band_names = list(src.descriptions)
        if not band_names or all(name is None or name == "" for name in band_names):
            band_names = [f"band_{i+1}" for i in range(src.count)]
        timestamps = np.array([_parse_band_timestamp(name) for name in band_names], dtype="U32")
        band_names_arr = np.array(band_names, dtype="U64")

        self._meta_cache = {
            "transform": list(src.transform),
            "crs_wkt": src.crs.to_wkt() if src.crs else None,
            "height": src.height,
            "width": src.width,
            "count": src.count,
            "timestamps": timestamps,
            "band_names": band_names_arr,
            "source_files": [str(self.path)],
        }
        return self._meta_cache

    def iter_windows(self, block_shape: Tuple[int, int] = (256, 256)) -> Iterator[Tuple[slice, slice]]:
        height = int(self.metadata()["height"])
        width = int(self.metadata()["width"])
        block_h, block_w = block_shape
        for row0 in range(0, height, block_h):
            for col0 in range(0, width, block_w):
                yield (slice(row0, min(row0 + block_h, height)), slice(col0, min(col0 + block_w, width)))

    def read_window(self, row_slice: slice, col_slice: slice) -> np.ndarray:
        src = self.src
        window = Window.from_slices(row_slice, col_slice)
        arr = src.read(window=window, masked=True).astype(np.float32)
        if np.ma.isMaskedArray(arr):
            arr = arr.filled(np.nan)
        return arr

    def read_pixel_series(self, row: int, col: int) -> np.ndarray:
        arr = self.read_window(slice(row, row + 1), slice(col, col + 1))
        return arr[:, 0, 0]


def find_image_chunks(
    data_dir: str,
    lon: float,
    lat: float,
    band: str,
    cache_dir: Optional[Union[str, Path]] = None,
) -> List[str]:
    """
    Build one time-stacked VRT for a given location and spectral band.

    Notes
    -----
    - The spectral band is encoded in the filename (e.g. ``BLUE`` / ``RED``).
    - The TIFF's internal bands represent the time series.
    - ``partN`` in the filename represents temporal chunking.
    - GEE auto-generated spatial tiles (``-row-col`` suffixes) are first mosaiced
      per part, then all parts are stacked into a single final VRT.

    Returns
    -------
    List[str]
        A one-element list containing the final VRT path for this
        coordinate/band. The list wrapper is kept for caller compatibility.
    """
    pattern = str(Path(data_dir) / f"*_{band}_lon{lon}_lat{lat}*.tif")
    files = sorted(glob.glob(pattern))
    if not files:
        _log_cache_event(
            "VRT",
            "miss",
            Path(data_dir) / f"{band}_lon{lon}_lat{lat}.vrt",
            detail="no matching TIFF files",
        )
        return []

    # Group files by their temporal part
    parts = defaultdict(list)
    for f in files:
        match = re.search(r"_(part\d+)(?:-\d{10}-\d{10})?\.tif$", Path(f).name)
        part_name = match.group(1) if match else "full"
        parts[part_name].append(f)

    cache_root = _resolve_cache_root(cache_dir) if cache_dir is not None else (Path(data_dir).parent / "cache" / "local")
    vrt_dir = _cache_subdir(cache_root, "vrts")

    ordered_part_files = [
        sorted(Path(f) for f in parts[part_name])
        for part_name in sorted(parts.keys(), key=_part_sort_key)
    ]

    final_vrt_path = vrt_dir / f"{band}_lon{lon}_lat{lat}.vrt"
    all_source_files = [tile_path for part_files in ordered_part_files for tile_path in part_files]
    if _is_stale(final_vrt_path, all_source_files):
        if final_vrt_path.exists():
            _log_cache_event("VRT", "refresh", final_vrt_path, detail="source TIFFs changed")
        else:
            _log_cache_event("VRT", "miss", final_vrt_path, detail=f"building from {len(all_source_files)} TIFF tile(s)")
        _write_stacked_vrt(final_vrt_path, ordered_part_files)
        _log_cache_event("VRT", "saved", final_vrt_path)
    else:
        _log_cache_event("VRT", "hit", final_vrt_path)

    for legacy_vrt in vrt_dir.glob(f"{band}_lon{lon}_lat{lat}_part*.vrt"):
        try:
            legacy_vrt.unlink()
        except OSError:
            pass

    legacy_part_dir = cache_root / ".vrt_parts"
    if legacy_part_dir.exists():
        for legacy_vrt in legacy_part_dir.glob(f"{band}_lon{lon}_lat{lat}_part*.vrt"):
            try:
                legacy_vrt.unlink()
            except OSError:
                pass
        try:
            if not any(legacy_part_dir.iterdir()):
                legacy_part_dir.rmdir()
        except OSError:
            pass

    return [str(final_vrt_path)]
