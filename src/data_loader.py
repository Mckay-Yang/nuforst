import hashlib
import json
import rasterio
import numpy as np
from pathlib import Path
from datetime import datetime
from typing import Dict, Tuple, Optional, Union, List
import glob
import re
from collections import defaultdict
import subprocess
import os

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
        if cache_dir is None:
            # Default to data/cache relative to the project root (parent of src)
            self.cache_dir = Path(__file__).resolve().parent.parent / "data" / "cache"
        else:
            self.cache_dir = Path(cache_dir)
            
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

        combined_payload = "|".join(payloads)
        return hashlib.md5(combined_payload.encode("utf-8")).hexdigest()[:12]

    def _cache_path(self) -> Path:
        if self.cache_dir is not None and not self.cache_dir.exists():
            self.cache_dir.mkdir(parents=True, exist_ok=True)

        # Use the first file's stem as a base name
        base_stem = self.tif_paths[0].stem
        # Optional: remove trailing chunk like '_part1' or '_2015' from the stem for cache name clarity
        parts = base_stem.split('_')
        if len(parts) > 1 and (parts[-1].startswith('part') or (parts[-1].isdigit() and len(parts[-1]) == 4)):
            base_stem = '_'.join(parts[:-1])

        if self.cache_dir is None:
            return self.tif_paths[0].parent / f"{base_stem}_{self._file_signature()}.npz"
        return self.cache_dir / f"{base_stem}_{self._file_signature()}.npz"

    def _parse_band_timestamp(self, name: str) -> str:
        """Best-effort parse; fall back to the raw band name."""
        tokens = [name.strip()]

        # Special handling for HLS '2_T46RET_20150114T041054' or '1_T01FBE_20190216T214619_BLUE'
        import re
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

    def _read_tif(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        all_arrs = []
        all_band_names = []
        all_timestamps = []

        first_meta = None

        for path in self.tif_paths:
            with rasterio.open(path) as src:
                # Memory safety: restrict to top-left 512x512 to prevent OOM on 19GB cubes in Colab
                read_w = min(src.width, 512)
                read_h = min(src.height, 512)
                window = rasterio.windows.Window(0, 0, read_w, read_h)
                
                arr = src.read(window=window, masked=True).astype(np.float32)
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
            print(f"[System] Loading from cache: {cache_path}")
            return self._load_npz(cache_path)
        print(f"[System] Reading {len(self.tif_paths)} TIFF(s) into memory...")
        cube, timestamps, band_names = self._read_tif()
        self._save_npz(cache_path, cube, timestamps, band_names)
        return {"cube": cube, "timestamps": timestamps, "band_names": band_names, **self.meta, "cache_path": str(cache_path)}


def find_image_chunks(data_dir: str, lon: float, lat: float, band: str) -> List[List[str]]:
    """
    Finds ALL spatial tiles and temporal parts for a given location and band.
    Mosaics the spatial tiles of the SAME temporal part into a VRT.
    Returns a list containing ONE inner list, which holds the ordered VRT paths (e.g., [part1.vrt, part2.vrt, ...]).
    """
    pattern = str(Path(data_dir) / f"*_{band}_lon{lon}_lat{lat}*.tif")
    files = glob.glob(pattern)
    if not files:
        return []
    
    # Group files by their temporal part
    parts = defaultdict(list)
    for f in files:
        match = re.search(r"_(part\d+)(?:-\d{10}-\d{10})?\.tif$", Path(f).name)
        part_name = match.group(1) if match else "full"
        parts[part_name].append(f)
        
    vrt_dir = Path(data_dir).parent / "cache" / "vrts"
    vrt_dir.mkdir(parents=True, exist_ok=True)
    
    vrt_paths = []
    for part_name in sorted(parts.keys()):
        part_files = parts[part_name]
        vrt_filename = f"{band}_lon{lon}_lat{lat}_{part_name}.vrt"
        vrt_path = vrt_dir / vrt_filename
        if not vrt_path.exists():
            subprocess.run(["gdalbuildvrt", "-q", str(vrt_path)] + part_files, check=True)
        vrt_paths.append(str(vrt_path))
             
    # Return as a single chunk containing the temporal sequence of VRTs
    return [vrt_paths]

