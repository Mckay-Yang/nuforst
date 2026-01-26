import hashlib
import json
from pathlib import Path
from datetime import datetime
from typing import Dict, Tuple, Optional, List
import numpy as np
import rasterio

class RSCube:
    """
    Handle spatiotemporal data cube from a multi-band GeoTIFF where each band is a timestamped slice.
    Data are cached as compressed npz to avoid repeatedly reading the large TIFF.
    """
    def __init__(self, tif_path: str, cache_dir: str = "./cache", npz_path: Optional[str] = None, force_refresh: bool = False) -> None:
        self.tif_path = Path(tif_path)
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.npz_path = Path(npz_path) if npz_path else None
        self.force_refresh = force_refresh
        self.meta: Dict[str, object] = {}

    def _file_signature(self) -> str:
        stat = self.tif_path.stat()
        payload = f"{self.tif_path.resolve()}:{stat.st_size}:{stat.st_mtime}"
        return hashlib.md5(payload.encode("utf-8")).hexdigest()[:12]

    def _cache_path(self) -> Path:
        if self.npz_path:
            return self.npz_path
        stem = self.tif_path.stem
        return self.cache_dir / f"{stem}_{self._file_signature()}.npz"

    def _parse_band_timestamp(self, name: str) -> str:
        """Best-effort parse; fall back to the raw band name."""
        tokens = [name.strip()]
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
        if not self.tif_path.exists():
            raise FileNotFoundError(f"Input image not found: {self.tif_path}")

        with rasterio.open(self.tif_path) as src:
            arr = src.read(masked=True).astype(np.float32)
            if np.ma.isMaskedArray(arr):
                arr = arr.filled(np.nan)
            band_names = list(src.descriptions)
            if not band_names or all(name is None or name == "" for name in band_names):
                band_names = [f"band_{i+1}" for i in range(src.count)]
            timestamps = [self._parse_band_timestamp(name) for name in band_names]
            self.meta = {
                "transform": list(src.transform),
                "crs_wkt": src.crs.to_wkt() if src.crs else None,
                "height": src.height,
                "width": src.width,
                "count": src.count,
            }
            return arr, np.array(timestamps, dtype="U32"), np.array(band_names, dtype="U64")

    def _save_npz(self, cache_path: Path, cube: np.ndarray, timestamps: np.ndarray, band_names: np.ndarray) -> None:
        meta = {**self.meta, "cache_path": str(cache_path), "tif_path": str(self.tif_path)}
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
        print(f"[System] Reading TIFF: {self.tif_path}")
        cube, timestamps, band_names = self._read_tif()
        self._save_npz(cache_path, cube, timestamps, band_names)
        return {"cube": cube, "timestamps": timestamps, "band_names": band_names, **self.meta, "cache_path": str(cache_path)}
