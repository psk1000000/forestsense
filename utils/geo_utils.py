"""
ForestSense — Geospatial Utilities
=====================================
Helper functions for coordinate conversion, area calculation,
GeoTIFF metadata reading, and quick visualization of raster data.
"""

import numpy as np
import rasterio
from rasterio.transform import from_bounds
from rasterio.crs import CRS
from pathlib import Path
import json
from typing import Tuple, List, Optional


# ── Coordinate & Area Helpers ─────────────────────────────────

def bbox_to_area_km2(bbox: List[float]) -> float:
    """
    Approximate area of a bounding box in km².
    bbox = [lon_min, lat_min, lon_max, lat_max]
    """
    import math
    lon_min, lat_min, lon_max, lat_max = bbox
    lat_center = (lat_min + lat_max) / 2.0
    # 1° latitude ≈ 111.32 km
    # 1° longitude ≈ 111.32 * cos(lat) km
    lat_km = (lat_max - lat_min) * 111.32
    lon_km = (lon_max - lon_min) * 111.32 * math.cos(math.radians(lat_center))
    return lat_km * lon_km


def pixels_to_hectares(pixel_count: int, resolution_m: float = 20.0) -> float:
    """Convert pixel count to area in hectares given pixel size in metres."""
    area_m2 = pixel_count * (resolution_m ** 2)
    return area_m2 / 10_000.0


def hectares_to_km2(ha: float) -> float:
    return ha / 100.0


# ── GeoTIFF I/O ───────────────────────────────────────────────

def read_geotiff_info(filepath: str) -> dict:
    """Return metadata dict for a GeoTIFF without reading pixel data."""
    with rasterio.open(filepath) as src:
        return {
            "path"      : filepath,
            "n_bands"   : src.count,
            "width"     : src.width,
            "height"    : src.height,
            "crs"       : str(src.crs),
            "transform" : list(src.transform),
            "dtype"     : str(src.dtypes[0]),
            "nodata"    : src.nodata,
            "bounds"    : {
                "left"  : src.bounds.left,
                "bottom": src.bounds.bottom,
                "right" : src.bounds.right,
                "top"   : src.bounds.top,
            },
            "res_m"     : abs(src.transform[0]),
        }


def save_geotiff(
    data      : np.ndarray,
    filepath  : str,
    reference_path: Optional[str] = None,
    crs       : str = "EPSG:4326",
    transform = None,
):
    """
    Save a numpy array (C, H, W) or (H, W) as a GeoTIFF.
    Optionally copies geo-referencing from a reference file.
    """
    if data.ndim == 2:
        data = data[np.newaxis, ...]   # Add channel dim

    C, H, W = data.shape
    profile = {
        "driver"   : "GTiff",
        "dtype"    : str(data.dtype),
        "width"    : W,
        "height"   : H,
        "count"    : C,
        "crs"      : CRS.from_string(crs),
        "compress" : "lzw",
        "tiled"    : True,
        "blockxsize": 256,
        "blockysize": 256,
    }

    if reference_path and Path(reference_path).exists():
        with rasterio.open(reference_path) as ref:
            profile["crs"]       = ref.crs
            profile["transform"] = ref.transform
    elif transform is not None:
        profile["transform"] = transform

    Path(filepath).parent.mkdir(parents=True, exist_ok=True)
    with rasterio.open(filepath, "w", **profile) as dst:
        dst.write(data)


# ── Quick Visualisation ───────────────────────────────────────

def quick_preview(stack: np.ndarray, save_path: Optional[str] = None):
    """
    Quick 4-panel preview of a (12, H, W) stack:
    Panel 1: RGB True Color
    Panel 2: SAR False Color (VV, VH, ratio)
    Panel 3: NDVI
    Panel 4: NDWI
    """
    import matplotlib.pyplot as plt

    def _norm(x):
        mn, mx = x.min(), x.max()
        return (x - mn) / (mx - mn + 1e-8)

    fig, axes = plt.subplots(1, 4, figsize=(16, 4), facecolor="#1a1a2e")
    titles     = ["RGB (True Color)", "SAR False Color", "NDVI", "NDWI"]

    # RGB
    rgb = np.stack([_norm(stack[5]), _norm(stack[4]), _norm(stack[3])], axis=-1)
    axes[0].imshow(np.clip(rgb, 0, 1))
    axes[0].set_title(titles[0], color="white")
    axes[0].axis("off")

    # SAR FCC
    sar = np.stack([_norm(stack[0]), _norm(stack[1]), _norm(stack[2])], axis=-1)
    axes[1].imshow(np.clip(sar, 0, 1))
    axes[1].set_title(titles[1], color="white")
    axes[1].axis("off")

    # NDVI
    ndvi = stack[9] * 2.0 - 1.0    # rescale back to [-1, 1]
    im = axes[2].imshow(ndvi, cmap="RdYlGn", vmin=-1, vmax=1)
    axes[2].set_title(titles[2], color="white")
    axes[2].axis("off")
    plt.colorbar(im, ax=axes[2], fraction=0.046, pad=0.04).ax.yaxis.set_tick_params(color="white")

    # NDWI
    ndwi = stack[11] * 2.0 - 1.0
    im2 = axes[3].imshow(ndwi, cmap="Blues", vmin=-1, vmax=1)
    axes[3].set_title(titles[3], color="white")
    axes[3].axis("off")
    plt.colorbar(im2, ax=axes[3], fraction=0.046, pad=0.04).ax.yaxis.set_tick_params(color="white")

    for ax in axes:
        ax.title.set_color("white")

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight", facecolor="#1a1a2e")
        print(f"Preview saved → {save_path}")
    else:
        plt.show()
    plt.close()


# ── Metadata I/O ─────────────────────────────────────────────

def load_metadata(path: str) -> dict:
    with open(path) as f:
        return json.load(f)


def get_region_stats(processed_dir: str, region_id: str) -> dict:
    """Load per-region stats JSON."""
    stats_path = Path(processed_dir) / "patches" / region_id / "stats.json"
    if not stats_path.exists():
        raise FileNotFoundError(f"Stats file not found: {stats_path}")
    return load_metadata(str(stats_path))


def get_global_stats(processed_dir: str) -> dict:
    """Load global preprocessing stats."""
    return load_metadata(str(Path(processed_dir) / "global_stats.json"))
