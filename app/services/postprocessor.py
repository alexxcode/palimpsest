"""
Postprocessor: morphological cleanup, mask vectorisation, and statistics.

All functions operate on numpy arrays and rasterio affine transforms.
Assumes a projected CRS (e.g. UTM) so that polygon areas are in m².
"""
from __future__ import annotations

import numpy as np
import rasterio.features
import scipy.ndimage as ndi
import shapely.geometry


# ---------------------------------------------------------------------------
# Morphological cleanup
# ---------------------------------------------------------------------------

def clean_mask(mask: np.ndarray, min_area_px: int = 50) -> np.ndarray:
    """
    Morphological cleanup of a binary change mask.

    Steps
    -----
    1. Binary closing (4-connectivity, 2 iterations) fills small holes.
    2. Remove connected components smaller than min_area_px pixels.

    Parameters
    ----------
    mask        : (H, W) uint8 or bool — binary change mask.
    min_area_px : minimum component size to keep (default 50 pixels).

    Returns
    -------
    (H, W) uint8 — cleaned binary mask.
    """
    struct = ndi.generate_binary_structure(2, 1)   # 4-connectivity cross
    closed = ndi.binary_closing(mask.astype(bool), structure=struct, iterations=2)

    labeled, _ = ndi.label(closed)
    sizes = np.bincount(labeled.ravel())           # size of each component
    keep  = sizes >= min_area_px
    keep[0] = False                                # never keep background label

    return keep[labeled].astype(np.uint8)


# ---------------------------------------------------------------------------
# Vectorisation
# ---------------------------------------------------------------------------

def mask_to_polygons(mask: np.ndarray, transform) -> list[dict]:
    """
    Convert a binary mask to a list of GeoJSON Feature dicts.

    Uses rasterio.features.shapes() to trace connected regions.
    Each feature has:
      "type": "Feature"
      "geometry": GeoJSON Polygon / MultiPolygon
      "properties": {"area_m2": float}

    Parameters
    ----------
    mask      : (H, W) uint8 binary mask.
    transform : affine.Affine — pixel-to-CRS mapping (from rasterio.open().transform).
                Must correspond to a metric CRS for area_m2 to be meaningful.

    Returns
    -------
    List of GeoJSON Feature dicts (empty list if mask is all-zero).
    """
    if int(mask.max()) == 0:
        return []

    features = []
    for geom_dict, value in rasterio.features.shapes(
        mask.astype(np.uint8),
        mask=(mask > 0),
        transform=transform,
    ):
        if value == 0:
            continue
        poly = shapely.geometry.shape(geom_dict)
        features.append({
            "type":       "Feature",
            "geometry":   geom_dict,
            "properties": {"area_m2": round(poly.area, 2)},
        })

    return features


# ---------------------------------------------------------------------------
# Statistics
# ---------------------------------------------------------------------------

def compute_stats(polygons: list[dict], aoi_area_m2: float) -> dict:
    """
    Summarise change detection results.

    Parameters
    ----------
    polygons    : list of Feature dicts from mask_to_polygons().
    aoi_area_m2 : total area of the analysed region in m².

    Returns
    -------
    Dict with keys: change_area_m2, change_pct, polygon_count.
    """
    total_m2 = sum(f["properties"]["area_m2"] for f in polygons)
    pct      = (total_m2 / aoi_area_m2 * 100.0) if aoi_area_m2 > 0 else 0.0
    return {
        "change_area_m2": round(total_m2, 2),
        "change_pct":     round(pct, 4),
        "polygon_count":  len(polygons),
    }
