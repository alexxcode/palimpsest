"""
POST /api/v1/detect  — full change-detection pipeline
GET  /api/v1/detect/{detection_id} — retrieve stored result
"""
from __future__ import annotations

import logging
import math
import time
import uuid
from datetime import datetime, timezone

import numpy as np
import pyproj
import shapely.geometry
import shapely.ops
import shapely.validation
from fastapi import APIRouter, HTTPException, Request

from app.core.config import settings
from app.schemas.request import DetectRequest
from app.schemas.response import DetectResponse
from app.services.detector import ChangeDetector
from app.services.earth_engine import get_sentinel2_pair
from app.services.postprocessor import clean_mask, compute_stats, mask_to_polygons
from app.services.preprocessor import apply_cloud_mask, normalize_pair
from app.services.storage import (
    get_detection_by_id,
    load_geotiff_from_gcs,
    save_detection,
    upload_mask_to_gcs,
)

logger = logging.getLogger(__name__)
router = APIRouter()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _aoi_area_m2(bbox: list[float]) -> float:
    """
    Approximate AOI area in m² from a WGS-84 bbox [lon_min, lat_min, lon_max, lat_max].
    Uses the equatorial approximation — < 5 % error for typical AOIs at mid-latitudes.
    """
    lon_min, lat_min, lon_max, lat_max = bbox
    lat_c = math.radians((lat_min + lat_max) / 2)
    dy    = abs(lat_max - lat_min) * 111_111.0
    dx    = abs(lon_max - lon_min) * 111_111.0 * math.cos(lat_c)
    return dx * dy


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@router.post("/detect", response_model=DetectResponse)
async def detect(request: DetectRequest, req: Request) -> DetectResponse:
    """
    Full pipeline
    -------------
    1. Retrieve Sentinel-2 pair via Earth Engine → GCS GeoTIFF.
    2. Load GeoTIFFs; split spectral bands [B2,B3,B4,B8] and SCL.
    3. Joint percentile normalisation → [0, 1] float32.
    4. Apply SCL cloud mask (zero-out clouds / shadows).
    5. Patch-based ChangeFormer inference.
    6. Morphological cleanup; vectorise to GeoJSON polygons.
    7. Upload binary mask to GCS.
    8. Insert row to BigQuery.
    9. Return DetectResponse.
    """
    detector: ChangeDetector = req.app.state.detector
    t0           = time.time()
    detection_id = uuid.uuid4().hex

    logger.info(
        "Detection %s  bbox=%s  before=%s  after=%s",
        detection_id, request.bbox, request.date_before, request.date_after,
    )

    # ------------------------------------------------------------------
    # 1. Earth Engine retrieval
    # ------------------------------------------------------------------
    try:
        pair = get_sentinel2_pair(
            bbox=request.bbox,
            date_before=request.date_before,
            date_after=request.date_after,
            max_cloud_pct=request.max_cloud_pct,
        )
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except TimeoutError as exc:
        raise HTTPException(status_code=504, detail=str(exc)) from exc
    except RuntimeError as exc:
        raise HTTPException(status_code=502, detail=str(exc)) from exc

    # ------------------------------------------------------------------
    # 2. Load GeoTIFFs from GCS
    #    Exported band order: B2=0, B3=1, B4=2, B8=3, SCL=4  (0-indexed)
    # ------------------------------------------------------------------
    before_raw, before_transform, before_crs = load_geotiff_from_gcs(pair.gcs_uri_before)
    after_raw,  after_transform,  _          = load_geotiff_from_gcs(pair.gcs_uri_after)

    # (5, H, W) → (H, W, 4) spectral  +  (H, W) SCL
    before_spec = before_raw[:4].transpose(1, 2, 0)       # B,G,R,NIR
    before_scl  = before_raw[4].astype(np.uint8)
    after_spec  = after_raw[:4].transpose(1, 2, 0)
    after_scl   = after_raw[4].astype(np.uint8)

    # ------------------------------------------------------------------
    # 3. Normalise (joint percentile) then 4. cloud-mask
    # ------------------------------------------------------------------
    b_norm, a_norm = normalize_pair(
        before_spec.astype(np.uint16),
        after_spec.astype(np.uint16),
    )
    b_masked = apply_cloud_mask(b_norm, before_scl)
    a_masked = apply_cloud_mask(a_norm, after_scl)

    # ------------------------------------------------------------------
    # 5. Change detection
    # ------------------------------------------------------------------
    mask, max_proba = detector.predict_large_debug(b_masked, a_masked)  # (H, W) uint8

    logger.info(
        "Detection %s  max_proba=%.4f  threshold=%.2f  raw_changed_px=%d",
        detection_id, max_proba, settings.confidence_threshold, int(mask.sum()),
    )

    # ------------------------------------------------------------------
    # 6. Postprocess
    # ------------------------------------------------------------------
    mask_clean = clean_mask(mask, min_area_px=settings.min_change_area_px)
    polygons   = mask_to_polygons(mask_clean, before_transform)
    stats      = compute_stats(polygons, _aoi_area_m2(request.bbox))

    # ------------------------------------------------------------------
    # 7. Persist mask to GCS
    # ------------------------------------------------------------------
    mask_uri = upload_mask_to_gcs(mask_clean, before_transform, before_crs, detection_id)

    # ------------------------------------------------------------------
    # 8. BigQuery — union all polygons → single WKT geometry (WGS-84)
    # ------------------------------------------------------------------
    # mask_to_polygons returns coordinates in the image's native CRS
    # (typically a UTM zone, in metres).  BigQuery GEOGRAPHY requires
    # WGS-84 (EPSG:4326) degrees, so we reproject before building the WKT.
    if polygons:
        shapes = [shapely.geometry.shape(f["geometry"]) for f in polygons]
        union  = shapely.ops.unary_union(shapes)
        try:
            src_crs  = pyproj.CRS(before_crs.to_epsg() if hasattr(before_crs, "to_epsg")
                                  else str(before_crs))
            wgs84    = pyproj.CRS("EPSG:4326")
            if src_crs != wgs84:
                project = pyproj.Transformer.from_crs(
                    src_crs, wgs84, always_xy=True
                ).transform
                union = shapely.ops.transform(project, union)
        except Exception as _reproj_err:
            logger.warning("Geometry reprojection failed (%s) — storing as-is", _reproj_err)
        # BigQuery GEOGRAPHY requires CCW exterior rings (RFC 7946).
        # shapely.ops.orient ensures the correct winding order.
        union = shapely.ops.orient(union, sign=1.0)
        wkt = union.wkt
    else:
        wkt = None

    save_detection(detection_id, {
        "detection_id":    detection_id,
        "area_name":       request.area_name,
        "date_before":     pair.date_before_actual.isoformat(),
        "date_after":      pair.date_after_actual.isoformat(),
        "change_area_m2":  stats["change_area_m2"],
        "change_pct":      stats["change_pct"],
        "geometry":        wkt,
        "confidence_score": None,
        "gcs_uri_before":  pair.gcs_uri_before,
        "gcs_uri_after":   pair.gcs_uri_after,
        "gcs_uri_mask":    mask_uri,
        "model_version":   ChangeDetector.MODEL_VERSION,
        "processed_at":    datetime.now(timezone.utc).isoformat(),
    })

    processing_time_s = round(time.time() - t0, 2)
    logger.info(
        "Detection %s done in %.1fs  area=%.0fm²  polygons=%d",
        detection_id, processing_time_s,
        stats["change_area_m2"], stats["polygon_count"],
    )

    return DetectResponse(
        detection_id=detection_id,
        date_before_actual=pair.date_before_actual,
        date_after_actual=pair.date_after_actual,
        change_area_m2=stats["change_area_m2"],
        change_pct=stats["change_pct"],
        polygons=polygons,
        gcs_uri_mask=mask_uri,
        model_version=ChangeDetector.MODEL_VERSION,
        processing_time_s=processing_time_s,
    )


@router.get("/detect/{detection_id}", response_model=DetectResponse)
async def get_detection(detection_id: str) -> DetectResponse:
    """Retrieve a stored detection result from BigQuery."""
    row = get_detection_by_id(detection_id)
    if row is None:
        raise HTTPException(status_code=404, detail=f"Detection '{detection_id}' not found")

    return DetectResponse(
        detection_id=row["detection_id"],
        date_before_actual=row["date_before"],
        date_after_actual=row["date_after"],
        change_area_m2=float(row.get("change_area_m2") or 0.0),
        change_pct=float(row.get("change_pct") or 0.0),
        polygons=[],          # geometry stored as WKT in BQ; not re-expanded here
        gcs_uri_mask=row.get("gcs_uri_mask") or "",
        model_version=row.get("model_version") or "",
        processing_time_s=0.0,
    )
