"""
Earth Engine Sentinel-2 retrieval service.

Queries S2_SR_HARMONIZED for a ±search_window_days window around each
target date, picks the least-cloudy qualifying scene, exports five bands
(B2 B3 B4 B8 SCL) to GCS as a Cloud-Optimised GeoTIFF, and polls the
export task with exponential back-off until completion or timeout.
"""
from __future__ import annotations

import logging
import time
import uuid
from dataclasses import dataclass
from datetime import date, timedelta

import ee

from app.core.config import settings

logger = logging.getLogger(__name__)

# Bands exported: Blue, Green, Red, NIR, Scene Classification Layer
_S2_BANDS = ["B2", "B3", "B4", "B8", "SCL"]

_EXPORT_SCALE_M   = 10    # native 10 m/px for 10 m bands
_EXPORT_TIMEOUT_S = 600   # 10 minutes maximum wait
_POLL_INTERVAL_S  = 5     # initial poll interval
_POLL_MAX_S       = 60    # back-off ceiling

_ee_initialised = False


# ---------------------------------------------------------------------------
# Earth Engine initialisation
# ---------------------------------------------------------------------------

def _init_ee() -> None:
    """Initialise EE once per process using application-default credentials.

    Uses google.auth.default() explicitly so this works in Cloud Run (service
    account ADC) without requiring an interactive ``gcloud auth`` flow or the
    gcloud binary being present in the container image.
    """
    global _ee_initialised
    if _ee_initialised:
        return
    import google.auth  # lazy import – always available via google-auth dep

    _EE_SCOPES = [
        "https://www.googleapis.com/auth/earthengine",
        "https://www.googleapis.com/auth/cloud-platform",
    ]
    credentials, _ = google.auth.default(scopes=_EE_SCOPES)
    ee.Initialize(credentials=credentials, project=settings.ee_project)
    _ee_initialised = True
    logger.info("Earth Engine initialised  project=%s", settings.ee_project)


# ---------------------------------------------------------------------------
# Public dataclass
# ---------------------------------------------------------------------------

@dataclass
class ImagePair:
    gcs_uri_before:     str
    gcs_uri_after:      str
    date_before_actual: date
    date_after_actual:  date
    cloud_cover_before: float
    cloud_cover_after:  float


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _best_image(
    bbox_geom,
    center_date: date,
    window_days: int,
    max_cloud_pct: float,
):
    """Return the least-cloudy S2 image within ±window_days of center_date."""
    start = (center_date - timedelta(days=window_days)).isoformat()
    end   = (center_date + timedelta(days=window_days + 1)).isoformat()

    col = (
        ee.ImageCollection("COPERNICUS/S2_SR_HARMONIZED")
        .filterBounds(bbox_geom)
        .filterDate(start, end)
        .filter(ee.Filter.lte("CLOUDY_PIXEL_PERCENTAGE", max_cloud_pct))
        .sort("CLOUDY_PIXEL_PERCENTAGE")
    )

    count = col.size().getInfo()
    if count == 0:
        raise ValueError(
            f"No S2 image near {center_date} with cloud ≤ {max_cloud_pct}% "
            f"(±{window_days} days)"
        )
    return col.first()


def _export_to_gcs(image, bbox_geom, export_name: str) -> str:
    """
    Submit an EE export task and poll until COMPLETED.
    Returns the gs:// URI of the written GeoTIFF.
    Raises TimeoutError / RuntimeError on failure.
    """
    blob_prefix = f"{settings.gcs_raw_prefix}{export_name}"

    # Cast to uniform uint16 — spectral bands are uint16, SCL is byte;
    # mixing types causes EE to refuse the export.
    task = ee.batch.Export.image.toCloudStorage(
        image=image.select(_S2_BANDS).toUint16(),
        description=export_name[:100],
        bucket=settings.gcs_bucket_name,
        fileNamePrefix=blob_prefix,
        region=bbox_geom,
        scale=_EXPORT_SCALE_M,
        fileFormat="GeoTIFF",
        formatOptions={"cloudOptimized": True},
        maxPixels=int(1e10),
    )
    task.start()
    logger.info(
        "EE export started  task=%s  → gs://%s/%s.tif",
        task.id, settings.gcs_bucket_name, blob_prefix,
    )

    interval = _POLL_INTERVAL_S
    deadline = time.time() + _EXPORT_TIMEOUT_S

    while time.time() < deadline:
        time.sleep(interval)
        status = task.status()
        state  = status["state"]
        logger.debug("EE task %s: %s", task.id, state)

        if state == "COMPLETED":
            uri = f"gs://{settings.gcs_bucket_name}/{blob_prefix}.tif"
            logger.info("Export complete: %s", uri)
            return uri

        if state in ("FAILED", "CANCELLED"):
            raise RuntimeError(
                f"EE export {task.id} ended with {state}: "
                f"{status.get('error_message', 'no detail')}"
            )

        interval = min(interval * 2, _POLL_MAX_S)

    task.cancel()
    raise TimeoutError(
        f"EE export {task.id} exceeded {_EXPORT_TIMEOUT_S}s — task cancelled"
    )


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def get_sentinel2_pair(
    bbox: list[float],
    date_before: date,
    date_after:  date,
    max_cloud_pct: float = 10.0,
    search_window_days: int = 30,
) -> ImagePair:
    """
    Retrieve and export a Sentinel-2 before/after pair to GCS.

    Steps
    -----
    1. Query S2_SR_HARMONIZED for each date window.
    2. Filter by cloud cover; pick least-cloudy scene.
    3. Export [B2, B3, B4, B8, SCL] to GCS as Cloud-Optimised GeoTIFF.
    4. Poll export with exponential back-off (max 10 min each).
    5. Return ImagePair with GCS URIs and actual acquisition metadata.

    Parameters
    ----------
    bbox : [lon_min, lat_min, lon_max, lat_max]  (WGS-84)

    Raises
    ------
    ValueError   – no scene found within window + cloud threshold
    TimeoutError – export task exceeded 10-minute deadline
    RuntimeError – EE export failed or was cancelled
    """
    _init_ee()

    lon_min, lat_min, lon_max, lat_max = bbox
    geom = ee.Geometry.Rectangle([lon_min, lat_min, lon_max, lat_max])

    img_b = _best_image(geom, date_before, search_window_days, max_cloud_pct)
    img_a = _best_image(geom, date_after,  search_window_days, max_cloud_pct)

    # Fetch acquisition date and cloud cover before starting long exports
    props_b = img_b.toDictionary(["system:time_start", "CLOUDY_PIXEL_PERCENTAGE"]).getInfo()
    props_a = img_a.toDictionary(["system:time_start", "CLOUDY_PIXEL_PERCENTAGE"]).getInfo()

    dt_before = date.fromtimestamp(props_b["system:time_start"] / 1000)
    dt_after  = date.fromtimestamp(props_a["system:time_start"] / 1000)
    cc_before = float(props_b["CLOUDY_PIXEL_PERCENTAGE"])
    cc_after  = float(props_a["CLOUDY_PIXEL_PERCENTAGE"])

    logger.info(
        "Exporting before=%s (cloud=%.1f%%) and after=%s (cloud=%.1f%%)",
        dt_before, cc_before, dt_after, cc_after,
    )

    run_id = uuid.uuid4().hex[:8]
    uri_b  = _export_to_gcs(img_b, geom, f"before_{run_id}")
    uri_a  = _export_to_gcs(img_a, geom, f"after_{run_id}")

    return ImagePair(
        gcs_uri_before=uri_b,
        gcs_uri_after=uri_a,
        date_before_actual=dt_before,
        date_after_actual=dt_after,
        cloud_cover_before=cc_before,
        cloud_cover_after=cc_after,
    )
