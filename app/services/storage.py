"""
Storage utilities.

  GCS  — load_geotiff_from_gcs(), upload_mask_to_gcs()
  BQ   — save_detection()
"""
from __future__ import annotations

import logging
import tempfile
from pathlib import Path

import numpy as np
import rasterio
from google.cloud import bigquery, storage

from app.core.config import settings

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _parse_gcs_uri(uri: str) -> tuple[str, str]:
    """Parse 'gs://bucket/path/blob' → ('bucket', 'path/blob')."""
    if not uri.startswith("gs://"):
        raise ValueError(f"Expected gs:// URI, got: {uri}")
    rest = uri[5:]
    bucket, _, blob = rest.partition("/")
    return bucket, blob


# ---------------------------------------------------------------------------
# GCS — read
# ---------------------------------------------------------------------------

def load_geotiff_from_gcs(gcs_uri: str) -> tuple[np.ndarray, object, object]:
    """
    Download a GeoTIFF from GCS and open it with rasterio.

    Returns
    -------
    data      : (bands, H, W) float32 — all bands in file order.
    transform : affine.Affine — pixel-to-CRS mapping.
    crs       : rasterio.crs.CRS.
    """
    bucket_name, blob_name = _parse_gcs_uri(gcs_uri)

    client = storage.Client(project=settings.gcp_project_id)
    bucket = client.bucket(bucket_name)
    blob   = bucket.blob(blob_name)

    with tempfile.NamedTemporaryFile(suffix=".tif", delete=False) as tmp:
        tmp_path = tmp.name

    try:
        logger.info("Downloading %s ...", gcs_uri)
        blob.download_to_filename(tmp_path)

        with rasterio.open(tmp_path) as ds:
            data      = ds.read().astype(np.float32)   # (bands, H, W)
            transform = ds.transform
            crs       = ds.crs
    finally:
        Path(tmp_path).unlink(missing_ok=True)

    logger.info("Loaded GeoTIFF: shape=%s  crs=%s", data.shape, crs.to_epsg() if crs else None)
    return data, transform, crs


# ---------------------------------------------------------------------------
# GCS — write
# ---------------------------------------------------------------------------

def upload_mask_to_gcs(
    mask:      np.ndarray,
    transform,
    crs,
    detection_id: str,
) -> str:
    """
    Write a binary change mask (H, W) uint8 as a single-band GeoTIFF to GCS.

    Returns the gs:// URI of the uploaded file.
    """
    blob_name = f"{settings.gcs_results_prefix}{detection_id}_mask.tif"
    uri       = f"gs://{settings.gcs_bucket_name}/{blob_name}"

    with tempfile.NamedTemporaryFile(suffix=".tif", delete=False) as tmp:
        tmp_path = tmp.name

    try:
        with rasterio.open(
            tmp_path, "w",
            driver="GTiff",
            height=mask.shape[0],
            width=mask.shape[1],
            count=1,
            dtype=np.uint8,
            crs=crs,
            transform=transform,
            compress="lzw",
        ) as ds:
            ds.write(mask[np.newaxis, ...])   # (1, H, W)

        client = storage.Client(project=settings.gcp_project_id)
        bucket = client.bucket(settings.gcs_bucket_name)
        blob   = bucket.blob(blob_name)
        blob.upload_from_filename(tmp_path, content_type="image/tiff")
    finally:
        Path(tmp_path).unlink(missing_ok=True)

    logger.info("Mask uploaded: %s", uri)
    return uri


# ---------------------------------------------------------------------------
# BigQuery
# ---------------------------------------------------------------------------

def save_detection(detection_id: str, payload: dict) -> None:
    """
    Insert one row into the BigQuery change_detections table.

    Expected payload keys (matching infra/bq_schema.json):
        detection_id, area_name, date_before, date_after,
        change_area_m2, change_pct,
        geometry  (WKT string for GEOGRAPHY column, or None),
        confidence_score, gcs_uri_before, gcs_uri_after,
        gcs_uri_mask, model_version, processed_at (ISO-8601 timestamp).

    Raises RuntimeError if BigQuery reports insertion errors.
    """
    client    = bigquery.Client(project=settings.gcp_project_id)
    table_ref = (
        f"{settings.gcp_project_id}"
        f".{settings.bq_dataset}"
        f".{settings.bq_table}"
    )

    errors = client.insert_rows_json(table_ref, [payload])
    if errors:
        raise RuntimeError(f"BigQuery insert errors for {detection_id}: {errors}")

    logger.info("Detection saved to BigQuery: %s", detection_id)


def get_detection_by_id(detection_id: str) -> dict | None:
    """
    Retrieve a stored detection row from BigQuery.
    Returns the first matching row as a dict, or None if not found.
    """
    client = bigquery.Client(project=settings.gcp_project_id)
    query  = (
        f"SELECT * FROM `{settings.gcp_project_id}"
        f".{settings.bq_dataset}.{settings.bq_table}` "
        f"WHERE detection_id = @detection_id LIMIT 1"
    )
    job_config = bigquery.QueryJobConfig(
        query_parameters=[
            bigquery.ScalarQueryParameter("detection_id", "STRING", detection_id)
        ]
    )
    rows = list(client.query(query, job_config=job_config).result())
    return dict(rows[0]) if rows else None
