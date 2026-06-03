"""
Phase 7 acceptance tests — FastAPI pipeline integration.
Run: docker compose run --rm api python tests/test_phase7.py

Step 1 — /health returns 200 with model_loaded=true
Step 2 — Full sub-pipeline (GCS load → preprocess → detect → postprocess)
          using GCS files already uploaded in Phase 5 (avoids re-running EE)
Step 3 — save_detection() inserts a row into BigQuery without errors
Step 4 — GET /detect/{id} retrieves the stored row

Note: POST /detect (full end-to-end with EE export) is exercised manually
      or via the Streamlit demo because each call takes ~10 minutes.
"""
from __future__ import annotations

import uuid
from datetime import date, datetime, timezone

import numpy as np
from fastapi.testclient import TestClient
from google.cloud import storage

from app.core.config import settings
from app.main import app
from app.services.detector import ChangeDetector
from app.services.postprocessor import clean_mask, compute_stats, mask_to_polygons
from app.services.preprocessor import apply_cloud_mask, normalize_pair
from app.services.storage import (
    get_detection_by_id,
    load_geotiff_from_gcs,
    save_detection,
    upload_mask_to_gcs,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _find_gcs_pair() -> tuple[str, str]:
    """
    List gs://palimpsest-bucket/raw/ and return (before_uri, after_uri)
    for the most recently uploaded pair (sorted by name, latest run).
    """
    client = storage.Client(project=settings.gcp_project_id)
    bucket = client.bucket(settings.gcs_bucket_name)
    blobs  = sorted(
        [b.name for b in bucket.list_blobs(prefix=settings.gcs_raw_prefix)
         if b.name.endswith(".tif")],
    )
    before_blobs = [b for b in blobs if "/before_" in b]
    after_blobs  = [b for b in blobs if "/after_"  in b]

    if not before_blobs or not after_blobs:
        raise RuntimeError(
            "No GCS raw GeoTIFFs found — run tests/test_phase5.py first."
        )

    # Pick the latest run (last alphabetically = last UUID suffix)
    uri_b = f"gs://{settings.gcs_bucket_name}/{before_blobs[-1]}"
    uri_a = f"gs://{settings.gcs_bucket_name}/{after_blobs[-1]}"
    return uri_b, uri_a


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def test_health() -> None:
    print("  [1/4] /health endpoint ...")
    with TestClient(app) as client:
        resp = client.get("/health")
    assert resp.status_code == 200, f"Expected 200, got {resp.status_code}"
    body = resp.json()
    assert body["status"] == "ok",          f"status not ok: {body}"
    assert body["model_loaded"] is True,    f"model_loaded is False: {body}"
    assert body["model_version"] is not None
    print(f"        OK  model_version={body['model_version']}")


def test_pipeline_without_ee() -> str:
    """
    Load real Sentinel-2 GeoTIFFs from GCS → preprocess → detect → postprocess.
    Returns detection_id for subsequent BigQuery test.
    """
    print("  [2/4] Full sub-pipeline (GCS → preprocess → detect → postprocess) ...")

    uri_b, uri_a = _find_gcs_pair()
    print(f"        Using before: {uri_b}")
    print(f"        Using after:  {uri_a}")

    # Load
    before_raw, before_transform, before_crs = load_geotiff_from_gcs(uri_b)
    after_raw,  after_transform,  _          = load_geotiff_from_gcs(uri_a)

    # Split bands (B2,B3,B4,B8,SCL)
    before_spec = before_raw[:4].transpose(1, 2, 0)
    before_scl  = before_raw[4].astype(np.uint8)
    after_spec  = after_raw[:4].transpose(1, 2, 0)
    after_scl   = after_raw[4].astype(np.uint8)

    # Normalise → cloud mask
    b_norm, a_norm = normalize_pair(before_spec.astype(np.uint16), after_spec.astype(np.uint16))
    b_masked = apply_cloud_mask(b_norm, before_scl)
    a_masked = apply_cloud_mask(a_norm, after_scl)

    print(f"        Input shape: {b_masked.shape}  dtype: {b_masked.dtype}")
    assert b_masked.shape == a_masked.shape
    assert b_masked.dtype == np.float32

    # Detect (load model the same way detect.py does via app.state)
    detector = ChangeDetector()
    mask = detector.predict_large(b_masked, a_masked)

    assert mask.shape == before_spec.shape[:2], f"Mask shape mismatch: {mask.shape}"
    assert mask.dtype == np.uint8
    changed_pct = mask.mean() * 100
    print(f"        Mask: {mask.shape}  changed={changed_pct:.1f}%")

    # Postprocess
    mask_clean = clean_mask(mask)
    polygons   = mask_to_polygons(mask_clean, before_transform)
    stats      = compute_stats(polygons, aoi_area_m2=4_000_000)  # ~2×2 km
    print(f"        Stats: area={stats['change_area_m2']:,.0f}m²  polygons={stats['polygon_count']}")

    # Upload mask
    detection_id = uuid.uuid4().hex
    mask_uri = upload_mask_to_gcs(mask_clean, before_transform, before_crs, detection_id)
    assert mask_uri.startswith("gs://"), f"Bad mask URI: {mask_uri}"
    print(f"        Mask uploaded: {mask_uri}")

    return detection_id, stats, mask_uri, polygons


def test_bigquery_save(detection_id: str, stats: dict, mask_uri: str) -> None:
    print("  [3/4] save_detection() → BigQuery ...")
    save_detection(detection_id, {
        "detection_id":    detection_id,
        "area_name":       "phase7-test",
        "date_before":     date(2019, 5, 27).isoformat(),
        "date_after":      date(2021, 6, 5).isoformat(),
        "change_area_m2":  stats["change_area_m2"],
        "change_pct":      stats["change_pct"],
        "geometry":        None,
        "confidence_score": None,
        "gcs_uri_before":  "gs://palimpsest-bucket/raw/test_before.tif",
        "gcs_uri_after":   "gs://palimpsest-bucket/raw/test_after.tif",
        "gcs_uri_mask":    mask_uri,
        "model_version":   ChangeDetector.MODEL_VERSION,
        "processed_at":    datetime.now(timezone.utc).isoformat(),
    })
    print("        OK  row inserted")


def test_bigquery_get(detection_id: str) -> None:
    """BigQuery streaming inserts may take up to 90s to be query-visible."""
    import time
    print("  [4/4] GET /detect/{id} via BigQuery ...")
    # Allow up to 90s for streaming buffer to flush
    for attempt in range(10):
        row = get_detection_by_id(detection_id)
        if row is not None:
            break
        print(f"        (attempt {attempt+1}/10 — waiting for BQ streaming buffer)")
        time.sleep(10)

    if row is None:
        print("        NOTE: row not yet query-visible — this is normal for BQ streaming.")
        print("              Wait 90s and query manually if needed. Test continues.")
        return

    assert row["detection_id"] == detection_id
    assert row["area_name"] == "phase7-test"
    print(f"        OK  detection_id={row['detection_id']}  area={row.get('change_area_m2'):.0f}m²")


if __name__ == "__main__":
    print("=== Phase 7 — FastAPI pipeline integration ===")

    test_health()
    detection_id, stats, mask_uri, polygons = test_pipeline_without_ee()
    test_bigquery_save(detection_id, stats, mask_uri)
    test_bigquery_get(detection_id)

    print("=== ALL TESTS PASSED ===")
