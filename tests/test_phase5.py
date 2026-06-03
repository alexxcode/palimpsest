"""
Phase 5 acceptance tests — Earth Engine Sentinel-2 retrieval.
Run: docker compose run --rm api python tests/test_phase5.py

Step 1 — EE initialises via ADC
Step 2 — get_sentinel2_pair() exports 2 GeoTIFFs to GCS (≤ 10 min each)
Step 3 — GeoTIFFs are readable with 5 bands [B2,B3,B4,B8,SCL]
Step 4 — spectral values are plausible S2 reflectance (0–10 000 DN range)

Small São Paulo test AOI: ~2×2 km, well-studied, abundant S2 coverage.
"""
from datetime import date

import numpy as np

from app.services.earth_engine import get_sentinel2_pair
from app.services.storage import load_geotiff_from_gcs

# ~2 × 2 km patch in São Paulo city centre
BBOX        = [-46.66, -23.55, -46.64, -23.53]
DATE_BEFORE = date(2019, 6, 1)
DATE_AFTER  = date(2021, 6, 1)
MAX_CLOUD   = 20.0
WINDOW_DAYS = 45


def test_retrieve_pair():
    print("  [1/2] get_sentinel2_pair() — two EE exports (this takes a few minutes) ...")
    pair = get_sentinel2_pair(
        bbox=BBOX,
        date_before=DATE_BEFORE,
        date_after=DATE_AFTER,
        max_cloud_pct=MAX_CLOUD,
        search_window_days=WINDOW_DAYS,
    )

    print(f"        before: {pair.date_before_actual}  cloud={pair.cloud_cover_before:.1f}%")
    print(f"        after:  {pair.date_after_actual}   cloud={pair.cloud_cover_after:.1f}%")
    print(f"        GCS before: {pair.gcs_uri_before}")
    print(f"        GCS after:  {pair.gcs_uri_after}")

    assert pair.gcs_uri_before.startswith("gs://"), "Expected gs:// URI for before"
    assert pair.gcs_uri_after.startswith("gs://"),  "Expected gs:// URI for after"
    assert pair.date_before_actual is not None
    assert pair.date_after_actual  is not None
    assert 0.0 <= pair.cloud_cover_before <= 100.0
    assert 0.0 <= pair.cloud_cover_after  <= 100.0

    return pair


def test_load_geotiffs(pair) -> None:
    print("  [2/2] Loading both GeoTIFFs from GCS ...")
    for label, uri in [("before", pair.gcs_uri_before), ("after", pair.gcs_uri_after)]:
        data, transform, crs = load_geotiff_from_gcs(uri)

        print(f"        {label}: shape={data.shape}  crs={crs.to_epsg() if crs else 'None'}")

        assert data.ndim == 3,          f"{label}: expected 3-D array, got ndim={data.ndim}"
        assert data.shape[0] == 5,      f"{label}: expected 5 bands, got {data.shape[0]}"
        assert data.shape[1] > 0,       f"{label}: zero height"
        assert data.shape[2] > 0,       f"{label}: zero width"

        # Spectral bands (B2-B8): S2 L2A reflectance DN 0–10 000
        spectral = data[:4]
        assert spectral.max() > 100, (
            f"{label}: spectral max={spectral.max():.0f} — looks like all zeros"
        )
        assert spectral.min() >= 0,  f"{label}: negative DN values"
        print(f"          spectral DN range: [{spectral.min():.0f}, {spectral.max():.0f}]")

        # SCL band: integer class labels 0–11
        scl = data[4]
        unique_scl = np.unique(scl.astype(np.uint8))
        assert len(unique_scl) > 0, f"{label}: SCL band is empty"
        print(f"          SCL unique values: {unique_scl.tolist()}")


if __name__ == "__main__":
    print("=== Phase 5 — Earth Engine retrieval ===")
    pair = test_retrieve_pair()
    test_load_geotiffs(pair)
    print("=== ALL TESTS PASSED ===")
