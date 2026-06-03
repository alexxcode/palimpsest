"""
Phase 6 acceptance tests — postprocessor + storage utilities (local only).
Run: docker compose run --rm api python tests/test_phase6.py

Step 1 — clean_mask: small noise removed, large region kept
Step 2 — mask_to_polygons: GeoJSON features with area_m2
Step 3 — compute_stats: totals and percentage
Step 4 — full postprocess chain on a synthetic 1024×1024 mask
"""

import numpy as np
import rasterio.transform

from app.services.postprocessor import clean_mask, compute_stats, mask_to_polygons


# 10 m/px UTM-like transform covering a 1024×1024 grid
_TRANSFORM = rasterio.transform.from_bounds(0, 0, 10_240, 10_240, 1024, 1024)
_AOI_M2    = 10_240 ** 2    # 104.8 km²


def test_clean_mask() -> None:
    print("  [1/4] clean_mask ...")
    mask = np.zeros((256, 256), dtype=np.uint8)
    mask[10:100, 10:100] = 1   # 90×90 = 8 100 px → survives
    mask[150:153, 150:153] = 1 # 3×3   = 9 px    → removed

    cleaned = clean_mask(mask, min_area_px=50)

    assert cleaned[50, 50]   == 1, "Large region must survive"
    assert cleaned[151, 151] == 0, "Small region must be removed"
    assert cleaned.dtype == np.uint8
    n_px = int(cleaned.sum())
    print(f"        OK  surviving_pixels={n_px}")


def test_mask_to_polygons() -> list[dict]:
    print("  [2/4] mask_to_polygons ...")
    mask = np.zeros((1024, 1024), dtype=np.uint8)
    mask[100:300, 100:400] = 1   # 200 × 300 px = 60 000 px → 6 000 000 m²

    features = mask_to_polygons(mask, _TRANSFORM)

    assert len(features) >= 1, f"Expected ≥1 polygon, got {len(features)}"
    for f in features:
        assert f["type"] == "Feature"
        assert "geometry" in f
        assert f["properties"]["area_m2"] > 0
    total_m2 = sum(f["properties"]["area_m2"] for f in features)
    print(f"        OK  polygons={len(features)}  total_m2={total_m2:,.0f}")
    return features


def test_compute_stats(features: list[dict]) -> None:
    print("  [3/4] compute_stats ...")
    stats = compute_stats(features, aoi_area_m2=_AOI_M2)

    assert stats["change_area_m2"] > 0
    assert 0.0 < stats["change_pct"] < 100.0
    assert stats["polygon_count"] == len(features)
    print(
        f"        OK  area={stats['change_area_m2']:,.0f}m²"
        f"  pct={stats['change_pct']:.3f}%"
        f"  polygons={stats['polygon_count']}"
    )


def test_full_chain() -> None:
    """Synthetic 1024×1024 mask through the complete postprocess pipeline."""
    print("  [4/4] Full chain: clean → vectorise → stats ...")
    rng  = np.random.default_rng(42)
    # Scattered 1-pixel noise + one genuine 200×200 block
    mask = (rng.random((1024, 1024)) > 0.99).astype(np.uint8)
    mask[400:600, 400:600] = 1

    cleaned  = clean_mask(mask, min_area_px=50)
    features = mask_to_polygons(cleaned, _TRANSFORM)
    stats    = compute_stats(features, aoi_area_m2=_AOI_M2)

    # The large block must dominate
    assert stats["change_area_m2"] >= (200 * 10) ** 2 * 0.5, (
        f"Expected ≥2 000 000 m², got {stats['change_area_m2']}"
    )
    print(
        f"        OK  area={stats['change_area_m2']:,.0f}m²"
        f"  pct={stats['change_pct']:.3f}%"
        f"  polygons={stats['polygon_count']}"
    )


if __name__ == "__main__":
    print("=== Phase 6 — Postprocessor ===")

    test_clean_mask()
    features = test_mask_to_polygons()
    test_compute_stats(features)
    test_full_chain()

    print("=== ALL TESTS PASSED ===")
