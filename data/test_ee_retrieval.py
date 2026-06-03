"""Verify Earth Engine connectivity and Sentinel-2 data availability."""
import datetime
import os

import ee


def main() -> None:
    ee.Initialize(project=os.getenv("EE_PROJECT"))

    # Test area: São Paulo region (same bbox used in API examples)
    bbox = [-46.65, -23.57, -46.60, -23.52]
    region = ee.Geometry.BBox(*bbox)

    collection = (
        ee.ImageCollection("COPERNICUS/S2_SR_HARMONIZED")
        .filterBounds(region)
        .filterDate("2019-01-01", "2019-12-31")
        .filter(ee.Filter.lt("CLOUDY_PIXEL_PERCENTAGE", 10))
    )

    count = collection.size().getInfo()
    print(f"Available Sentinel-2 scenes (2019, cloud < 10%): {count}")

    if count == 0:
        print("WARNING: no scenes found — check cloud threshold or date range")
        return

    least_cloudy = collection.sort("CLOUDY_PIXEL_PERCENTAGE").first()
    ts_ms = least_cloudy.get("system:time_start").getInfo()
    dt = datetime.datetime.fromtimestamp(ts_ms / 1000)
    cloud_pct = least_cloudy.get("CLOUDY_PIXEL_PERCENTAGE").getInfo()

    print(f"Least cloudy scene: {dt.strftime('%Y-%m-%d')} ({cloud_pct:.1f}% cloud)")
    print("Earth Engine retrieval test OK")


if __name__ == "__main__":
    main()
