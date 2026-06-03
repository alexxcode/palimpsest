from datetime import date

from pydantic import BaseModel


class DetectResponse(BaseModel):
    detection_id: str
    date_before_actual: date
    date_after_actual: date
    change_area_m2: float
    change_pct: float
    polygons: list[dict]
    gcs_uri_mask: str
    model_version: str
    processing_time_s: float
