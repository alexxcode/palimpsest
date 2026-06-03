from datetime import date

from pydantic import BaseModel, Field


class DetectRequest(BaseModel):
    bbox: list[float] = Field(..., min_length=4, max_length=4)
    date_before: date
    date_after: date
    max_cloud_pct: float = Field(10.0, ge=0.0, le=100.0)
    area_name: str | None = None
