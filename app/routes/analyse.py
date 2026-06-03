"""
POST /api/v1/detect/{detection_id}/analyse

Runs the ChangeAnalyst agent (Gemini 2.0 Flash on Vertex AI) on a previously
stored detection and returns a structured natural-language interpretation.
Uses Application Default Credentials — no external API key required.
"""
from __future__ import annotations

import logging

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from app.services.analyst import ChangeAnalyst
from app.services.storage import get_detection_by_id

logger = logging.getLogger(__name__)
router = APIRouter()


class AnalyseRequest(BaseModel):
    """Optional extra context not stored in BigQuery."""
    polygon_count: int | None = None


class AnalysisResponse(BaseModel):
    detection_id:  str
    location:      str
    summary:       str
    analysis:      str
    likely_causes: list[str]
    confidence:    str           # HIGH | MEDIUM | LOW
    context_used:  list[str]     # tool call snippets for transparency
    generated_at:  str           # ISO-8601 UTC timestamp


@router.post(
    "/detect/{detection_id}/analyse",
    response_model=AnalysisResponse,
    summary="AI analysis of a change detection result",
)
def analyse_detection(
    detection_id: str,
    body: AnalyseRequest = AnalyseRequest(),
) -> AnalysisResponse:
    """
    Retrieve a stored detection from BigQuery and run the Gemini analysis agent.

    The agent:
    1. Reverse-geocodes the change centroid to identify the location.
    2. Searches Wikipedia for relevant events/projects in that area and period.
    3. Returns a structured interpretation (summary, analysis, likely causes, confidence).

    Uses Vertex AI (Application Default Credentials).
    Returns HTTP 404 if the detection_id is not in BigQuery.
    """
    # ── Retrieve detection ─────────────────────────────────────────
    detection = get_detection_by_id(detection_id)
    if detection is None:
        raise HTTPException(
            status_code=404,
            detail=f"Detection '{detection_id}' not found in BigQuery.",
        )

    # Inject optional fields from request body
    if body.polygon_count is not None:
        detection["polygon_count"] = body.polygon_count

    # ── Run agent ──────────────────────────────────────────────────
    try:
        analyst = ChangeAnalyst()
        result  = analyst.analyse(detection)
    except RuntimeError as exc:
        # API key not configured
        raise HTTPException(status_code=503, detail=str(exc))
    except Exception as exc:
        logger.exception("Analysis agent failed for detection %s", detection_id)
        # Sanitise error: strip API keys from URLs that httpx includes
        import re
        safe_msg = re.sub(r"key=[^\s'\"&]+", "key=***", str(exc))
        raise HTTPException(
            status_code=500,
            detail=f"Analysis failed: {safe_msg}",
        )

    return AnalysisResponse(**result)
