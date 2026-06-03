"""
Change Detection Analysis Agent — Gemini REST API via httpx.

Uses the Gemini REST API directly (no gRPC SDK) to avoid the
"503 Illegal metadata" conflict that occurs when google-generativeai
uses gRPC transport inside Cloud Run.

Requires GEMINI_API_KEY in environment (settings.gemini_api_key).
Get a free key at https://aistudio.google.com/app/apikey

Tools
-----
  lookup_location(lat, lon)  → human-readable place name via Nominatim
  search_context(query)      → Wikipedia extract for relevant events/projects

Workflow
--------
  1. Build a detection summary from the BigQuery row + geometry centroid.
  2. Run agentic function-calling loop via REST (max 6 turns).
  3. Structured JSON generation with responseMimeType=application/json.
"""
from __future__ import annotations

import json
import logging
import time
from datetime import datetime, timezone

import httpx
import shapely.wkt

from app.core.config import settings

logger = logging.getLogger(__name__)

# ── Constants ─────────────────────────────────────────────────────────────────

BASE_URL = "https://generativelanguage.googleapis.com/v1beta/models"

# ── System prompt ─────────────────────────────────────────────────────────────

_SYSTEM = """\
You are a remote sensing analyst specialising in satellite-based change detection.
You interpret results from an automated deep learning pipeline that compares
Sentinel-2 multispectral imagery (10 m/px) at two dates using a Siamese
EfficientNet-B2 model fine-tuned on the OSCD dataset.

When given a detection result you:
1. Use lookup_location to identify the exact geographic area.
2. Use search_context to find relevant events, projects, or disasters
   for that location and the period between the two image dates.
3. Synthesise an accurate, scientifically grounded interpretation.

Style rules:
- State areas in hectares (ha) or km²; 1 ha = 10,000 m².
- Express uncertainty honestly: "consistent with", "likely", "may indicate".
- Note when change area is small (< 1 ha), moderate (1–10 ha), or large (> 10 ha).
- The model operates at 10 m/px — changes < 500 m² (0.05 ha) may be artefacts.
- Write in English, scientific but accessible to a non-specialist audience."""

# ── Tool schemas (Gemini REST format) ────────────────────────────────────────

_TOOLS = [
    {
        "functionDeclarations": [
            {
                "name": "lookup_location",
                "description": (
                    "Reverse-geocode a WGS-84 coordinate pair to obtain a "
                    "human-readable location name: country, city, district, landmark."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "lat": {"type": "number", "description": "Latitude in decimal degrees"},
                        "lon": {"type": "number", "description": "Longitude in decimal degrees"},
                    },
                    "required": ["lat", "lon"],
                },
            },
            {
                "name": "search_context",
                "description": (
                    "Search Wikipedia for background context about a location or event. "
                    "Use to find details about infrastructure projects, natural disasters, "
                    "armed conflicts, urban development, or environmental events that could "
                    "explain the detected changes."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "query": {
                            "type": "string",
                            "description": (
                                "Search query combining location and event type, e.g. "
                                "'Beirut port explosion 2020' or "
                                "'NEOM The Line construction Saudi Arabia'."
                            ),
                        },
                    },
                    "required": ["query"],
                },
            },
        ]
    }
]

# ── Gemini REST helper ────────────────────────────────────────────────────────


def _gemini(
    api_key: str,
    contents: list[dict],
    *,
    tools: list[dict] | None = None,
    system_instruction: str | None = None,
    generation_config: dict | None = None,
) -> dict:
    """POST to Gemini generateContent endpoint and return parsed JSON.

    Retries on 429/503 with aggressive backoff (free tier = 15 RPM).
    """
    body: dict = {"contents": contents}
    if tools:
        body["tools"] = tools
    if system_instruction:
        body["systemInstruction"] = {"parts": [{"text": system_instruction}]}
    if generation_config:
        body["generationConfig"] = generation_config

    model = settings.gemini_model or "gemini-2.0-flash"
    url = f"{BASE_URL}/{model}:generateContent"

    max_retries = 5
    for attempt in range(max_retries + 1):
        resp = httpx.post(
            url,
            params={"key": api_key},
            json=body,
            timeout=90.0,
        )
        if resp.status_code in (429, 503) and attempt < max_retries:
            # Log full error body on first attempt for debugging
            if attempt == 0:
                logger.error(
                    "Gemini %s response body: %s",
                    resp.status_code, resp.text[:500],
                )
            retry_after = resp.headers.get("retry-after")
            if retry_after and retry_after.isdigit():
                wait = min(int(retry_after), 60)
            else:
                wait = [10, 20, 30, 45, 60][attempt]
            logger.warning(
                "Gemini %s (attempt %d/%d), retrying in %ds…",
                resp.status_code, attempt + 1, max_retries + 1, wait,
            )
            time.sleep(wait)
            continue
        resp.raise_for_status()
        return resp.json()

    resp.raise_for_status()
    return resp.json()


# ── Tool implementations ──────────────────────────────────────────────────────


def _lookup_location(lat: float, lon: float) -> str:
    """OpenStreetMap Nominatim reverse geocoding (no API key required)."""
    try:
        r = httpx.get(
            "https://nominatim.openstreetmap.org/reverse",
            params={"lat": lat, "lon": lon, "format": "json", "zoom": 14},
            headers={"User-Agent": "Palimpsest-ChangeDetection/1.0 (research)"},
            timeout=8.0,
        )
        r.raise_for_status()
        data = r.json()
        addr = data.get("address", {})
        parts = [
            addr.get("amenity") or addr.get("building") or addr.get("road"),
            addr.get("suburb") or addr.get("neighbourhood") or addr.get("city_district"),
            addr.get("city") or addr.get("town") or addr.get("village"),
            addr.get("state") or addr.get("region"),
            addr.get("country"),
        ]
        location = ", ".join(p for p in parts if p)
        return location or data.get("display_name", f"{lat:.4f}°, {lon:.4f}°")
    except Exception as exc:
        logger.warning("Nominatim lookup failed: %s", exc)
        return f"{lat:.4f}°N, {lon:.4f}°E"


def _search_context(query: str) -> str:
    """Wikipedia search — returns a 6-sentence intro extract for the top result."""
    try:
        search_r = httpx.get(
            "https://en.wikipedia.org/w/api.php",
            params={
                "action": "query", "list": "search",
                "srsearch": query, "srlimit": 3, "format": "json",
            },
            timeout=8.0,
        )
        search_r.raise_for_status()
        results = search_r.json().get("query", {}).get("search", [])
        if not results:
            return f"No Wikipedia results found for '{query}'."

        page_id = results[0]["pageid"]
        title   = results[0]["title"]
        ext_r   = httpx.get(
            "https://en.wikipedia.org/w/api.php",
            params={
                "action": "query", "pageids": page_id,
                "prop": "extracts", "exintro": True,
                "explaintext": True, "exsentences": 6,
                "format": "json",
            },
            timeout=8.0,
        )
        ext_r.raise_for_status()
        pages   = ext_r.json().get("query", {}).get("pages", {})
        extract = pages.get(str(page_id), {}).get("extract", "").strip()

        if not extract:
            return f"No extract available for '{title}'."
        return f"[Wikipedia: {title}]\n{extract}"

    except Exception as exc:
        logger.warning("Wikipedia search failed: %s", exc)
        return f"Context search unavailable ({exc})."


def _dispatch(name: str, args: dict) -> str:
    if name == "lookup_location":
        return _lookup_location(float(args["lat"]), float(args["lon"]))
    if name == "search_context":
        return _search_context(str(args["query"]))
    return f"Unknown tool: {name}"


# ── Geometry helper ───────────────────────────────────────────────────────────


def _centroid_from_wkt(wkt: str | None) -> tuple[float, float]:
    if not wkt:
        return 0.0, 0.0
    try:
        geom = shapely.wkt.loads(wkt)
        c    = geom.centroid
        return c.y, c.x
    except Exception:
        return 0.0, 0.0


# ── Analyst ───────────────────────────────────────────────────────────────────


class ChangeAnalyst:
    """
    Gemini agent (REST) that interprets a stored change detection result.

    Uses the Gemini REST API via httpx — no gRPC, no SDK, works in Cloud Run.
    Requires GEMINI_API_KEY (settings.gemini_api_key).
    """

    def __init__(self) -> None:
        if not settings.gemini_api_key:
            raise RuntimeError(
                "GEMINI_API_KEY is not configured. "
                "Get a free key at https://aistudio.google.com/app/apikey"
            )
        self._key = settings.gemini_api_key.strip()  # remove \r\n from Secret Manager

    # ------------------------------------------------------------------
    def analyse(self, detection: dict) -> dict:
        """
        Analyse a change detection result and return a structured interpretation.
        """
        change_m2   = float(detection.get("change_area_m2") or 0)
        change_ha   = change_m2 / 10_000
        change_pct  = float(detection.get("change_pct") or 0)
        area_name   = detection.get("area_name") or "not specified"
        date_before = detection.get("date_before")
        date_after  = detection.get("date_after")
        geom_wkt    = detection.get("geometry")
        n_polygons  = detection.get("polygon_count", "unknown")
        model_ver   = detection.get("model_version", "changeformer-effb2-oscd-v1")

        center_lat, center_lon = _centroid_from_wkt(geom_wkt)

        detection_summary = (
            f"Area name    : {area_name}\n"
            f"Centroid     : {center_lat:.5f}°N, {center_lon:.5f}°E\n"
            f"Before date  : {date_before}\n"
            f"After date   : {date_after}\n"
            f"Changed area : {change_ha:.2f} ha "
            f"({change_m2:,.0f} m² / {change_pct:.2f}% of AOI)\n"
            f"Polygons     : {n_polygons}\n"
            f"Model        : {model_ver}"
        )

        user_content = (
            f"Analyse the following satellite change detection result.\n\n"
            f"{detection_summary}\n\n"
            f"First use lookup_location to identify the exact area, then "
            f"use search_context to find relevant context for this location "
            f"and time period."
        )

        # ── Agentic function-calling loop ────────────────────────────
        contents: list[dict] = [
            {"role": "user", "parts": [{"text": user_content}]}
        ]
        gathered_context: list[str] = []

        for _turn in range(3):
            data = _gemini(
                self._key,
                contents,
                tools=_TOOLS,
                system_instruction=_SYSTEM,
            )

            model_parts = data["candidates"][0]["content"]["parts"]
            contents.append({"role": "model", "parts": model_parts})

            # Collect function calls
            fn_calls = [p["functionCall"] for p in model_parts if "functionCall" in p]
            if not fn_calls:
                break

            # Execute tools and build response parts
            fn_response_parts = []
            for fc in fn_calls:
                args   = fc.get("args", {})
                result = _dispatch(fc["name"], args)
                snippet = f"[{fc['name']}({json.dumps(args)})] → {result[:300]}"
                gathered_context.append(snippet)
                logger.info(
                    "Agent tool %s → %s…",
                    fc["name"], result[:80].replace("\n", " ")
                )
                fn_response_parts.append({
                    "functionResponse": {
                        "name":     fc["name"],
                        "response": {"result": result},
                    }
                })

            contents.append({"role": "user", "parts": fn_response_parts})

        # ── Structured output generation ─────────────────────────────
        ctx_block = (
            "\n\n".join(gathered_context)
            if gathered_context
            else "(no external context gathered)"
        )

        struct_prompt = (
            f"Based on the following context and detection data, "
            f"produce a JSON analysis.\n\n"
            f"CONTEXT:\n{ctx_block}\n\n"
            f"DETECTION DATA:\n{detection_summary}\n\n"
            f"Return a JSON object with exactly these keys:\n"
            f"- location: place name from the geocoding result\n"
            f"- summary: 1-2 sentence overview of what changed\n"
            f"- analysis: 3-5 sentence scientific interpretation referencing "
            f"  the area size, dates, and any context found\n"
            f"- likely_causes: array of 2-4 plausible explanations (strings)\n"
            f"- confidence: HIGH if strong contextual evidence, "
            f"  MEDIUM if partial match, LOW if speculative\n\n"
            f"Remote sensing context: {_SYSTEM}"
        )

        struct_data = _gemini(
            self._key,
            [{"role": "user", "parts": [{"text": struct_prompt}]}],
            generation_config={
                "responseMimeType": "application/json",
                "temperature": 0.2,
            },
        )
        raw_json = struct_data["candidates"][0]["content"]["parts"][0]["text"].strip()

        # Strip accidental markdown fences
        if raw_json.startswith("```"):
            raw_json = raw_json.split("```")[1]
            if raw_json.startswith("json"):
                raw_json = raw_json[4:]
        raw_json = raw_json.strip()

        try:
            structured = json.loads(raw_json)
        except json.JSONDecodeError:
            logger.warning(
                "Structured JSON parse failed; using fallback. Raw: %s", raw_json[:200]
            )
            structured = {
                "location":      area_name,
                "summary":       f"Change detection found {change_ha:.2f} ha of change.",
                "analysis":      raw_json[:500],
                "likely_causes": ["Unknown — manual review recommended"],
                "confidence":    "LOW",
            }

        return {
            "detection_id":  detection.get("detection_id"),
            "location":      structured.get("location", area_name),
            "summary":       structured.get("summary", ""),
            "analysis":      structured.get("analysis", ""),
            "likely_causes": structured.get("likely_causes", []),
            "confidence":    structured.get("confidence", "MEDIUM"),
            "context_used":  gathered_context,
            "generated_at":  datetime.now(timezone.utc).isoformat(),
        }
