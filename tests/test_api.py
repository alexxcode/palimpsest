"""Integration tests for FastAPI endpoints — Phase 7."""
import pytest
from httpx import ASGITransport, AsyncClient

from app.main import app


@pytest.mark.asyncio
async def test_health():
    async with AsyncClient(
        transport=ASGITransport(app=app), base_url="http://test"
    ) as client:
        resp = await client.get("/health")
    assert resp.status_code == 200
    assert resp.json()["status"] == "ok"


@pytest.mark.asyncio
async def test_detect_not_implemented():
    async with AsyncClient(
        transport=ASGITransport(app=app), base_url="http://test"
    ) as client:
        resp = await client.post(
            "/api/v1/detect",
            json={
                "bbox": [-46.65, -23.57, -46.60, -23.52],
                "date_before": "2019-06-01",
                "date_after": "2023-06-01",
            },
        )
    assert resp.status_code == 501
