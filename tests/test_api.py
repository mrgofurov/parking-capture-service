import pytest
from httpx import ASGITransport, AsyncClient
from app.main import create_app


@pytest.mark.asyncio
async def test_health_endpoints():
    app = create_app()
    # Bypass heavy lifespan during simple HTTP endpoint testing
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        # /health
        resp = await client.get("/health")
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "ok"
        assert "camera_connected" in data
        assert "gpu_available" in data

        # /ready
        resp_ready = await client.get("/ready")
        assert resp_ready.status_code == 200
        assert resp_ready.json() == {"status": "ready"}

        # /health/live & /health/ready
        resp_live = await client.get("/health/live")
        assert resp_live.status_code == 200
        assert resp_live.json()["status"] == "UP"

        resp_ready_legacy = await client.get("/health/ready")
        assert resp_ready_legacy.status_code == 200
        assert resp_ready_legacy.json()["status"] == "READY"


@pytest.mark.asyncio
async def test_metrics_endpoint():
    app = create_app()
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        resp = await client.get("/metrics")
        assert resp.status_code == 200
        content = resp.text
        # Check that standard defined metrics are exposed
        assert "frames_received_total" in content
        assert "frames_processed_total" in content
