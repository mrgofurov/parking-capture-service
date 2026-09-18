import asyncio
import pytest
from app.backend.client import BackendClient
from app.parking.event import ParkingEvent


@pytest.mark.asyncio
async def test_backend_client_offline_queuing():
    client = BackendClient(
        base_url="http://non-existent-backend-host.invalid:9999",
        event_path="/api/v1/parking/camera-events",
        timeout=0.2,
        max_retries=1,
        queue_size=10,
    )

    ev = ParkingEvent(
        camera_id="entry-01",
        direction="entry",
        plate_number="01A123BC",
        confidence=0.92,
    )

    # When backend is unreachable, send_event returns None and enqueues event
    res = await client.send_event(ev)
    assert res is None
    assert len(client.pending_queue) == 1
    assert client.pending_queue[0].plate_number == "01A123BC"

    # Queue bounds check
    for i in range(15):
        await client.enqueue_event(
            ParkingEvent(
                camera_id="entry-01",
                direction="entry",
                plate_number=f"01A{i:03d}BC",
                confidence=0.9,
            )
        )
    # Must not exceed max_queue_size
    assert len(client.pending_queue) == 10


@pytest.mark.asyncio
async def test_live_parking_backend_integration():
    """Test actual live integration with running parking-backend on port 8085."""
    client = BackendClient(
        base_url="http://localhost:8085",
        event_path="/api/v1/anpr/event",
        timeout=2.0,
    )

    # Test Heartbeat
    hb_ok = await client.send_heartbeat()
    assert hb_ok is True

    # Test ENTRY event
    entry_ev = ParkingEvent(
        camera_id="entry-01",
        direction="entry",
        plate_number="01A888AA",
        confidence=0.96,
    )
    entry_res = await client.send_event(entry_ev)
    assert entry_res is not None
    assert entry_res.success is True
    assert entry_res.barrier_opened is True

    # Test EXIT event
    exit_ev = ParkingEvent(
        camera_id="exit-01",
        direction="exit",
        plate_number="01A888AA",
        confidence=0.97,
    )
    exit_res = await client.send_event(exit_ev)
    assert exit_res is not None
    assert exit_res.success is True

