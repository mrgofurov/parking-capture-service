import time
import pytest
from app.parking.state import ParkingStateMachine, VehicleState
from app.parking.zone import ROIZone
from app.vision.consensus import ConsensusAggregator


def test_state_machine_lifecycle():
    consensus = ConsensusAggregator(min_frames=2, min_confidence=0.75)
    roi = ROIZone(enabled=False)
    sm = ParkingStateMachine(
        camera_id="entry-01",
        direction="entry",
        consensus_aggregator=consensus,
        roi_zone=roi,
        track_timeout=0.5,
    )

    # 1. First appearance
    v1 = sm.update_vehicle("veh-100", (100, 100, 300, 300))
    assert v1.state == VehicleState.ENTERED

    # 2. Tracking
    v2 = sm.update_vehicle("veh-100", (105, 102, 305, 302))
    assert v2.state == VehicleState.TRACKING

    # 3. OCR rate limiting
    assert sm.should_perform_ocr("veh-100", interval_ms=200) is True
    # Immediate next check should be False (rate limited)
    assert sm.should_perform_ocr("veh-100", interval_ms=200) is False

    # 4. Add consensus candidates
    consensus.add_candidate("veh-100", "01A123BC", 0.90, 0.88, 70.0)
    assert sm.check_and_emit_event("veh-100") is None  # only 1 frame

    consensus.add_candidate("veh-100", "01A123BC", 0.92, 0.90, 80.0)
    event = sm.check_and_emit_event("veh-100")  # 2 frames reached!
    assert event is not None
    assert event.plate_number == "01A123BC"
    assert event.camera_id == "entry-01"
    assert v1.state == VehicleState.EVENT_EMITTED

    # Second check must not emit duplicate
    assert sm.check_and_emit_event("veh-100") is None
