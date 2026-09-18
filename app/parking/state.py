import time
from enum import Enum
from typing import Dict, List, Optional, Tuple
import numpy as np

from app.parking.event import ParkingEvent
from app.parking.zone import ROIZone
from app.vision.consensus import ConsensusAggregator, ConsensusResult
from app.utils.logging import logger, log_event
from app.utils.metrics import PARKING_EVENTS


class VehicleState(str, Enum):
    ENTERED = "ENTERED"
    TRACKING = "TRACKING"
    CONSENSUS_ACHIEVED = "CONSENSUS_ACHIEVED"
    EVENT_EMITTED = "EVENT_EMITTED"
    EXITED = "EXITED"


class TrackedVehicle:
    def __init__(self, track_id: str, bbox: Tuple[int, int, int, int]):
        self.track_id = track_id
        self.bbox = bbox
        self.state = VehicleState.ENTERED
        self.first_seen = time.time()
        self.last_seen = time.time()
        self.frames_seen = 1
        self.last_ocr_time = 0.0
        self.confirmed_plate: Optional[str] = None
        self.confirmed_confidence: float = 0.0
        self.best_frame: Optional[np.ndarray] = None
        self.event_emitted = False

    def update(self, bbox: Tuple[int, int, int, int]) -> None:
        self.bbox = bbox
        self.last_seen = time.time()
        self.frames_seen += 1
        if self.state == VehicleState.ENTERED and self.frames_seen >= 2:
            self.state = VehicleState.TRACKING


class ParkingStateMachine:
    """
    Coordinates vehicle track states, ROI containment, consensus triggers,
    and event emissions.
    """

    def __init__(
        self,
        camera_id: str,
        direction: str,
        consensus_aggregator: ConsensusAggregator,
        roi_zone: ROIZone,
        track_timeout: float = 5.0,
    ):
        self.camera_id = camera_id
        self.direction = direction
        self.consensus = consensus_aggregator
        self.roi = roi_zone
        self.track_timeout = track_timeout
        self.vehicles: Dict[str, TrackedVehicle] = {}

    def update_vehicle(
        self,
        track_id: str,
        bbox: Tuple[int, int, int, int],
    ) -> TrackedVehicle:
        """Update or register a tracked vehicle."""
        if track_id not in self.vehicles:
            vehicle = TrackedVehicle(track_id, bbox)
            self.vehicles[track_id] = vehicle
            logger.debug(f"New vehicle track: {track_id}")
        else:
            vehicle = self.vehicles[track_id]
            vehicle.update(bbox)
        return vehicle

    def should_perform_ocr(self, track_id: str, interval_ms: int = 200) -> bool:
        """
        Rate-limit OCR calls per vehicle track to avoid excessive inference.
        """
        vehicle = self.vehicles.get(track_id)
        if not vehicle:
            return False

        # If already confirmed and event emitted, no need to OCR anymore
        if vehicle.event_emitted:
            return False

        now = time.time()
        if (now - vehicle.last_ocr_time) * 1000.0 >= interval_ms:
            vehicle.last_ocr_time = now
            return True
        return False

    def check_and_emit_event(self, track_id: str) -> Optional[ParkingEvent]:
        """
        Evaluate consensus for the track. If consensus is reached and not yet emitted,
        create and return ParkingEvent.
        """
        vehicle = self.vehicles.get(track_id)
        if not vehicle or vehicle.event_emitted:
            return None

        result: ConsensusResult = self.consensus.evaluate(track_id)

        if result.confirmed and result.plate_number:
            vehicle.state = VehicleState.CONSENSUS_ACHIEVED
            vehicle.confirmed_plate = result.plate_number
            vehicle.confirmed_confidence = result.confidence
            vehicle.best_frame = result.best_frame
            vehicle.event_emitted = True
            vehicle.state = VehicleState.EVENT_EMITTED

            event = ParkingEvent(
                camera_id=self.camera_id,
                direction=self.direction,
                plate_number=result.plate_number,
                confidence=result.confidence,
                track_id=track_id,
                sample_count=result.sample_count,
            )

            PARKING_EVENTS.labels(camera_id=self.camera_id, direction=self.direction).inc()
            log_event(
                event_type="PARKING_EVENT_GENERATED",
                camera_id=self.camera_id,
                track_id=track_id,
                plate_number=result.plate_number,
                ocr_confidence=result.confidence,
            )
            return event

        return None

    def cleanup_stale_tracks(self) -> List[ParkingEvent]:
        """
        Remove stale vehicle tracks that timed out.
        If a vehicle reached consensus before disappearing but event was not yet emitted,
        emit it now on vehicle exit.
        """
        now = time.time()
        emitted_on_exit: List[ParkingEvent] = []
        stale_ids = []

        for track_id, vehicle in self.vehicles.items():
            if now - vehicle.last_seen > self.track_timeout:
                stale_ids.append(track_id)
                # Attempt late consensus evaluation on exit if not emitted yet
                if not vehicle.event_emitted:
                    ev = self.check_and_emit_event(track_id)
                    if ev:
                        emitted_on_exit.append(ev)

        for tid in stale_ids:
            del self.vehicles[tid]
            self.consensus.sessions.pop(tid, None)

        return emitted_on_exit
