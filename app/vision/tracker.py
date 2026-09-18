import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
import numpy as np

from app.vision.detector import VehicleDetection
from app.utils.logging import logger


@dataclass
class TrackedVehicleObject:
    track_id: str
    bounding_box: Tuple[int, int, int, int]  # (x1, y1, x2, y2)
    confidence: float
    class_name: str
    first_seen: float = field(default_factory=time.time)
    last_seen: float = field(default_factory=time.time)
    frames_seen: int = 1
    best_detection_confidence: float = 0.0


def compute_iou(boxA: Tuple[int, int, int, int], boxB: Tuple[int, int, int, int]) -> float:
    """Compute Intersection over Union between two bounding boxes."""
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])

    interArea = max(0, xB - xA) * max(0, yB - yA)
    boxAArea = max(1, (boxA[2] - boxA[0]) * (boxA[3] - boxA[1]))
    boxBArea = max(1, (boxB[2] - boxB[0]) * (boxB[3] - boxB[1]))

    iou = interArea / float(boxAArea + boxBArea - interArea)
    return max(0.0, min(1.0, iou))


class VehicleTracker:
    """
    Multi-object vehicle tracker.
    Associates vehicle detections across consecutive frames to maintain consistent track_id.
    Maintains track history, first_seen, last_seen, and handles track timeouts.
    """

    def __init__(self, tracker_type: str = "bytetrack", iou_threshold: float = 0.3, max_lost_seconds: float = 3.0):
        self.tracker_type = tracker_type
        self.iou_threshold = iou_threshold
        self.max_lost_seconds = max_lost_seconds
        self.active_tracks: Dict[str, TrackedVehicleObject] = {}
        self._next_track_id = 1

    def update(self, detections: List[VehicleDetection]) -> List[TrackedVehicleObject]:
        """
        Match incoming detections to existing tracks using IoU association.
        """
        now = time.time()
        updated_tracks: List[TrackedVehicleObject] = []
        unmatched_detections = list(detections)

        # 1. Match with active tracks
        matched_track_ids = set()
        for track_id, track in list(self.active_tracks.items()):
            best_iou = 0.0
            best_det = None

            for det in unmatched_detections:
                iou = compute_iou(track.bounding_box, det.bbox)
                if iou > best_iou:
                    best_iou = iou
                    best_det = det

            if best_iou >= self.iou_threshold and best_det is not None:
                track.bounding_box = best_det.bbox
                track.confidence = best_det.confidence
                track.class_name = best_det.class_name
                track.last_seen = now
                track.frames_seen += 1
                if best_det.confidence > track.best_detection_confidence:
                    track.best_detection_confidence = best_det.confidence

                matched_track_ids.add(track_id)
                unmatched_detections.remove(best_det)
                updated_tracks.append(track)

        # 2. Create new tracks for unmatched detections
        for det in unmatched_detections:
            new_id = f"veh-{self._next_track_id}"
            self._next_track_id += 1
            new_track = TrackedVehicleObject(
                track_id=new_id,
                bounding_box=det.bbox,
                confidence=det.confidence,
                class_name=det.class_name,
                first_seen=now,
                last_seen=now,
                frames_seen=1,
                best_detection_confidence=det.confidence,
            )
            self.active_tracks[new_id] = new_track
            updated_tracks.append(new_track)

        # 3. Prune tracks that have been lost for too long
        lost_ids = [
            tid for tid, trk in self.active_tracks.items()
            if now - trk.last_seen > self.max_lost_seconds
        ]
        for tid in lost_ids:
            del self.active_tracks[tid]

        return updated_tracks
