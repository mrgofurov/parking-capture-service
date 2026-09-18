import pytest
from app.vision.detector import VehicleDetection
from app.vision.tracker import VehicleTracker, compute_iou


def test_compute_iou():
    box1 = (0, 0, 100, 100)
    box2 = (0, 0, 100, 100)
    assert compute_iou(box1, box2) == 1.0

    box3 = (200, 200, 300, 300)
    assert compute_iou(box1, box3) == 0.0

    box4 = (50, 0, 150, 100)
    assert 0.3 < compute_iou(box1, box4) < 0.4


def test_tracker_association():
    tracker = VehicleTracker(iou_threshold=0.3)

    # Frame 1: Initial vehicle detection
    det1 = [VehicleDetection(bbox=(10, 10, 100, 100), confidence=0.85, class_name="car", class_id=2)]
    tracks1 = tracker.update(det1)
    assert len(tracks1) == 1
    track_id = tracks1[0].track_id

    # Frame 2: Vehicle slightly moved
    det2 = [VehicleDetection(bbox=(15, 12, 105, 102), confidence=0.88, class_name="car", class_id=2)]
    tracks2 = tracker.update(det2)
    assert len(tracks2) == 1
    # Track ID must remain constant across frames
    assert tracks2[0].track_id == track_id
    assert tracks2[0].frames_seen == 2
