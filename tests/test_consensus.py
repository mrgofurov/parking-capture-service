import numpy as np
import pytest
from app.vision.consensus import ConsensusAggregator


def test_consensus_success_with_multiple_frames():
    agg = ConsensusAggregator(min_frames=2, min_confidence=0.75)
    dummy_frame = np.zeros((100, 100, 3), dtype=np.uint8)

    # Frame 1: lower confidence or slight distortion
    agg.add_candidate(
        track_id="veh-1",
        raw_text="01A123BC",
        ocr_confidence=0.82,
        plate_confidence=0.85,
        sharpness=50.0,
        frame=dummy_frame,
    )

    # Initially 1 frame: should not be confirmed since min_frames=2
    res1 = agg.evaluate("veh-1")
    assert res1.confirmed is False
    assert res1.sample_count == 1

    # Frame 2: confirming frame
    agg.add_candidate(
        track_id="veh-1",
        raw_text="01A123BC",
        ocr_confidence=0.92,
        plate_confidence=0.90,
        sharpness=85.0,
        frame=dummy_frame,
    )

    res2 = agg.evaluate("veh-1")
    assert res2.confirmed is True
    assert res2.plate_number == "01A123BC"
    assert res2.confidence >= 0.75
    assert res2.sample_count == 2
    assert res2.format_valid is True


def test_consensus_outlier_rejection():
    """
    Simulate user scenario:
    Frame 1 -> 01A123BC (0.85)
    Frame 2 -> 01A12JBC (0.50)  <-- noisy OCR
    Frame 3 -> 01A123BC (0.91)
    Winner must be 01A123BC.
    """
    agg = ConsensusAggregator(min_frames=2, min_confidence=0.70)

    agg.add_candidate("veh-2", "01A123BC", 0.85, 0.88, 60.0)
    agg.add_candidate("veh-2", "01A12JBC", 0.50, 0.60, 40.0)
    agg.add_candidate("veh-2", "01A123BC", 0.91, 0.92, 90.0)

    res = agg.evaluate("veh-2")
    assert res.confirmed is True
    assert res.plate_number == "01A123BC"
    assert res.sample_count == 2


def test_consensus_insufficient_confidence():
    agg = ConsensusAggregator(min_frames=2, min_confidence=0.90)

    # Only low confidence reads
    agg.add_candidate("veh-3", "01A123BC", 0.60, 0.50, 20.0)
    agg.add_candidate("veh-3", "01A123BC", 0.62, 0.55, 22.0)

    res = agg.evaluate("veh-3")
    assert res.confirmed is False
    assert res.confidence < 0.90
