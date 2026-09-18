import pytest
from app.parking.zone import ROIZone


def test_roi_disabled():
    roi = ROIZone(enabled=False)
    # Any box should be accepted when ROI is disabled
    assert roi.is_vehicle_in_zone((10, 10, 50, 50)) is True
    assert roi.contains_point(5000, 5000) is True


def test_roi_enabled_filtering():
    roi = ROIZone(enabled=True, x1=100, y1=100, x2=500, y2=500)

    # Completely inside
    assert roi.is_vehicle_in_zone((150, 150, 300, 300)) is True

    # Completely outside
    assert roi.is_vehicle_in_zone((600, 600, 800, 800)) is False

    # Point containment
    assert roi.contains_point(200, 200) is True
    assert roi.contains_point(50, 50) is False
