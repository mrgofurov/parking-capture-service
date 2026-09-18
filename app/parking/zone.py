from typing import Tuple, Optional


class ROIZone:
    """
    Region of Interest (ROI) filter for parking lanes.
    Ensures vehicles outside the designated lane are excluded from inference.
    """

    def __init__(
        self,
        enabled: bool = False,
        x1: int = 0,
        y1: int = 0,
        x2: int = 1920,
        y2: int = 1080,
    ):
        self.enabled = enabled
        self.x1 = min(x1, x2)
        self.y1 = min(y1, y2)
        self.x2 = max(x1, x2)
        self.y2 = max(y1, y2)

    def contains_point(self, x: float, y: float) -> bool:
        if not self.enabled:
            return True
        return self.x1 <= x <= self.x2 and self.y1 <= y <= self.y2

    def contains_bbox_center(self, bbox: Tuple[int, int, int, int]) -> bool:
        """
        Check if the center point of bounding box [x1, y1, x2, y2] is inside the ROI.
        """
        if not self.enabled:
            return True
        bx1, by1, bx2, by2 = bbox
        cx = (bx1 + bx2) / 2.0
        cy = (by1 + by2) / 2.0
        return self.contains_point(cx, cy)

    def bbox_overlap_ratio(self, bbox: Tuple[int, int, int, int]) -> float:
        """
        Calculate intersection over vehicle area ratio.
        """
        if not self.enabled:
            return 1.0
        bx1, by1, bx2, by2 = bbox
        box_area = max(1, (bx2 - bx1) * (by2 - by1))

        inter_x1 = max(self.x1, bx1)
        inter_y1 = max(self.y1, by1)
        inter_x2 = min(self.x2, bx2)
        inter_y2 = min(self.y2, by2)

        if inter_x1 >= inter_x2 or inter_y1 >= inter_y2:
            return 0.0

        inter_area = (inter_x2 - inter_x1) * (inter_y2 - inter_y1)
        return inter_area / float(box_area)

    def is_vehicle_in_zone(self, bbox: Tuple[int, int, int, int], min_overlap: float = 0.3) -> bool:
        """
        Returns True if vehicle is sufficiently inside the ROI.
        """
        if not self.enabled:
            return True
        return self.bbox_overlap_ratio(bbox) >= min_overlap
