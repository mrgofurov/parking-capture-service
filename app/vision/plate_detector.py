from dataclasses import dataclass
from typing import List, Optional, Tuple
import cv2
import numpy as np
import torch

from app.utils.logging import logger
from app.utils.metrics import PLATE_DETECTIONS


@dataclass
class PlateDetection:
    bbox: Tuple[int, int, int, int]  # (x1, y1, x2, y2) in frame coordinates
    confidence: float
    crop: np.ndarray


class LicensePlateDetector:
    """
    YOLO License Plate Detector.
    Can run on full frame or cropped vehicle bounding boxes (hierarchical detection).
    Loaded once at startup. Supports CUDA GPU acceleration with automatic CPU fallback.
    """

    def __init__(
        self,
        model_path: str = "yolov8n.pt",
        confidence_threshold: float = 0.45,
        device: Optional[str] = None,
    ):
        self.model_path = model_path
        self.confidence_threshold = confidence_threshold

        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device

        self.model = None
        self._load_model()

    def _load_model(self) -> None:
        try:
            from ultralytics import YOLO
            logger.info(f"Loading YOLO License Plate Detector from {self.model_path} on {self.device}...")
            self.model = YOLO(self.model_path)
            logger.info(f"YOLO License Plate Detector loaded successfully on {self.device}")
        except Exception as e:
            logger.warning(
                f"Failed to load Plate model '{self.model_path}' ({e}). Running in fallback mode."
            )
            self.model = None

    def detect_in_vehicle_crop(
        self,
        frame: np.ndarray,
        vehicle_bbox: Tuple[int, int, int, int],
        camera_id: str = "default",
    ) -> List[PlateDetection]:
        """
        Detect license plates inside a vehicle bounding box crop.
        Translates coordinates back to full image coordinates.
        """
        vx1, vy1, vx2, vy2 = vehicle_bbox
        h, w = frame.shape[:2]

        vx1 = max(0, min(w - 1, vx1))
        vy1 = max(0, min(h - 1, vy1))
        vx2 = max(0, min(w, vx2))
        vy2 = max(0, min(h, vy2))

        if vx2 - vx1 < 20 or vy2 - vy1 < 20:
            return []

        vehicle_crop = frame[vy1:vy2, vx1:vx2]
        if vehicle_crop.size == 0:
            return []

        # If dedicated plate model is available, run inference on crop
        if self.model is not None:
            try:
                results = self.model.predict(
                    vehicle_crop,
                    conf=self.confidence_threshold,
                    device=self.device,
                    verbose=False,
                )
                plate_detections: List[PlateDetection] = []
                for r in results:
                    boxes = r.boxes
                    if boxes is None:
                        continue
                    for box in boxes:
                        conf = float(box.conf[0].item())
                        xyxy = box.xyxy[0].cpu().numpy().astype(int)
                        lx1, ly1, lx2, ly2 = xyxy.tolist()

                        # Convert back to full frame coordinates
                        fx1 = vx1 + lx1
                        fy1 = vy1 + ly1
                        fx2 = vx1 + lx2
                        fy2 = vy1 + ly2

                        crop = frame[fy1:fy2, fx1:fx2]
                        if crop.size > 0:
                            plate_detections.append(
                                PlateDetection(bbox=(fx1, fy1, fx2, fy2), confidence=conf, crop=crop)
                            )
                            PLATE_DETECTIONS.labels(camera_id=camera_id).inc()
                if plate_detections:
                    return plate_detections
            except Exception as e:
                logger.debug(f"Plate inference error on crop: {e}")

        # Fallback / heuristic when no dedicated plate model is loaded:
        # In cars, the license plate is typically in the lower 40% of the vehicle box
        crop_h = vy2 - vy1
        crop_w = vx2 - vx1
        plate_y1 = int(vy1 + crop_h * 0.55)
        plate_y2 = int(vy1 + crop_h * 0.95)
        plate_x1 = int(vx1 + crop_w * 0.20)
        plate_x2 = int(vx1 + crop_w * 0.80)

        heuristic_crop = frame[plate_y1:plate_y2, plate_x1:plate_x2]
        if heuristic_crop.size > 0:
            PLATE_DETECTIONS.labels(camera_id=camera_id).inc()
            return [
                PlateDetection(
                    bbox=(plate_x1, plate_y1, plate_x2, plate_y2),
                    confidence=0.50,
                    crop=heuristic_crop,
                )
            ]

        return []
