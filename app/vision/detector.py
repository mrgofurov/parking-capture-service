from dataclasses import dataclass
from typing import List, Optional, Tuple
import numpy as np
import torch

from app.utils.logging import logger
from app.utils.metrics import VEHICLE_DETECTIONS


@dataclass
class VehicleDetection:
    bbox: Tuple[int, int, int, int]  # (x1, y1, x2, y2)
    confidence: float
    class_name: str
    class_id: int


# COCO vehicle class IDs
TARGET_VEHICLE_CLASSES = {
    2: "car",
    3: "motorcycle",
    5: "bus",
    7: "truck",
}


class VehicleDetector:
    """
    YOLO Vehicle Detector based on Ultralytics YOLO.
    Loaded once at startup. Detects cars, trucks, buses, motorcycles.
    Supports CUDA GPU acceleration with automatic CPU fallback.
    """

    def __init__(
        self,
        model_path: str = "yolov8n.pt",
        confidence_threshold: float = 0.40,
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
            logger.info(f"Loading YOLO Vehicle Detector from {self.model_path} on {self.device}...")
            self.model = YOLO(self.model_path)
            # Run one warm-up inference if possible
            dummy = np.zeros((320, 320, 3), dtype=np.uint8)
            self.model.predict(dummy, device=self.device, verbose=False)
            logger.info(f"YOLO Vehicle Detector loaded successfully on {self.device}")
        except Exception as e:
            logger.warning(
                f"Failed to load YOLO model '{self.model_path}' ({e}). Running in fallback mode."
            )
            self.model = None

    def detect(self, frame: np.ndarray, camera_id: str = "default") -> List[VehicleDetection]:
        """
        Detect vehicles in frame.
        """
        if frame is None or frame.size == 0 or self.model is None:
            return []

        results = self.model.predict(
            frame,
            conf=self.confidence_threshold,
            classes=list(TARGET_VEHICLE_CLASSES.keys()),
            device=self.device,
            verbose=False,
        )

        detections: List[VehicleDetection] = []
        if not results:
            return detections

        for r in results:
            boxes = r.boxes
            if boxes is None:
                continue

            for box in boxes:
                cls_id = int(box.cls[0].item())
                conf = float(box.conf[0].item())
                xyxy = box.xyxy[0].cpu().numpy().astype(int)
                x1, y1, x2, y2 = xyxy.tolist()

                cls_name = TARGET_VEHICLE_CLASSES.get(cls_id, "vehicle")
                detections.append(
                    VehicleDetection(
                        bbox=(x1, y1, x2, y2),
                        confidence=conf,
                        class_name=cls_name,
                        class_id=cls_id,
                    )
                )
                VEHICLE_DETECTIONS.labels(camera_id=camera_id, vehicle_class=cls_name).inc()

        return detections
