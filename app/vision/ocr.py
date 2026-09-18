import time
from dataclasses import dataclass
from typing import List, Optional, Tuple
import numpy as np

from app.utils.logging import logger
from app.utils.metrics import OCR_ATTEMPTS, OCR_LATENCY, OCR_SUCCESS


@dataclass
class OCRResult:
    text: str
    confidence: float
    bounding_box: Optional[List[Tuple[int, int]]] = None


class PaddleOCREngine:
    """
    PaddleOCR Engine wrapper.
    Initialized once at startup as a singleton.
    Supports CUDA GPU acceleration with automatic CPU fallback.
    """

    _instance: Optional["PaddleOCREngine"] = None

    def __init__(self, use_gpu: Optional[bool] = None, lang: str = "en"):
        self.lang = lang
        self.ocr_model = None

        # GPU detection
        if use_gpu is None:
            try:
                import torch
                self.use_gpu = torch.cuda.is_available()
            except Exception:
                self.use_gpu = False
        else:
            self.use_gpu = use_gpu

        self._initialize_model()

    def _initialize_model(self) -> None:
        try:
            from paddleocr import PaddleOCR
            logger.info(f"Initializing PaddleOCR (use_gpu={self.use_gpu}, lang={self.lang})...")
            self.ocr_model = PaddleOCR(
                use_angle_cls=True,
                lang=self.lang,
                use_gpu=self.use_gpu,
                show_log=False,
            )
            # Warm-up inference
            dummy = np.ones((64, 200, 3), dtype=np.uint8) * 255
            self.ocr_model.ocr(dummy, cls=True)
            logger.info("PaddleOCR engine successfully initialized and warmed up")
        except Exception as e:
            logger.warning(
                f"PaddleOCR failed to initialize with use_gpu={self.use_gpu} ({e}). "
                "Attempting CPU fallback..."
            )
            try:
                from paddleocr import PaddleOCR
                self.use_gpu = False
                self.ocr_model = PaddleOCR(
                    use_angle_cls=False,
                    lang=self.lang,
                    use_gpu=False,
                    show_log=False,
                )
                logger.info("PaddleOCR initialized successfully in CPU mode")
            except Exception as e2:
                logger.error(f"Failed to initialize PaddleOCR in CPU mode: {e2}")
                self.ocr_model = None

    def recognize(self, plate_crop: np.ndarray, camera_id: str = "default") -> List[OCRResult]:
        """
        Run OCR on license plate crop image.
        Returns list of detected text snippets with confidence.
        """
        if plate_crop is None or plate_crop.size == 0 or self.ocr_model is None:
            return []

        start_time = time.time()
        OCR_ATTEMPTS.labels(camera_id=camera_id).inc()

        results: List[OCRResult] = []
        try:
            ocr_res = self.ocr_model.ocr(plate_crop, cls=True)
            elapsed = time.time() - start_time
            OCR_LATENCY.labels(camera_id=camera_id).observe(elapsed)

            if not ocr_res or not ocr_res[0]:
                return []

            for line in ocr_res[0]:
                # line format: [ [ [x1,y1],[x2,y2],[x3,y3],[x4,y4] ], (text, confidence) ]
                bbox_points, (text, conf) = line
                clean_text = str(text).strip()
                if clean_text:
                    results.append(
                        OCRResult(
                            text=clean_text,
                            confidence=float(conf),
                            bounding_box=bbox_points,
                        )
                    )

            if results:
                OCR_SUCCESS.labels(camera_id=camera_id).inc()

        except Exception as e:
            logger.error(f"PaddleOCR recognition exception: {e}")

        return results


def get_ocr_engine(use_gpu: Optional[bool] = None) -> PaddleOCREngine:
    """Singleton getter for PaddleOCREngine."""
    if PaddleOCREngine._instance is None:
        PaddleOCREngine._instance = PaddleOCREngine(use_gpu=use_gpu)
    return PaddleOCREngine._instance
