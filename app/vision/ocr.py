import re
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


class OCREngine:
    """
    Unified OCR Engine supporting EasyOCR and PaddleOCR.
    EasyOCR runs natively on PyTorch (CPU/CUDA) without oneDNN issues,
    providing high accuracy on license plates and vehicle crops.
    """

    _instance: Optional["OCREngine"] = None

    def __init__(self, use_gpu: Optional[bool] = None, lang: str = "en"):
        self.lang = lang
        self.reader = None
        self.engine_type = "none"

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
        # 1. Try EasyOCR (native PyTorch, highly reliable)
        try:
            import easyocr
            logger.info(f"Initializing EasyOCR (gpu={self.use_gpu}, lang={self.lang})...")
            self.reader = easyocr.Reader([self.lang], gpu=self.use_gpu)
            self.engine_type = "easyocr"
            # Warm-up inference
            dummy = np.ones((64, 200, 3), dtype=np.uint8) * 255
            self.reader.readtext(dummy)
            logger.info("EasyOCR engine successfully initialized and warmed up")
            return
        except Exception as e:
            logger.warning(f"EasyOCR initialization failed ({e}), attempting PaddleOCR...")

        # 2. Try PaddleOCR as fallback
        try:
            from paddleocr import PaddleOCR
            logger.info(f"Initializing PaddleOCR (use_gpu={self.use_gpu}, lang={self.lang})...")
            self.reader = PaddleOCR(
                use_doc_orientation_classify=False,
                use_doc_unwarping=False,
                use_textline_orientation=False,
                lang=self.lang,
            )
            self.engine_type = "paddleocr"
            dummy = np.ones((64, 200, 3), dtype=np.uint8) * 255
            self.reader.predict(dummy)
            logger.info("PaddleOCR engine successfully initialized")
        except Exception as e2:
            logger.error(f"Failed to initialize any OCR engine: {e2}")
            self.reader = None
            self.engine_type = "none"

    def recognize(self, plate_crop: np.ndarray, camera_id: str = "default") -> List[OCRResult]:
        """
        Run OCR on license plate crop image.
        Returns list of detected text snippets with confidence.
        """
        if plate_crop is None or plate_crop.size == 0 or self.reader is None:
            return []

        start_time = time.time()
        OCR_ATTEMPTS.labels(camera_id=camera_id).inc()

        results: List[OCRResult] = []
        try:
            if self.engine_type == "easyocr":
                # EasyOCR returns list of (bbox, text, prob)
                ocr_res = self.reader.readtext(plate_crop)
                elapsed = time.time() - start_time
                OCR_LATENCY.labels(camera_id=camera_id).observe(elapsed)

                for item in ocr_res:
                    if len(item) >= 3:
                        bbox, raw_text, conf = item[0], item[1], float(item[2])
                    elif len(item) == 2:
                        bbox, raw_text, conf = None, item[0], float(item[1])
                    else:
                        continue

                    # Filter and clean alphanumeric characters
                    clean_text = re.sub(r"[^A-Za-z0-9]", "", str(raw_text)).upper()
                    if len(clean_text) >= 3:
                        results.append(
                            OCRResult(
                                text=clean_text,
                                confidence=conf,
                                bounding_box=bbox,
                            )
                        )

            elif self.engine_type == "paddleocr":
                ocr_res = self.reader.predict(plate_crop)
                elapsed = time.time() - start_time
                OCR_LATENCY.labels(camera_id=camera_id).observe(elapsed)
                for res in ocr_res:
                    for text, conf in zip(res.get("rec_texts", []), res.get("rec_scores", [])):
                        clean = re.sub(r"[^A-Za-z0-9]", "", str(text)).upper()
                        if len(clean) >= 3:
                            results.append(OCRResult(text=clean, confidence=float(conf)))

            if results:
                OCR_SUCCESS.labels(camera_id=camera_id).inc()

        except Exception as e:
            logger.error(f"OCR recognition error ({self.engine_type}): {e}")

        return results


# Backward compatibility alias
PaddleOCREngine = OCREngine


def get_ocr_engine(use_gpu: Optional[bool] = None) -> OCREngine:
    """Singleton getter for OCR engine."""
    if OCREngine._instance is None:
        OCREngine._instance = OCREngine(use_gpu=use_gpu)
    return OCREngine._instance
