import logging
import json
import sys
from datetime import datetime, timezone
from typing import Any, Dict, Optional


class JSONFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        log_data: Dict[str, Any] = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }
        if hasattr(record, "extra_data") and isinstance(record.extra_data, dict):
            log_data.update(record.extra_data)
        if record.exc_info:
            log_data["exception"] = self.formatException(record.exc_info)
        return json.dumps(log_data)


def setup_logger(name: str = "parking-capture", level: str = "INFO") -> logging.Logger:
    logger = logging.getLogger(name)
    logger.setLevel(getattr(logging, level.upper(), logging.INFO))
    if not logger.handlers:
        handler = logging.StreamHandler(sys.stdout)
        handler.setFormatter(JSONFormatter())
        logger.addHandler(handler)
        logger.propagate = False
    return logger


logger = setup_logger()


def log_event(
    event_type: str,
    camera_id: str,
    track_id: Optional[str] = None,
    plate_number: Optional[str] = None,
    ocr_confidence: Optional[float] = None,
    detection_confidence: Optional[float] = None,
    processing_time_ms: Optional[float] = None,
    **kwargs: Any,
) -> None:
    extra: Dict[str, Any] = {
        "event_type": event_type,
        "camera_id": camera_id,
    }
    if track_id is not None:
        extra["track_id"] = track_id
    if plate_number is not None:
        extra["plate_number"] = plate_number
    if ocr_confidence is not None:
        extra["ocr_confidence"] = round(ocr_confidence, 4)
    if detection_confidence is not None:
        extra["detection_confidence"] = round(detection_confidence, 4)
    if processing_time_ms is not None:
        extra["processing_time_ms"] = round(processing_time_ms, 2)
    extra.update(kwargs)

    logger.info(f"[{event_type}] {plate_number or track_id or camera_id}", extra={"extra_data": extra})
