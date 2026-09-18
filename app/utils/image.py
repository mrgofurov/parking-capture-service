import base64
import os
import shutil
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional, Tuple
import cv2
import numpy as np
from app.utils.logging import logger


def encode_image_to_jpeg(image: np.ndarray, quality: int = 85) -> bytes:
    """Encode OpenCV BGR image to JPEG bytes."""
    success, buffer = cv2.imencode(".jpg", image, [int(cv2.IMWRITE_JPEG_QUALITY), quality])
    if not success:
        raise ValueError("Failed to encode image to JPEG")
    return buffer.tobytes()


def encode_image_to_base64(image: np.ndarray, quality: int = 85) -> str:
    """Encode OpenCV BGR image to base64 string."""
    jpeg_bytes = encode_image_to_jpeg(image, quality=quality)
    return base64.b64encode(jpeg_bytes).decode("utf-8")


def decode_base64_to_image(b64_string: str) -> np.ndarray:
    """Decode base64 string to OpenCV BGR image."""
    img_data = base64.b64decode(b64_string)
    nparr = np.frombuffer(img_data, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError("Failed to decode base64 into image")
    return img


def save_snapshot(
    image: np.ndarray,
    upload_dir: str,
    camera_id: str,
    plate_number: str,
    timestamp: Optional[datetime] = None,
) -> str:
    """
    Save captured frame with organized structure:
    captures/<camera_id>/YYYY-MM-DD/<timestamp>_<plate_number>.jpg
    """
    if timestamp is None:
        timestamp = datetime.now()

    date_str = timestamp.strftime("%Y-%m-%d")
    time_str = timestamp.strftime("%H%M%S_%f")[:10]
    safe_plate = plate_number.replace("/", "_").replace("\\", "_")

    target_dir = Path(upload_dir) / camera_id / date_str
    target_dir.mkdir(parents=True, exist_ok=True)

    filename = f"{time_str}_{safe_plate}.jpg"
    file_path = target_dir / filename

    cv2.imwrite(str(file_path), image, [int(cv2.IMWRITE_JPEG_QUALITY), 90])
    return str(file_path)


def cleanup_expired_captures(upload_dir: str, retention_days: int) -> int:
    """
    Remove snapshot directories older than retention_days.
    Returns number of deleted directories.
    """
    if retention_days <= 0:
        return 0

    cutoff_date = datetime.now() - timedelta(days=retention_days)
    deleted_count = 0
    base_path = Path(upload_dir)

    if not base_path.exists():
        return 0

    try:
        for camera_dir in base_path.iterdir():
            if not camera_dir.is_dir():
                continue
            for date_dir in camera_dir.iterdir():
                if not date_dir.is_dir():
                    continue
                try:
                    dir_date = datetime.strptime(date_dir.name, "%Y-%m-%d")
                    if dir_date < cutoff_date:
                        shutil.rmtree(date_dir)
                        deleted_count += 1
                        logger.info(f"Cleaned up expired snapshot directory: {date_dir}")
                except ValueError:
                    # Not a YYYY-MM-DD directory name, skip
                    continue
    except Exception as e:
        logger.error(f"Error during capture directory cleanup: {e}")

    return deleted_count
