from dataclasses import dataclass
from typing import Tuple
import cv2
import numpy as np


@dataclass
class QualityMetrics:
    sharpness: float       # Laplacian variance (higher = sharper)
    brightness: float      # Mean pixel intensity (0-255)
    contrast: float        # Standard deviation of pixel intensities
    width: int
    height: int
    aspect_ratio: float
    is_acceptable: bool
    rejection_reason: str = ""


def compute_plate_quality(
    plate_img: np.ndarray,
    min_width: int = 80,
    min_height: int = 20,
    min_sharpness: float = 25.0,
) -> QualityMetrics:
    """
    Compute quality metrics on cropped license plate to prevent running expensive
    OCR on blurry, micro, or degraded images.
    """
    if plate_img is None or plate_img.size == 0:
        return QualityMetrics(
            sharpness=0.0,
            brightness=0.0,
            contrast=0.0,
            width=0,
            height=0,
            aspect_ratio=0.0,
            is_acceptable=False,
            rejection_reason="Empty crop",
        )

    h, w = plate_img.shape[:2]
    aspect_ratio = round(float(w) / max(1, h), 2)

    # 1. Size check
    if w < min_width or h < min_height:
        return QualityMetrics(
            sharpness=0.0,
            brightness=0.0,
            contrast=0.0,
            width=w,
            height=h,
            aspect_ratio=aspect_ratio,
            is_acceptable=False,
            rejection_reason=f"Plate too small ({w}x{h} < {min_width}x{min_height})",
        )

    # Convert to grayscale for metric calculations
    if len(plate_img.shape) == 3:
        gray = cv2.cvtColor(plate_img, cv2.COLOR_BGR2GRAY)
    else:
        gray = plate_img

    # 2. Sharpness via Laplacian variance
    laplacian = cv2.Laplacian(gray, cv2.CV_64F)
    sharpness = float(laplacian.var())

    # 3. Brightness and Contrast
    brightness = float(np.mean(gray))
    contrast = float(np.std(gray))

    # 4. Blur filter
    if sharpness < min_sharpness:
        return QualityMetrics(
            sharpness=round(sharpness, 2),
            brightness=round(brightness, 2),
            contrast=round(contrast, 2),
            width=w,
            height=h,
            aspect_ratio=aspect_ratio,
            is_acceptable=False,
            rejection_reason=f"Plate image blurry (sharpness {sharpness:.1f} < {min_sharpness})",
        )

    # 5. Extreme lighting check (completely black or completely blown out white)
    if brightness < 15.0:
        return QualityMetrics(
            sharpness=round(sharpness, 2),
            brightness=round(brightness, 2),
            contrast=round(contrast, 2),
            width=w,
            height=h,
            aspect_ratio=aspect_ratio,
            is_acceptable=False,
            rejection_reason="Too dark",
        )
    if brightness > 245.0 and contrast < 10.0:
        return QualityMetrics(
            sharpness=round(sharpness, 2),
            brightness=round(brightness, 2),
            contrast=round(contrast, 2),
            width=w,
            height=h,
            aspect_ratio=aspect_ratio,
            is_acceptable=False,
            rejection_reason="Overexposed",
        )

    return QualityMetrics(
        sharpness=round(sharpness, 2),
        brightness=round(brightness, 2),
        contrast=round(contrast, 2),
        width=w,
        height=h,
        aspect_ratio=aspect_ratio,
        is_acceptable=True,
        rejection_reason="",
    )
