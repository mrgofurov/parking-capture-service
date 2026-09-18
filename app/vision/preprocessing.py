from typing import Optional
import cv2
import numpy as np


def resize_plate(img: np.ndarray, target_height: int = 64) -> np.ndarray:
    """Resize plate maintaining aspect ratio for optimal OCR recognition."""
    h, w = img.shape[:2]
    if h == 0 or w == 0:
        return img
    scale = target_height / float(h)
    target_width = max(16, int(w * scale))
    return cv2.resize(img, (target_width, target_height), interpolation=cv2.INTER_CUBIC)


def enhance_contrast(gray: np.ndarray, clip_limit: float = 2.0) -> np.ndarray:
    """Apply CLAHE (Contrast Limited Adaptive Histogram Equalization)."""
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(8, 8))
    return clahe.apply(gray)


def sharpen_image(img: np.ndarray) -> np.ndarray:
    """Apply mild unsharp masking/sharpening kernel."""
    kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]], dtype=np.float32)
    return cv2.filter2D(img, -1, kernel)


def denoise_image(img: np.ndarray) -> np.ndarray:
    """Apply fast bilateral filter or Gaussian blur to suppress camera noise."""
    return cv2.bilateralFilter(img, 5, 50, 50)


def correct_perspective(img: np.ndarray) -> np.ndarray:
    """
    Attempt simple deskewing based on minimum area bounding rect if rotated.
    """
    if len(img.shape) == 3:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    else:
        gray = img

    thresh = cv2.adaptiveThreshold(
        gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2
    )
    coords = np.column_stack(np.where(thresh > 0))
    if len(coords) < 50:
        return img

    angle = cv2.minAreaRect(coords)[-1]
    if angle < -45:
        angle = -(90 + angle)
    else:
        angle = -angle

    # Only deskew if angle is moderate (between 2 and 20 degrees)
    if abs(angle) < 2.0 or abs(angle) > 25.0:
        return img

    h, w = img.shape[:2]
    center = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    rotated = cv2.warpAffine(
        img, M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE
    )
    return rotated


def preprocess_plate(
    plate_img: np.ndarray,
    target_height: int = 64,
    apply_clahe: bool = True,
    apply_sharpen: bool = True,
    apply_deskew: bool = False,
) -> np.ndarray:
    """
    Intelligent preprocessor for license plate crops before feeding to PaddleOCR.
    Optimized to enhance contrast and clarity without over-processing.
    """
    if plate_img is None or plate_img.size == 0:
        return plate_img

    # 1. Resize to optimal OCR resolution
    processed = resize_plate(plate_img, target_height=target_height)

    # 2. Deskew if requested
    if apply_deskew:
        processed = correct_perspective(processed)

    # 3. Convert to grayscale
    if len(processed.shape) == 3:
        gray = cv2.cvtColor(processed, cv2.COLOR_BGR2GRAY)
    else:
        gray = processed

    # 4. Enhance contrast if lighting is poor
    if apply_clahe:
        gray = enhance_contrast(gray, clip_limit=2.5)

    # 5. Mild sharpening
    if apply_sharpen:
        gray = sharpen_image(gray)

    # Convert back to 3-channel BGR image since PaddleOCR expects 3-channel input
    return cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
