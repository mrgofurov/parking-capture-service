import time
from typing import List, Optional, Tuple
import cv2
import numpy as np


class FrameAnnotator:
    """
    Renders high-quality, professional computer vision overlays on video frames:
    - Vehicle bounding boxes with track IDs and confidence
    - License plate bounding boxes with highlighted corner brackets
    - Uzbek license plate pill badge with blue flag accent
    - Lane status and ANPR confirmation HUD
    """

    @staticmethod
    def draw_rounded_rect(
        img: np.ndarray,
        pt1: Tuple[int, int],
        pt2: Tuple[int, int],
        color: Tuple[int, int, int],
        thickness: int = 2,
        radius: int = 6,
    ) -> None:
        """Draw a clean rectangle with corner brackets for a high-tech edge AI look."""
        x1, y1 = pt1
        x2, y2 = pt2
        w = x2 - x1
        h = y2 - y1

        # Main rectangle
        cv2.rectangle(img, (x1, y1), (x2, y2), color, thickness)

        # Corner brackets
        corner_len = max(8, min(w // 4, h // 4, 24))
        thick_corner = thickness + 2
        # Top-Left
        cv2.line(img, (x1, y1), (x1 + corner_len, y1), color, thick_corner)
        cv2.line(img, (x1, y1), (x1, y1 + corner_len), color, thick_corner)
        # Top-Right
        cv2.line(img, (x2, y1), (x2 - corner_len, y1), color, thick_corner)
        cv2.line(img, (x2, y1), (x2, y1 + corner_len), color, thick_corner)
        # Bottom-Left
        cv2.line(img, (x1, y2), (x1 + corner_len, y2), color, thick_corner)
        cv2.line(img, (x1, y2), (x1, y2 - corner_len), color, thick_corner)
        # Bottom-Right
        cv2.line(img, (x2, y2), (x2 - corner_len, y2), color, thick_corner)
        cv2.line(img, (x2, y2), (x2, y2 - corner_len), color, thick_corner)

    @staticmethod
    def draw_plate_badge(
        img: np.ndarray,
        plate_bbox: Tuple[int, int, int, int],
        plate_text: str,
        confidence: float = 0.90,
    ) -> None:
        """Draw an Uzbek-style license plate badge above or on the plate bbox."""
        px1, py1, px2, py2 = plate_bbox
        clean_text = plate_text.strip().upper()
        if not clean_text:
            return

        # Format Uzbek plate format if matching 8 characters e.g. 01A777AA -> 01 | A 777 AA
        formatted = clean_text
        if len(clean_text) == 8 and clean_text[:2].isdigit():
            formatted = f"{clean_text[:2]} | {clean_text[2:]}"

        font = cv2.FONT_HERSHEY_SIMPLEX
        scale = 0.65
        thick = 2
        (text_w, text_h), baseline = cv2.getTextSize(formatted, font, scale, thick)

        badge_w = text_w + 36
        badge_h = text_h + 16

        # Position badge above plate if possible, else below
        badge_x = max(10, px1)
        badge_y = py1 - badge_h - 6
        if badge_y < 40:
            badge_y = py2 + 6

        img_h, img_w = img.shape[:2]
        if badge_x + badge_w > img_w:
            badge_x = img_w - badge_w - 10

        # Draw white badge card with border
        cv2.rectangle(
            img,
            (badge_x, badge_y),
            (badge_x + badge_w, badge_y + badge_h),
            (255, 255, 255),
            cv2.FILLED,
        )
        cv2.rectangle(
            img,
            (badge_x, badge_y),
            (badge_x + badge_w, badge_y + badge_h),
            (20, 20, 20),
            2,
        )

        # Draw blue vertical accent stripe on left (representing flag)
        stripe_w = 12
        cv2.rectangle(
            img,
            (badge_x + 2, badge_y + 2),
            (badge_x + stripe_w, badge_y + badge_h - 2),
            (200, 100, 0),  # BGR blue
            cv2.FILLED,
        )

        # Draw plate text
        text_origin = (badge_x + stripe_w + 8, badge_y + badge_h - 7)
        cv2.putText(
            img,
            formatted,
            text_origin,
            font,
            scale,
            (10, 10, 10),
            thick,
            cv2.LINE_AA,
        )

    @staticmethod
    def draw_vehicle(
        img: np.ndarray,
        bbox: Tuple[int, int, int, int],
        track_id: str,
        class_name: str = "Avto",
        confidence: float = 0.90,
    ) -> None:
        """Draw tracked vehicle frame and tag."""
        x1, y1, x2, y2 = bbox
        color = (0, 220, 100)  # Bright emerald green

        # Bounding box
        FrameAnnotator.draw_rounded_rect(img, (x1, y1), (x2, y2), color, thickness=2)

        # Vehicle label pill
        label = f"{class_name.upper()} #{track_id} ({int(confidence * 100)}%)"
        font = cv2.FONT_HERSHEY_SIMPLEX
        scale = 0.45
        thick = 1
        (tw, th), _ = cv2.getTextSize(label, font, scale, thick)

        pill_y1 = max(0, y1 - th - 10)
        pill_y2 = y1
        cv2.rectangle(img, (x1, pill_y1), (x1 + tw + 14, pill_y2), color, cv2.FILLED)
        cv2.putText(
            img,
            label,
            (x1 + 6, pill_y2 - 5),
            font,
            scale,
            (0, 0, 0),
            thick,
            cv2.LINE_AA,
        )

    @staticmethod
    def draw_hud(
        img: np.ndarray,
        camera_id: str,
        camera_name: str,
        direction: str,
        fps: float,
        last_event_text: Optional[str] = None,
        last_event_time: float = 0.0,
    ) -> None:
        """Render top HUD telemetry and confirmation banner."""
        h, w = img.shape[:2]
        font = cv2.FONT_HERSHEY_SIMPLEX

        # Top bar background
        overlay = img.copy()
        cv2.rectangle(overlay, (0, 0), (w, 36), (15, 23, 42), cv2.FILLED)
        cv2.addWeighted(overlay, 0.75, img, 0.25, 0, img)

        # Camera info
        dir_color = (0, 220, 100) if direction.upper() == "ENTRY" else (240, 150, 0)
        dir_label = "KIRISH (ENTRY)" if direction.upper() == "ENTRY" else "CHIQISH (EXIT)"
        cv2.putText(
            img,
            f"[LIVE] {dir_label} | {camera_name}",
            (14, 24),
            font,
            0.52,
            dir_color,
            2,
            cv2.LINE_AA,
        )

        # FPS & Engine tag
        fps_text = f"FPS: {fps:.1f} | EDGE ANPR AI"
        (tw, th), _ = cv2.getTextSize(fps_text, font, 0.50, 1)
        cv2.putText(
            img,
            fps_text,
            (w - tw - 16, 24),
            font,
            0.50,
            (200, 210, 225),
            1,
            cv2.LINE_AA,
        )

        # If recent event confirmed within last 4 seconds, show banner
        if last_event_text and (time.time() - last_event_time < 4.0):
            banner_h = 44
            banner_y = h - banner_h - 12
            # Glowing banner
            cv2.rectangle(img, (12, banner_y), (w - 12, banner_y + banner_h), (0, 180, 80), cv2.FILLED)
            cv2.rectangle(img, (12, banner_y), (w - 12, banner_y + banner_h), (255, 255, 255), 2)
            banner_text = f"[OK] ANPR TASDIQLANDI: {last_event_text} -> SHLAGBAUM OCHILDI"
            cv2.putText(
                img,
                banner_text,
                (24, banner_y + 28),
                font,
                0.60,
                (0, 0, 0),
                2,
                cv2.LINE_AA,
            )
