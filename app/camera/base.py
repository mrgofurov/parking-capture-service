import time
from abc import ABC, abstractmethod
from enum import Enum
from typing import Optional, Tuple
import numpy as np


class CameraStatus(str, Enum):
    ONLINE = "ONLINE"
    OFFLINE = "OFFLINE"
    RECONNECTING = "RECONNECTING"


class BaseCameraStream(ABC):
    """
    Abstract base class for camera streams (RTSP, Video File, Synthetic).
    """

    def __init__(self, camera_id: str, target_fps: int = 10):
        self.camera_id = camera_id
        self.target_fps = target_fps
        self.frame_interval = 1.0 / max(1, target_fps)
        self.status = CameraStatus.OFFLINE
        self.last_frame_time = 0.0
        self.total_frames_read = 0
        self.last_fps = 0.0
        self._fps_window_start = time.time()
        self._fps_frame_count = 0

    @abstractmethod
    def start(self) -> None:
        """Initialize and start reading frames."""
        pass

    @abstractmethod
    def stop(self) -> None:
        """Stop reading frames and release resources."""
        pass

    @abstractmethod
    def read_frame(self) -> Tuple[bool, Optional[np.ndarray]]:
        """
        Read the latest available frame.
        Returns: (success: bool, frame: Optional[np.ndarray])
        """
        pass

    def _update_fps(self) -> None:
        """Calculate real-time running FPS."""
        now = time.time()
        self._fps_frame_count += 1
        elapsed = now - self._fps_window_start
        if elapsed >= 2.0:
            self.last_fps = round(self._fps_frame_count / elapsed, 2)
            self._fps_frame_count = 0
            self._fps_window_start = now
