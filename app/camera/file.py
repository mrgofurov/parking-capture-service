import os
import time
from typing import Optional, Tuple
import cv2
import numpy as np

from app.camera.base import BaseCameraStream, CameraStatus
from app.utils.logging import logger
from app.utils.metrics import CAMERA_CONNECTED, FRAMES_RECEIVED


class FileCameraStream(BaseCameraStream):
    """
    Video file stream adapter supporting continuous looping for testing & benchmarks.
    """

    def __init__(
        self,
        camera_id: str,
        file_path: str,
        target_fps: int = 10,
        loop: bool = True,
    ):
        super().__init__(camera_id=camera_id, target_fps=target_fps)
        self.file_path = file_path
        self.loop = loop
        self.cap: Optional[cv2.VideoCapture] = None
        self._running = False
        self._last_read_time = 0.0

    def start(self) -> None:
        if not os.path.exists(self.file_path):
            logger.error(f"Video file not found: {self.file_path}")
            self.status = CameraStatus.OFFLINE
            CAMERA_CONNECTED.labels(camera_id=self.camera_id).set(0)
            return

        self.cap = cv2.VideoCapture(self.file_path)
        if not self.cap.isOpened():
            logger.error(f"Failed to open video file: {self.file_path}")
            self.status = CameraStatus.OFFLINE
            CAMERA_CONNECTED.labels(camera_id=self.camera_id).set(0)
            return

        self._running = True
        self.status = CameraStatus.ONLINE
        CAMERA_CONNECTED.labels(camera_id=self.camera_id).set(1)
        logger.info(f"FileCameraStream started for {self.camera_id} from {self.file_path}")

    def stop(self) -> None:
        self._running = False
        if self.cap:
            self.cap.release()
            self.cap = None
        self.status = CameraStatus.OFFLINE
        CAMERA_CONNECTED.labels(camera_id=self.camera_id).set(0)
        logger.info(f"FileCameraStream stopped for {self.camera_id}")

    def read_frame(self) -> Tuple[bool, Optional[np.ndarray]]:
        if not self._running or self.cap is None:
            return False, None

        ret, frame = self.cap.read()
        if not ret or frame is None:
            if self.loop:
                # Seek to beginning
                self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                ret, frame = self.cap.read()
                if not ret or frame is None:
                    return False, None
            else:
                self.status = CameraStatus.OFFLINE
                CAMERA_CONNECTED.labels(camera_id=self.camera_id).set(0)
                return False, None

        self._last_read_time = time.time()
        self.last_frame_time = self._last_read_time
        self.total_frames_read += 1
        self._update_fps()
        FRAMES_RECEIVED.labels(camera_id=self.camera_id).inc()

        return True, frame
