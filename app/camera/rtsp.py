import queue
import threading
import time
from typing import Optional, Tuple
import cv2
import numpy as np

from app.camera.base import BaseCameraStream, CameraStatus
from app.camera.reconnect import ReconnectManager
from app.utils.logging import logger
from app.utils.metrics import CAMERA_CONNECTED, FRAMES_RECEIVED


class RTSPCameraStream(BaseCameraStream):
    """
    Production-grade RTSP stream reader with background grabber thread,
    exponential backoff auto-reconnect, and zero-latency latest frame buffer.
    """

    def __init__(
        self,
        camera_id: str,
        rtsp_url: str,
        target_fps: int = 10,
        max_queue_size: int = 2,
    ):
        super().__init__(camera_id=camera_id, target_fps=target_fps)
        self.rtsp_url = rtsp_url
        self.max_queue_size = max_queue_size
        self.reconnect_mgr = ReconnectManager(camera_id=camera_id)

        self._frame_queue: queue.Queue = queue.Queue(maxsize=max_queue_size)
        self._running = False
        self._thread: Optional[threading.Thread] = None
        self._cap: Optional[cv2.VideoCapture] = None

    def start(self) -> None:
        if self._running:
            return
        self._running = True
        self._thread = threading.Thread(target=self._capture_worker, daemon=True)
        self._thread.start()
        logger.info(f"RTSPCameraStream reader thread started for {self.camera_id}")

    def stop(self) -> None:
        self._running = False
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=3.0)
        self._release_cap()
        self.status = CameraStatus.OFFLINE
        CAMERA_CONNECTED.labels(camera_id=self.camera_id).set(0)
        logger.info(f"RTSPCameraStream stopped for {self.camera_id}")

    def read_frame(self) -> Tuple[bool, Optional[np.ndarray]]:
        """
        Read the latest available frame from the bounded queue.
        """
        if not self._running:
            return False, None

        try:
            frame = self._frame_queue.get(timeout=0.1)
            self.last_frame_time = time.time()
            self._update_fps()
            return True, frame
        except queue.Empty:
            return False, None

    def _open_stream(self) -> bool:
        """Attempt to open RTSP stream with low-latency flags."""
        self._release_cap()
        logger.info(f"Opening RTSP stream for {self.camera_id}...")

        # Configure environment / backend flags for low latency
        cap = cv2.VideoCapture(self.rtsp_url, cv2.CAP_FFMPEG)
        # Attempt to set buffer size to minimal to prevent delay accumulation
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

        if not cap.isOpened():
            logger.warning(f"Failed to open RTSP stream for {self.camera_id}")
            cap.release()
            return False

        self._cap = cap
        self.status = CameraStatus.ONLINE
        self.reconnect_mgr.reset()
        CAMERA_CONNECTED.labels(camera_id=self.camera_id).set(1)
        logger.info(f"RTSP stream successfully connected for {self.camera_id}")
        return True

    def _release_cap(self) -> None:
        if self._cap is not None:
            try:
                self._cap.release()
            except Exception:
                pass
            self._cap = None

    def _capture_worker(self) -> None:
        """
        Background loop reading RTSP frames continuously and keeping only the freshest frames.
        """
        while self._running:
            if self._cap is None or not self._cap.isOpened():
                self.status = CameraStatus.RECONNECTING
                CAMERA_CONNECTED.labels(camera_id=self.camera_id).set(0)
                if not self._open_stream():
                    self.reconnect_mgr.wait_for_reconnect()
                    continue

            # Read frame
            ret, frame = self._cap.read()
            if not ret or frame is None:
                logger.warning(f"Corrupted frame or stream disconnected for {self.camera_id}")
                self._release_cap()
                self.status = CameraStatus.RECONNECTING
                CAMERA_CONNECTED.labels(camera_id=self.camera_id).set(0)
                self.reconnect_mgr.wait_for_reconnect()
                continue

            self.total_frames_read += 1
            FRAMES_RECEIVED.labels(camera_id=self.camera_id).inc()

            # Push to queue with drop-oldest strategy
            if self._frame_queue.full():
                try:
                    self._frame_queue.get_nowait()
                except queue.Empty:
                    pass

            try:
                self._frame_queue.put_nowait(frame)
            except queue.Full:
                pass
