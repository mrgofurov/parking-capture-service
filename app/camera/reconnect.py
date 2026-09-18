import random
import time
from typing import Callable, Optional
from app.utils.logging import logger
from app.utils.metrics import STREAM_RECONNECTS


class ReconnectManager:
    """
    Manages exponential backoff reconnection logic for RTSP streams.
    Delays increase: 1s, 2s, 4s, 8s, up to max_delay (e.g. 30s) with jitter.
    """

    def __init__(
        self,
        camera_id: str,
        initial_delay: float = 1.0,
        max_delay: float = 30.0,
        backoff_factor: float = 2.0,
        jitter_factor: float = 0.2,
    ):
        self.camera_id = camera_id
        self.initial_delay = initial_delay
        self.max_delay = max_delay
        self.backoff_factor = backoff_factor
        self.jitter_factor = jitter_factor
        self.current_delay = initial_delay
        self.attempt_count = 0

    def reset(self) -> None:
        """Reset reconnect counter and delay upon successful stream open."""
        self.current_delay = self.initial_delay
        self.attempt_count = 0

    def wait_for_reconnect(self) -> float:
        """
        Record reconnect attempt, wait with backoff, and return actual slept duration.
        """
        self.attempt_count += 1
        STREAM_RECONNECTS.labels(camera_id=self.camera_id).inc()

        # Add jitter
        jitter = random.uniform(-self.jitter_factor, self.jitter_factor) * self.current_delay
        sleep_duration = max(0.5, self.current_delay + jitter)

        logger.warning(
            f"Stream reconnect attempt #{self.attempt_count} for camera {self.camera_id}. "
            f"Backing off for {sleep_duration:.2f}s..."
        )
        time.sleep(sleep_duration)

        # Increase delay for next time
        self.current_delay = min(self.max_delay, self.current_delay * self.backoff_factor)
        return sleep_duration
