import time
from typing import Dict, Optional
from app.utils.logging import logger
from app.utils.metrics import DUPLICATE_EVENTS


class DeduplicationManager:
    """
    Prevents duplicate parking events using cooldown window:
    (camera_id, direction, plate_number) -> last_event_time.
    Supports in-memory TTL caching with automatic expiration, and optional Redis backend.
    """

    def __init__(self, cooldown_seconds: int = 10, redis_url: Optional[str] = None):
        self.cooldown_seconds = cooldown_seconds
        self.memory_cache: Dict[str, float] = {}
        self.redis_client = None

        if redis_url:
            try:
                import redis
                self.redis_client = redis.from_url(redis_url)
                logger.info(f"Connected to Redis for deduplication: {redis_url}")
            except Exception as e:
                logger.warning(f"Redis connection failed ({e}), falling back to in-memory deduplication")
                self.redis_client = None

    def _make_key(self, camera_id: str, direction: str, plate_number: str) -> str:
        return f"cooldown:{camera_id}:{direction.lower()}:{plate_number.upper()}"

    def is_duplicate(self, camera_id: str, direction: str, plate_number: str) -> bool:
        """
        Check if an event was emitted within the cooldown window.
        Returns True if duplicate (should be dropped).
        """
        key = self._make_key(camera_id, direction, plate_number)
        now = time.time()

        if self.redis_client:
            try:
                exists = self.redis_client.get(key)
                if exists:
                    DUPLICATE_EVENTS.labels(camera_id=camera_id).inc()
                    return True
                return False
            except Exception as e:
                logger.warning(f"Redis get error ({e}), checking in-memory cache")

        # In-memory check
        last_time = self.memory_cache.get(key)
        if last_time is not None:
            if now - last_time < self.cooldown_seconds:
                DUPLICATE_EVENTS.labels(camera_id=camera_id).inc()
                return True
            else:
                # Expired
                del self.memory_cache[key]

        return False

    def record_event(self, camera_id: str, direction: str, plate_number: str) -> None:
        """
        Record that an event was emitted to trigger the cooldown timer.
        """
        key = self._make_key(camera_id, direction, plate_number)
        now = time.time()

        if self.redis_client:
            try:
                self.redis_client.setex(key, self.cooldown_seconds, "1")
            except Exception as e:
                logger.warning(f"Redis setex error: {e}")

        # Always update in-memory cache
        self.memory_cache[key] = now
        self._cleanup_expired(now)

    def _cleanup_expired(self, now: float) -> None:
        """Prune expired keys to avoid unbounded memory growth."""
        if len(self.memory_cache) > 500:
            expired = [
                k for k, t in self.memory_cache.items()
                if now - t >= self.cooldown_seconds
            ]
            for k in expired:
                del self.memory_cache[k]
