import asyncio
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional
import httpx

from app.parking.event import ParkingEvent
from app.utils.logging import logger
from app.utils.metrics import BACKEND_REQUESTS, BACKEND_REQUEST_ERRORS


@dataclass
class BackendResult:
    success: bool
    barrier_opened: bool = False
    requires_payment: bool = False
    amount_due: float = 0.0
    message: str = ""
    session_code: Optional[str] = None
    raw_data: Optional[Dict[str, Any]] = None


class BackendClient:
    """
    Asynchronous client for sending parking capture events to central parking-backend.
    Features:
    - Configurable base URL & event path
    - Parses barrier control & payment requirements from backend response
    - Retry with exponential backoff (e.g. 1s, 2s, 5s)
    - In-memory bounded queue for offline resiliency
    - Background task to flush queued events and send heartbeat when backend is online
    """

    def __init__(
        self,
        base_url: str,
        event_path: str = "/api/v1/anpr/event",
        token: Optional[str] = None,
        timeout: float = 4.0,
        max_retries: int = 3,
        queue_size: int = 200,
    ):
        self.base_url = base_url.rstrip("/")
        self.event_path = "/" + event_path.lstrip("/")
        self.token = token
        self.timeout = timeout
        self.max_retries = max_retries
        self.max_queue_size = queue_size
        self.pending_queue: List[ParkingEvent] = []
        self._queue_lock = asyncio.Lock()
        self._flusher_task: Optional[asyncio.Task] = None
        self._running = False

    @property
    def event_url(self) -> str:
        return f"{self.base_url}{self.event_path}"

    @property
    def heartbeat_url(self) -> str:
        return f"{self.base_url}/api/v1/anpr/heartbeat"

    def _get_headers(self) -> dict:
        headers = {"Content-Type": "application/json"}
        if self.token:
            headers["Authorization"] = f"Bearer {self.token}"
        return headers

    async def send_heartbeat(self, camera_id: Optional[str] = None) -> bool:
        """Send camera heartbeat to parking-backend."""
        try:
            params = {"camera_id": camera_id} if camera_id else {}
            async with httpx.AsyncClient(timeout=2.0) as client:
                resp = await client.get(self.heartbeat_url, params=params, headers=self._get_headers())
                return resp.status_code == 200
        except Exception:
            return False

    async def start(self) -> None:
        """Start background queue flusher and heartbeat worker."""
        self._running = True
        self._flusher_task = asyncio.create_task(self._periodic_flush())
        logger.info(f"Backend client started for endpoint: {self.event_url}")

    async def stop(self) -> None:
        """Stop background queue flusher."""
        self._running = False
        if self._flusher_task:
            self._flusher_task.cancel()
            try:
                await self._flusher_task
            except asyncio.CancelledError:
                pass

    async def enqueue_event(self, event: ParkingEvent) -> None:
        """Enqueue event for background delivery if immediate send fails."""
        async with self._queue_lock:
            if len(self.pending_queue) >= self.max_queue_size:
                dropped = self.pending_queue.pop(0)
                logger.warning(f"Backend queue full, dropped oldest event for plate {dropped.plate_number}")
            self.pending_queue.append(event)

    async def send_event(self, event: ParkingEvent) -> Optional[BackendResult]:
        """
        Attempt to send parking event to backend with retry & exponential backoff.
        Returns BackendResult with barrier status if acknowledged, None if failed and queued.
        """
        payload = event.to_backend_payload()
        headers = self._get_headers()

        delays = [1.0, 2.0, 5.0]
        last_exception = None

        for attempt in range(1, self.max_retries + 1):
            try:
                async with httpx.AsyncClient(timeout=self.timeout) as client:
                    resp = await client.post(self.event_url, json=payload, headers=headers)
                    if 200 <= resp.status_code < 300:
                        BACKEND_REQUESTS.labels(status="success").inc()
                        data = resp.json()
                        inner = data.get("data") or {}
                        session = inner.get("session") or {}

                        result = BackendResult(
                            success=True,
                            barrier_opened=inner.get("barrier_opened", False),
                            requires_payment=inner.get("requires_payment", False),
                            amount_due=float(inner.get("amount_due", 0.0)),
                            message=inner.get("message") or data.get("message", ""),
                            session_code=session.get("session_code"),
                            raw_data=data,
                        )

                        logger.info(
                            f"Backend response for {event.plate_number}: '{result.message}' | "
                            f"Barrier: {'OPENED' if result.barrier_opened else 'CLOSED'}"
                            f"{f' | Amount Due: {result.amount_due:.0f} UZS' if result.requires_payment else ''}"
                        )
                        return result
                    elif resp.status_code >= 500:
                        BACKEND_REQUEST_ERRORS.labels(error_type=f"http_{resp.status_code}").inc()
                        logger.warning(
                            f"Backend returned 5xx (attempt {attempt}/{self.max_retries}): status {resp.status_code}"
                        )
                    else:
                        BACKEND_REQUEST_ERRORS.labels(error_type=f"http_{resp.status_code}").inc()
                        logger.error(
                            f"Backend rejected event (status {resp.status_code}): {resp.text}"
                        )
                        return BackendResult(success=False, message=resp.text)
            except (httpx.ConnectError, httpx.TimeoutException, httpx.NetworkError) as e:
                last_exception = e
                BACKEND_REQUEST_ERRORS.labels(error_type=type(e).__name__).inc()
                logger.warning(
                    f"Backend connection error (attempt {attempt}/{self.max_retries}): {e}"
                )

            if attempt < self.max_retries:
                sleep_sec = delays[attempt - 1] if attempt - 1 < len(delays) else 5.0
                await asyncio.sleep(sleep_sec)

        # If all retries failed, enqueue for offline delivery
        logger.warning(f"Queuing event {event.plate_number} for retry due to backend failure ({last_exception})")
        await self.enqueue_event(event)
        return None

    async def _periodic_flush(self) -> None:
        """Periodically flush pending events."""
        while self._running:
            try:
                await asyncio.sleep(5.0)
                async with self._queue_lock:
                    if not self.pending_queue:
                        continue
                    to_send = list(self.pending_queue)
                    self.pending_queue.clear()

                still_failing = []
                for ev in to_send:
                    success = await self._send_single_attempt(ev)
                    if not success:
                        still_failing.append(ev)
                        # Avoid pounding an offline server
                        break

                if still_failing:
                    async with self._queue_lock:
                        # Prepend still failing events back to queue
                        self.pending_queue = still_failing + self.pending_queue
                        if len(self.pending_queue) > self.max_queue_size:
                            self.pending_queue = self.pending_queue[-self.max_queue_size:]

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in backend queue flusher: {e}")

    async def _send_single_attempt(self, event: ParkingEvent) -> bool:
        """Single attempt to send queued event."""
        try:
            async with httpx.AsyncClient(timeout=self.timeout) as client:
                resp = await client.post(
                    self.event_url,
                    json=event.to_backend_payload(),
                    headers=self._get_headers(),
                )
                if 200 <= resp.status_code < 300:
                    BACKEND_REQUESTS.labels(status="success").inc()
                    logger.info(f"Flushed pending event for plate: {event.plate_number}")
                    return True
                return False
        except Exception:
            return False
