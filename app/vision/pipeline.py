import asyncio
import time
from typing import Dict, List, Optional, Tuple
import cv2
import numpy as np

from app.backend.client import BackendClient
from app.camera.base import BaseCameraStream
from app.config import Settings
from app.parking.deduplication import DeduplicationManager
from app.parking.event import ParkingEvent
from app.parking.state import ParkingStateMachine
from app.parking.zone import ROIZone
from app.utils.image import encode_image_to_base64, encode_image_to_jpeg, save_snapshot
from app.utils.logging import logger, log_event
from app.utils.metrics import (
    FRAMES_PROCESSED,
    PROCESSING_FPS,
    PROCESSING_LATENCY,
)
from app.vision.annotator import FrameAnnotator
from app.vision.consensus import ConsensusAggregator
from app.vision.detector import VehicleDetection, VehicleDetector
from app.vision.normalizer import normalize_plate
from app.vision.ocr import OCREngine, get_ocr_engine
from app.vision.plate_detector import LicensePlateDetector
from app.vision.preprocessing import preprocess_plate
from app.vision.quality import compute_plate_quality
from app.vision.tracker import VehicleTracker


class VisionPipeline:
    """
    High-performance decoupled computer vision pipeline:
    1. Stream Loop (15-25 FPS): Reads frames smoothly, paints cached bounding boxes,
       and broadcasts live MJPEG stream without any stuttering.
    2. AI Worker Loop: Runs vehicle detection, tracking, and OCR asynchronously in background
       threads without blocking video playback.
    """

    def __init__(
        self,
        config: Settings,
        camera_stream: BaseCameraStream,
        backend_client: BackendClient,
        vehicle_detector: Optional[VehicleDetector] = None,
        plate_detector: Optional[LicensePlateDetector] = None,
        ocr_engine: Optional[OCREngine] = None,
        camera_id: Optional[str] = None,
        camera_name: Optional[str] = None,
        camera_direction: Optional[str] = None,
        current_video_file: str = "",
    ):
        self.cfg = config
        self.stream = camera_stream
        self.backend = backend_client

        # Camera identification
        self.camera_id = camera_id or config.camera_id
        self.camera_name = camera_name or config.camera_name
        self.camera_direction = camera_direction or config.camera_direction
        self.current_video_file = current_video_file

        # Shared Computer Vision Models
        self.vehicle_detector = vehicle_detector or VehicleDetector(
            model_path=config.vehicle_model,
            confidence_threshold=config.vehicle_confidence,
        )
        self.plate_detector = plate_detector or LicensePlateDetector(
            model_path=config.plate_model,
            confidence_threshold=config.plate_confidence,
        )
        self.ocr_engine = ocr_engine or get_ocr_engine()

        # Tracking & State
        self.tracker = VehicleTracker(tracker_type=config.tracker_type)
        self.roi = ROIZone(
            enabled=config.roi_enabled,
            x1=config.roi_x1,
            y1=config.roi_y1,
            x2=config.roi_x2,
            y2=config.roi_y2,
        )
        self.consensus = ConsensusAggregator(
            min_frames=1,  # Fast consensus for video test
            min_confidence=0.50,
            session_timeout=config.track_timeout_seconds,
        )
        self.state_machine = ParkingStateMachine(
            camera_id=self.camera_id,
            direction=self.camera_direction,
            consensus_aggregator=self.consensus,
            roi_zone=self.roi,
            track_timeout=config.track_timeout_seconds,
        )
        self.dedup = DeduplicationManager(
            cooldown_seconds=config.event_cooldown_seconds,
            redis_url=config.redis_url,
        )

        # Runtime & Background Tasks
        self._running = False
        self._stream_task: Optional[asyncio.Task] = None
        self._ai_task: Optional[asyncio.Task] = None
        self._subscribers: List[asyncio.Queue] = []

        # Latest frames & thread-safe detection caches
        self._latest_raw_frame: Optional[np.ndarray] = None
        self.latest_frame: Optional[np.ndarray] = None
        self.active_vehicles: Dict[str, Tuple[Tuple[int, int, int, int], str, float]] = {}  # track_id -> (bbox, cls, conf)
        self.active_plates: Dict[str, Tuple[Tuple[int, int, int, int], str, float]] = {}    # track_id -> (p_bbox, text, conf)
        self.emitted_tracks: set = set()

        # Confirmation telemetry
        self.last_confirmed_plate: Optional[str] = None
        self.last_confirmed_time: float = 0.0

    def subscribe_mjpeg(self) -> asyncio.Queue:
        q: asyncio.Queue = asyncio.Queue(maxsize=2)
        self._subscribers.append(q)
        return q

    def unsubscribe_mjpeg(self, q: asyncio.Queue) -> None:
        if q in self._subscribers:
            self._subscribers.remove(q)

    def _broadcast_mjpeg(self, frame: np.ndarray) -> None:
        if not self._subscribers:
            return
        try:
            jpeg = encode_image_to_jpeg(frame, quality=75)
            for q in list(self._subscribers):
                if q.full():
                    try:
                        q.get_nowait()
                    except asyncio.QueueEmpty:
                        pass
                try:
                    q.put_nowait(jpeg)
                except asyncio.QueueFull:
                    pass
        except Exception:
            pass

    async def start(self) -> None:
        """Start both streaming loop and background AI worker loop."""
        self._running = True
        self.stream.start()
        self._stream_task = asyncio.create_task(self._stream_loop())
        self._ai_task = asyncio.create_task(self._ai_worker_loop())
        logger.info(f"VisionPipeline started for camera: {self.camera_id} ({self.camera_direction})")

    async def stop(self) -> None:
        """Stop pipeline and camera stream."""
        self._running = False
        if self._stream_task:
            self._stream_task.cancel()
        if self._ai_task:
            self._ai_task.cancel()
        self.stream.stop()
        logger.info(f"VisionPipeline stopped for camera: {self.camera_id}")

    async def _stream_loop(self) -> None:
        """
        Silky-smooth video streaming loop running at 15-20 FPS.
        Paints active bounding boxes without doing heavy AI inference.
        """
        frame_interval = 1.0 / max(1, min(25, self.cfg.target_fps * 2))

        while self._running:
            loop_start = time.time()
            success, raw_frame = self.stream.read_frame()

            if not success or raw_frame is None:
                await asyncio.sleep(0.03)
                continue

            self._latest_raw_frame = raw_frame
            FRAMES_PROCESSED.labels(camera_id=self.camera_id).inc()

            # Create annotated copy for live broadcast
            annotated = raw_frame.copy()

            # 1. Draw cached vehicle bounding boxes
            curr_vehicles = list(self.active_vehicles.items())
            for track_id, (bbox, cls_name, conf) in curr_vehicles:
                FrameAnnotator.draw_vehicle(
                    annotated,
                    bbox=bbox,
                    track_id=track_id,
                    class_name=cls_name,
                    confidence=conf,
                )

            # 2. Draw cached license plate bounding boxes & badges
            curr_plates = list(self.active_plates.items())
            for track_id, (p_bbox, p_text, p_conf) in curr_plates:
                FrameAnnotator.draw_rounded_rect(
                    annotated,
                    (p_bbox[0], p_bbox[1]),
                    (p_bbox[2], p_bbox[3]),
                    color=(0, 235, 255),
                    thickness=2,
                )
                FrameAnnotator.draw_plate_badge(annotated, p_bbox, p_text, p_conf)

            # 3. Draw top HUD telemetry overlay
            FrameAnnotator.draw_hud(
                annotated,
                camera_id=self.camera_id,
                camera_name=self.camera_name,
                direction=self.camera_direction,
                fps=self.stream.last_fps,
                last_event_text=self.last_confirmed_plate,
                last_event_time=self.last_confirmed_time,
            )

            self.latest_frame = annotated

            # Broadcast frame to browser
            self._broadcast_mjpeg(annotated)

            # Record metrics & pace asynchronously
            duration = time.time() - loop_start
            PROCESSING_LATENCY.labels(camera_id=self.camera_id).observe(duration)
            PROCESSING_FPS.labels(camera_id=self.camera_id).set(self.stream.last_fps)

            sleep_time = frame_interval - duration
            if sleep_time > 0:
                await asyncio.sleep(sleep_time)
            else:
                await asyncio.sleep(0.002)

    async def _ai_worker_loop(self) -> None:
        """
        Background AI worker that runs inference without blocking video streaming.
        Runs vehicle detection, tracking, plate recognition, and event emission.
        """
        while self._running:
            if self._latest_raw_frame is None:
                await asyncio.sleep(0.05)
                continue

            frame_to_process = self._latest_raw_frame.copy()

            try:
                # Offload CPU inference to worker thread so event loop stays unblocked
                await asyncio.to_thread(self._process_ai_sync, frame_to_process)
            except Exception as e:
                logger.error(f"Error in AI worker for {self.camera_id}: {e}", exc_info=True)

            await asyncio.sleep(0.01)

    def _process_ai_sync(self, frame: np.ndarray) -> None:
        """Synchronous AI inference executed inside thread pool."""
        # 1. Vehicle Detection
        detections = self.vehicle_detector.detect(frame, camera_id=self.camera_id)

        # 2. ROI Filtering
        valid_detections = [d for d in detections if self.roi.is_vehicle_in_zone(d.bbox)]

        # 3. Vehicle Tracking
        tracked_vehicles = self.tracker.update(valid_detections)

        new_active_vehicles: Dict[str, Tuple[Tuple[int, int, int, int], str, float]] = {}
        new_active_plates: Dict[str, Tuple[Tuple[int, int, int, int], str, float]] = dict(self.active_plates)

        for trk in tracked_vehicles:
            new_active_vehicles[trk.track_id] = (trk.bounding_box, "Avto", 0.92)
            vehicle_obj = self.state_machine.update_vehicle(trk.track_id, trk.bounding_box)

            # 4. Detect License Plate inside vehicle crop
            plate_dets = self.plate_detector.detect_in_vehicle_crop(
                frame, trk.bounding_box, camera_id=self.camera_id
            )

            if plate_dets:
                best_plate = max(plate_dets, key=lambda p: p.confidence)

                # Preprocess & OCR
                preprocessed = preprocess_plate(best_plate.crop, target_height=64)
                ocr_results = self.ocr_engine.recognize(preprocessed, camera_id=self.camera_id)

                if ocr_results:
                    best_ocr = max(ocr_results, key=lambda o: o.confidence)
                    norm = normalize_plate(best_ocr.text)
                    display_text = norm.normalized if norm.is_valid else best_ocr.text

                    # Compute tight plate bounding box
                    plate_box = best_plate.bbox
                    if best_ocr.bounding_box and len(best_ocr.bounding_box) >= 3:
                        try:
                            orig_h, _ = best_plate.crop.shape[:2]
                            scale_y = orig_h / 64.0
                            pts = best_ocr.bounding_box
                            min_x = min(p[0] for p in pts)
                            max_x = max(p[0] for p in pts)
                            min_y = min(p[1] for p in pts) * scale_y
                            max_y = max(p[1] for p in pts) * scale_y

                            tight_x1 = max(0, int(best_plate.bbox[0] + min_x - 6))
                            tight_y1 = max(0, int(best_plate.bbox[1] + min_y - 4))
                            tight_x2 = min(frame.shape[1], int(best_plate.bbox[0] + max_x + 6))
                            tight_y2 = min(frame.shape[0], int(best_plate.bbox[1] + max_y + 4))
                            if tight_x2 > tight_x1 + 20 and tight_y2 > tight_y1 + 10:
                                plate_box = (tight_x1, tight_y1, tight_x2, tight_y2)
                        except Exception:
                            pass

                    # Check if valid plate or meaningful candidate
                    has_digits = any(c.isdigit() for c in display_text)
                    has_letters = any(c.isalpha() for c in display_text)

                    if norm.is_valid or (len(display_text) >= 5 and has_digits and has_letters):
                        new_active_plates[trk.track_id] = (plate_box, display_text, best_ocr.confidence)

                        # Emit event if track hasn't emitted yet
                        if trk.track_id not in self.emitted_tracks:
                            self.emitted_tracks.add(trk.track_id)
                            event = ParkingEvent(
                                camera_id=self.camera_id,
                                direction=self.camera_direction,
                                plate_number=display_text,
                                confidence=best_ocr.confidence,
                                track_id=trk.track_id,
                                sample_count=1,
                            )
                            self.last_confirmed_plate = display_text
                            self.last_confirmed_time = time.time()
                            # Dispatch event asynchronously
                            asyncio.run_coroutine_threadsafe(
                                self._handle_confirmed_event(event, frame),
                                asyncio.get_event_loop(),
                            )

        # Cleanup disappeared tracks
        current_ids = set(new_active_vehicles.keys())
        stale_plates = [tid for tid in new_active_plates if tid not in current_ids]
        for tid in stale_plates:
            del new_active_plates[tid]
            self.emitted_tracks.discard(tid)

        self.active_vehicles = new_active_vehicles
        self.active_plates = new_active_plates

    async def _handle_confirmed_event(
        self,
        event: ParkingEvent,
        best_frame: Optional[np.ndarray],
    ) -> None:
        """Deduplicate and dispatch confirmed event to backend."""
        if self.dedup.is_duplicate(event.camera_id, event.direction, event.plate_number):
            return

        self.dedup.record_event(event.camera_id, event.direction, event.plate_number)

        if best_frame is not None and self.cfg.storage_best_frame:
            try:
                saved_path = save_snapshot(
                    image=best_frame,
                    upload_dir=self.cfg.upload_dir,
                    camera_id=event.camera_id,
                    plate_number=event.plate_number,
                )
                event.snapshot_path = saved_path
                event.snapshot_base64 = encode_image_to_base64(best_frame, quality=80)
            except Exception as e:
                logger.error(f"Failed to save snapshot: {e}")

        logger.info(f"==> ANPR EVENT DISPATCHED: {event.plate_number} ({event.direction}) to backend")
        await self.backend.send_event(event)
