import asyncio
import time
from typing import Callable, List, Optional
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
from app.vision.consensus import ConsensusAggregator
from app.vision.detector import VehicleDetector
from app.vision.ocr import PaddleOCREngine, get_ocr_engine
from app.vision.plate_detector import LicensePlateDetector
from app.vision.preprocessing import preprocess_plate
from app.vision.quality import compute_plate_quality
from app.vision.tracker import VehicleTracker


class VisionPipeline:
    """
    Complete end-to-end computer vision pipeline for parking capture:
    Camera Stream -> Frame Sampling -> ROI Filter -> Vehicle Detection -> Tracking ->
    Plate Detection -> Quality Check -> Preprocessing -> PaddleOCR -> Normalization ->
    Consensus Aggregator -> Deduplication -> Backend Dispatch.
    """

    def __init__(
        self,
        config: Settings,
        camera_stream: BaseCameraStream,
        backend_client: BackendClient,
        vehicle_detector: Optional[VehicleDetector] = None,
        plate_detector: Optional[LicensePlateDetector] = None,
        ocr_engine: Optional[PaddleOCREngine] = None,
    ):
        self.cfg = config
        self.stream = camera_stream
        self.backend = backend_client

        # Computer Vision Models (Loaded once)
        self.vehicle_detector = vehicle_detector or VehicleDetector(
            model_path=config.vehicle_model,
            confidence_threshold=config.vehicle_confidence,
        )
        self.plate_detector = plate_detector or LicensePlateDetector(
            model_path=config.plate_model,
            confidence_threshold=config.plate_confidence,
        )
        self.ocr_engine = ocr_engine or get_ocr_engine()

        # Tracking & Parking State
        self.tracker = VehicleTracker(tracker_type=config.tracker_type)
        self.roi = ROIZone(
            enabled=config.roi_enabled,
            x1=config.roi_x1,
            y1=config.roi_y1,
            x2=config.roi_x2,
            y2=config.roi_y2,
        )
        self.consensus = ConsensusAggregator(
            min_frames=config.min_consensus_count,
            min_confidence=config.min_final_confidence,
            session_timeout=config.track_timeout_seconds,
        )
        self.state_machine = ParkingStateMachine(
            camera_id=config.camera_id,
            direction=config.camera_direction,
            consensus_aggregator=self.consensus,
            roi_zone=self.roi,
            track_timeout=config.track_timeout_seconds,
        )
        self.dedup = DeduplicationManager(
            cooldown_seconds=config.event_cooldown_seconds,
            redis_url=config.redis_url,
        )

        # Runtime state
        self._running = False
        self._task: Optional[asyncio.Task] = None
        self.latest_frame: Optional[np.ndarray] = None
        self._subscribers: List[asyncio.Queue] = []

    def subscribe_mjpeg(self) -> asyncio.Queue:
        """Subscribe an async queue to receive JPEG frames for live browser stream."""
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
        """Start camera capture and pipeline processing task."""
        self._running = True
        self.stream.start()
        self._task = asyncio.create_task(self._process_loop())
        logger.info(f"VisionPipeline started for camera: {self.cfg.camera_id}")

    async def stop(self) -> None:
        """Stop pipeline and camera stream."""
        self._running = False
        if self._task:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
        self.stream.stop()
        logger.info(f"VisionPipeline stopped for camera: {self.cfg.camera_id}")

    async def _process_loop(self) -> None:
        """Main processing loop with target FPS pacing."""
        frame_interval = 1.0 / max(1, self.cfg.target_fps)

        while self._running:
            loop_start = time.time()
            success, frame = self.stream.read_frame()

            if not success or frame is None:
                await asyncio.sleep(0.02)
                continue

            self.latest_frame = frame
            FRAMES_PROCESSED.labels(camera_id=self.cfg.camera_id).inc()

            # Process single frame
            try:
                await self._process_single_frame(frame)
            except Exception as e:
                logger.error(f"Error during frame processing: {e}", exc_info=True)

            # Broadcast for live UI stream
            self._broadcast_mjpeg(frame)

            # Record metrics
            duration = time.time() - loop_start
            PROCESSING_LATENCY.labels(camera_id=self.cfg.camera_id).observe(duration)
            PROCESSING_FPS.labels(camera_id=self.cfg.camera_id).set(self.stream.last_fps)

            # Pacing
            sleep_needed = frame_interval - duration
            if sleep_needed > 0:
                await asyncio.sleep(sleep_needed)
            else:
                await asyncio.sleep(0.001)

    async def _process_single_frame(self, frame: np.ndarray) -> None:
        # 1. Vehicle Detection
        detections = self.vehicle_detector.detect(frame, camera_id=self.cfg.camera_id)

        # 2. ROI Filtering
        valid_detections = [
            d for d in detections
            if self.roi.is_vehicle_in_zone(d.bbox)
        ]

        # 3. Vehicle Tracking
        tracked_vehicles = self.tracker.update(valid_detections)

        # 4. Process each tracked vehicle
        for trk in tracked_vehicles:
            vehicle_obj = self.state_machine.update_vehicle(trk.track_id, trk.bounding_box)

            # Check if OCR interval permits inference for this track
            if not self.state_machine.should_perform_ocr(trk.track_id, interval_ms=self.cfg.ocr_interval_ms):
                continue

            # 5. Detect License Plate within vehicle bounding box
            plate_dets = self.plate_detector.detect_in_vehicle_crop(
                frame, trk.bounding_box, camera_id=self.cfg.camera_id
            )
            if not plate_dets:
                continue

            # Take plate with highest confidence
            best_plate_det = max(plate_dets, key=lambda p: p.confidence)

            # 6. Quality Check
            quality = compute_plate_quality(
                best_plate_det.crop,
                min_width=self.cfg.min_plate_width,
                min_height=self.cfg.min_plate_height,
                min_sharpness=self.cfg.min_sharpness_score,
            )
            if not quality.is_acceptable:
                logger.debug(f"Plate crop rejected: {quality.rejection_reason}")
                continue

            # 7. Image Preprocessing (CLAHE, sharpen, resize)
            preprocessed_crop = preprocess_plate(
                best_plate_det.crop,
                target_height=64,
                apply_clahe=True,
                apply_sharpen=True,
            )

            # 8. PaddleOCR Recognition
            ocr_results = self.ocr_engine.recognize(preprocessed_crop, camera_id=self.cfg.camera_id)
            if not ocr_results:
                continue

            best_ocr = max(ocr_results, key=lambda o: o.confidence)

            # 9. Add Observation to Consensus Aggregator
            self.consensus.add_candidate(
                track_id=trk.track_id,
                raw_text=best_ocr.text,
                ocr_confidence=best_ocr.confidence,
                plate_confidence=best_plate_det.confidence,
                sharpness=quality.sharpness,
                frame=frame.copy() if self.cfg.storage_best_frame else None,
            )

            # 10. Check consensus & emit event if confirmed
            event = self.state_machine.check_and_emit_event(trk.track_id)
            if event:
                await self._handle_confirmed_event(event, vehicle_obj.best_frame)

        # 11. Cleanup stale tracks and emit any confirmed events on exit
        exit_events = self.state_machine.cleanup_stale_tracks()
        for ev in exit_events:
            await self._handle_confirmed_event(ev, None)

        # Cleanup stale consensus sessions
        self.consensus.cleanup_stale_sessions()

    async def _handle_confirmed_event(
        self,
        event: ParkingEvent,
        best_frame: Optional[np.ndarray],
    ) -> None:
        """Deduplicate, attach snapshot, and dispatch confirmed event to backend."""
        # Check duplicate debounce
        if self.dedup.is_duplicate(event.camera_id, event.direction, event.plate_number):
            logger.info(
                f"Duplicate event dropped for plate {event.plate_number} "
                f"on {event.camera_id} (within {self.cfg.event_cooldown_seconds}s cooldown)"
            )
            return

        # Record event for deduplication
        self.dedup.record_event(event.camera_id, event.direction, event.plate_number)

        # Save snapshot & attach base64 if available
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

        # Send to backend
        await self.backend.send_event(event)
