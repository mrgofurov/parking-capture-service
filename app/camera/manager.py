import glob
import os
from typing import Any, Dict, List, Optional, Tuple
from app.backend.client import BackendClient
from app.camera.file import FileCameraStream
from app.camera.rtsp import RTSPCameraStream
from app.config import Settings
from app.utils.logging import logger
from app.vision.detector import VehicleDetector
from app.vision.ocr import get_ocr_engine
from app.vision.pipeline import VisionPipeline
from app.vision.plate_detector import LicensePlateDetector


class CameraManager:
    """
    Orchestrates multiple camera pipelines (e.g. Entry Lane and Exit Lane).
    Shares AI models (YOLO & OCR) across pipelines for maximum efficiency.
    Supports dynamic video switching from test-videos/ folder or RTSP sources.
    """

    def __init__(self, config: Settings, backend_client: BackendClient, videos_dir: Optional[str] = None):
        self.cfg = config
        self.backend = backend_client

        # Discover test videos directory
        if videos_dir and os.path.exists(videos_dir):
            self.videos_dir = videos_dir
        else:
            candidates = [
                os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../test-videos")),
                "/home/murtazo/projects/smart-parking/test-videos",
                "./test-videos",
                "../test-videos",
            ]
            self.videos_dir = next((c for c in candidates if os.path.exists(c)), "./test-videos")

        # Load shared models once in memory
        logger.info("CameraManager: Loading shared AI models...")
        self.vehicle_detector = VehicleDetector(
            model_path=config.vehicle_model,
            confidence_threshold=config.vehicle_confidence,
        )
        self.plate_detector = LicensePlateDetector(
            model_path=config.plate_model,
            confidence_threshold=config.plate_confidence,
        )
        self.ocr_engine = get_ocr_engine()
        logger.info("CameraManager: Shared AI models ready.")

        # Pipelines registry: camera_id -> VisionPipeline
        self.pipelines: Dict[str, VisionPipeline] = {}
        self._init_default_cameras()

    def _get_default_videos(self) -> Tuple[str, str]:
        videos = sorted(glob.glob(os.path.join(self.videos_dir, "*.mp4")))
        entry_vid = ""
        exit_vid = ""
        if len(videos) > 0:
            entry_vid = videos[0]
            exit_vid = videos[min(3, len(videos) - 1)]
        return entry_vid, exit_vid

    def _init_default_cameras(self) -> None:
        entry_vid, exit_vid = self._get_default_videos()

        # Camera 1: Entry
        cam_entry_id = "cam-entry"
        stream_entry = FileCameraStream(
            camera_id=cam_entry_id,
            file_path=entry_vid,
            target_fps=self.cfg.target_fps,
            loop=True,
        ) if entry_vid else RTSPCameraStream(cam_entry_id, self.cfg.rtsp_url or "rtsp://localhost:554/entry", self.cfg.target_fps)

        self.pipelines[cam_entry_id] = VisionPipeline(
            config=self.cfg,
            camera_stream=stream_entry,
            backend_client=self.backend,
            vehicle_detector=self.vehicle_detector,
            plate_detector=self.plate_detector,
            ocr_engine=self.ocr_engine,
            camera_id=cam_entry_id,
            camera_name="Kirish Yo'lagi Kamerasi (Lane 1)",
            camera_direction="ENTRY",
            current_video_file=os.path.basename(entry_vid) if entry_vid else "",
        )

        # Camera 2: Exit
        cam_exit_id = "cam-exit"
        stream_exit = FileCameraStream(
            camera_id=cam_exit_id,
            file_path=exit_vid,
            target_fps=self.cfg.target_fps,
            loop=True,
        ) if exit_vid else RTSPCameraStream(cam_exit_id, self.cfg.rtsp_url or "rtsp://localhost:554/exit", self.cfg.target_fps)

        self.pipelines[cam_exit_id] = VisionPipeline(
            config=self.cfg,
            camera_stream=stream_exit,
            backend_client=self.backend,
            vehicle_detector=self.vehicle_detector,
            plate_detector=self.plate_detector,
            ocr_engine=self.ocr_engine,
            camera_id=cam_exit_id,
            camera_name="Chiqish Yo'lagi Kamerasi (Lane 2)",
            camera_direction="EXIT",
            current_video_file=os.path.basename(exit_vid) if exit_vid else "",
        )

    async def start_all(self) -> None:
        for cam_id, pipeline in self.pipelines.items():
            await pipeline.start()
        logger.info(f"CameraManager started {len(self.pipelines)} camera pipelines.")

    async def stop_all(self) -> None:
        for cam_id, pipeline in self.pipelines.items():
            await pipeline.stop()
        logger.info("CameraManager stopped all camera pipelines.")

    def get_pipeline(self, camera_id: str) -> Optional[VisionPipeline]:
        if camera_id in self.pipelines:
            return self.pipelines[camera_id]
        # Return entry pipeline by default if not found
        return self.pipelines.get("cam-entry")

    def list_available_videos(self) -> List[Dict[str, Any]]:
        videos = sorted(glob.glob(os.path.join(self.videos_dir, "*.mp4")))
        results = []
        for v in videos:
            name = os.path.basename(v)
            size_mb = round(os.path.getsize(v) / (1024 * 1024), 2)
            results.append({
                "filename": name,
                "full_path": v,
                "size_mb": size_mb,
            })
        return results

    def list_cameras_info(self) -> List[Dict[str, Any]]:
        cameras_info = []
        for cam_id, pipeline in self.pipelines.items():
            stream = pipeline.stream
            status = stream.status.value if stream else "OFFLINE"
            fps = stream.last_fps if stream else 0.0
            cameras_info.append({
                "id": pipeline.camera_id,
                "name": pipeline.camera_name,
                "direction": pipeline.camera_direction,
                "parking_id": self.cfg.parking_id,
                "status": status,
                "fps": round(fps, 1),
                "target_fps": pipeline.cfg.target_fps,
                "state": "ACTIVE" if status == "ONLINE" else "IDLE",
                "current_video": pipeline.current_video_file,
                "last_recognition": pipeline.last_confirmed_plate or "",
            })
        return cameras_info

    async def switch_video(self, camera_id: str, video_filename: str) -> bool:
        """Switch video file source for a specific camera pipeline."""
        pipeline = self.pipelines.get(camera_id)
        if not pipeline:
            return False

        full_path = os.path.join(self.videos_dir, video_filename)
        if not os.path.exists(full_path):
            logger.error(f"Video file not found for switch: {full_path}")
            return False

        # Stop old stream
        old_stream = pipeline.stream
        old_stream.stop()

        # Create new stream
        new_stream = FileCameraStream(
            camera_id=camera_id,
            file_path=full_path,
            target_fps=self.cfg.target_fps,
            loop=True,
        )
        new_stream.start()
        pipeline.stream = new_stream
        pipeline.current_video_file = video_filename
        logger.info(f"Camera {camera_id} switched video source to: {video_filename}")
        return True

