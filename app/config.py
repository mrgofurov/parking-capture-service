import os
from typing import Optional
from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore"
    )

    # Environment
    app_env: str = Field(default="production", alias="APP_ENV")
    service_port: int = Field(default=8086, alias="SERVICE_PORT")
    log_level: str = Field(default="INFO", alias="LOG_LEVEL")

    # Camera Configuration
    camera_id: str = Field(default="entry-01", alias="CAMERA_ID")
    camera_name: str = Field(default="Entry Camera 01", alias="CAMERA_NAME")
    camera_direction: str = Field(default="entry", alias="CAMERA_DIRECTION")  # "entry" or "exit"
    parking_id: str = Field(default="main-parking", alias="PARKING_ID")

    # Video Source ("rtsp" or "file")
    video_source: str = Field(default="rtsp", alias="VIDEO_SOURCE")
    rtsp_url: str = Field(default="", alias="RTSP_URL")
    video_file: str = Field(default="", alias="VIDEO_FILE")
    target_fps: int = Field(default=10, alias="TARGET_FPS")
    max_frame_queue: int = Field(default=50, alias="MAX_FRAME_QUEUE")

    # Synthetic / Replay Mode
    synthetic_mode: bool = Field(default=False, alias="SYNTHETIC_MODE")

    # ROI (Region of Interest)
    roi_enabled: bool = Field(default=False, alias="ROI_ENABLED")
    roi_x1: int = Field(default=0, alias="ROI_X1")
    roi_y1: int = Field(default=0, alias="ROI_Y1")
    roi_x2: int = Field(default=1920, alias="ROI_X2")
    roi_y2: int = Field(default=1080, alias="ROI_Y2")

    # Vision & Models
    vehicle_model: str = Field(default="yolov8n.pt", alias="VEHICLE_MODEL")
    plate_model: str = Field(default="yolov8n.pt", alias="PLATE_MODEL")
    vehicle_confidence: float = Field(default=0.40, alias="VEHICLE_CONFIDENCE")
    plate_confidence: float = Field(default=0.45, alias="PLATE_CONFIDENCE")
    ocr_min_confidence: float = Field(default=0.60, alias="OCR_MIN_CONFIDENCE")

    # Plate Quality & Cropping
    min_plate_width: int = Field(default=80, alias="MIN_PLATE_WIDTH")
    min_plate_height: int = Field(default=20, alias="MIN_PLATE_HEIGHT")
    min_sharpness_score: float = Field(default=25.0, alias="MIN_SHARPNESS_SCORE")
    ocr_interval_ms: int = Field(default=200, alias="OCR_INTERVAL_MS")

    # Multi-Frame Consensus & Tracking
    min_consensus_count: int = Field(default=2, alias="MIN_CONSENSUS_COUNT")
    min_final_confidence: float = Field(default=0.78, alias="MIN_FINAL_CONFIDENCE")
    track_timeout_seconds: float = Field(default=5.0, alias="TRACK_TIMEOUT_SECONDS")
    tracker_type: str = Field(default="bytetrack", alias="TRACKER_TYPE")  # "bytetrack" or "botsort"

    # Event Generation & Deduplication
    event_cooldown_seconds: int = Field(default=10, alias="EVENT_COOLDOWN_SECONDS")
    storage_best_frame: bool = Field(default=True, alias="STORAGE_BEST_FRAME")
    upload_dir: str = Field(default="./snapshots", alias="UPLOAD_DIR")
    capture_retention_days: int = Field(default=7, alias="CAPTURE_RETENTION_DAYS")

    # Backend Integration
    backend_url: str = Field(default="http://localhost:8085", alias="BACKEND_URL")
    backend_event_path: str = Field(default="/api/v1/anpr/event", alias="BACKEND_EVENT_PATH")
    backend_token: Optional[str] = Field(default="", alias="BACKEND_TOKEN")
    backend_timeout_seconds: float = Field(default=4.0, alias="BACKEND_TIMEOUT_SECONDS")
    backend_max_retries: int = Field(default=3, alias="BACKEND_MAX_RETRIES")
    backend_queue_size: int = Field(default=200, alias="BACKEND_QUEUE_SIZE")

    # Redis (Optional)
    redis_url: Optional[str] = Field(default=None, alias="REDIS_URL")


settings = Settings()
