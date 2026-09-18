from datetime import datetime, timezone
from typing import Dict, Any, Optional
from fastapi import APIRouter, Request
import torch

from app.camera.base import CameraStatus

router = APIRouter(tags=["Health"])


@router.get("/health")
async def health_check(request: Request) -> Dict[str, Any]:
    pipeline = getattr(request.app.state, "pipeline", None)

    camera_connected = False
    last_frame_at = None
    fps = 0.0

    if pipeline and pipeline.stream:
        camera_connected = pipeline.stream.status == CameraStatus.ONLINE
        fps = pipeline.stream.last_fps
        if pipeline.stream.last_frame_time > 0:
            last_frame_at = datetime.fromtimestamp(
                pipeline.stream.last_frame_time, timezone.utc
            ).isoformat()

    gpu_available = False
    try:
        gpu_available = torch.cuda.is_available()
    except Exception:
        pass

    return {
        "status": "ok",
        "camera_connected": camera_connected,
        "last_frame_at": last_frame_at,
        "gpu_available": gpu_available,
        "fps": fps,
    }


@router.get("/ready")
async def readiness_check() -> Dict[str, str]:
    return {"status": "ready"}


@router.get("/health/live")
async def liveness_check() -> Dict[str, str]:
    return {"status": "UP", "service": "parking-capture-service"}


@router.get("/health/ready")
async def health_ready_check() -> Dict[str, str]:
    return {"status": "READY", "service": "parking-capture-service"}
