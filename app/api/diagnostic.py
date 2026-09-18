import asyncio
from typing import Any, Dict, List, Optional
import cv2
from fastapi import APIRouter, File, HTTPException, Request, Response, UploadFile
from fastapi.responses import StreamingResponse
import numpy as np
from pydantic import BaseModel

from app.camera.file import FileCameraStream
from app.camera.rtsp import RTSPCameraStream
from app.parking.event import ParkingEvent
from app.utils.image import encode_image_to_jpeg
from app.vision.normalizer import normalize_plate
from app.vision.preprocessing import preprocess_plate
from app.vision.quality import compute_plate_quality

router = APIRouter(prefix="/api/v1", tags=["Diagnostics & Control"])


class SwitchVideoRequest(BaseModel):
    video_filename: str


class CameraConfigRequest(BaseModel):
    mode: str = "VIDEO_FILE"  # "RTSP" or "VIDEO_FILE"
    rtsp_url: Optional[str] = None
    video_file_path: Optional[str] = None
    fps: Optional[int] = 10


class TestRtspRequest(BaseModel):
    rtsp_url: str


@router.get("/cameras")
async def list_cameras(request: Request) -> Dict[str, Any]:
    manager = getattr(request.app.state, "camera_manager", None)
    if manager:
        return {
            "success": True,
            "data": manager.list_cameras_info(),
        }

    # Fallback to single pipeline if manager not present
    cfg = request.app.state.config
    pipeline = getattr(request.app.state, "pipeline", None)
    status = pipeline.stream.status.value if pipeline and pipeline.stream else "OFFLINE"
    fps = pipeline.stream.last_fps if pipeline and pipeline.stream else 0.0

    return {
        "success": True,
        "data": [
            {
                "id": cfg.camera_id,
                "name": cfg.camera_name,
                "direction": cfg.camera_direction,
                "parking_id": cfg.parking_id,
                "video_source": cfg.video_source,
                "status": status,
                "fps": round(fps, 1),
                "target_fps": cfg.target_fps,
                "state": "ACTIVE" if status == "ONLINE" else "IDLE",
            }
        ],
    }


@router.get("/videos")
async def list_videos(request: Request) -> Dict[str, Any]:
    """List all available test videos in test-videos/ folder."""
    manager = getattr(request.app.state, "camera_manager", None)
    if not manager:
        return {"success": True, "data": []}
    return {
        "success": True,
        "data": manager.list_available_videos(),
    }


@router.get("/cameras/{camera_id}/snapshot")
async def get_camera_snapshot(camera_id: str, request: Request) -> Response:
    manager = getattr(request.app.state, "camera_manager", None)
    pipeline = manager.get_pipeline(camera_id) if manager else getattr(request.app.state, "pipeline", None)

    if not pipeline or pipeline.latest_frame is None:
        raise HTTPException(status_code=404, detail="Snapshot not available")

    jpeg_bytes = encode_image_to_jpeg(pipeline.latest_frame, quality=85)
    return Response(
        content=jpeg_bytes,
        media_type="image/jpeg",
        headers={"Cache-Control": "no-cache, no-store, must-revalidate"},
    )


@router.get("/cameras/{camera_id}/stream")
async def stream_camera(camera_id: str, request: Request) -> StreamingResponse:
    manager = getattr(request.app.state, "camera_manager", None)
    pipeline = manager.get_pipeline(camera_id) if manager else getattr(request.app.state, "pipeline", None)

    if not pipeline:
        raise HTTPException(status_code=404, detail="Camera pipeline not initialized")

    frame_queue = pipeline.subscribe_mjpeg()

    async def frame_generator():
        try:
            while True:
                jpeg_bytes = await frame_queue.get()
                yield (
                    b"--frame\r\n"
                    b"Content-Type: image/jpeg\r\n\r\n" + jpeg_bytes + b"\r\n"
                )
        except asyncio.CancelledError:
            pass
        finally:
            pipeline.unsubscribe_mjpeg(frame_queue)

    return StreamingResponse(
        frame_generator(),
        media_type="multipart/x-mixed-replace; boundary=frame",
        headers={"Cache-Control": "no-cache"},
    )


@router.post("/cameras/{camera_id}/switch-video")
async def switch_camera_video(
    camera_id: str, payload: SwitchVideoRequest, request: Request
) -> Dict[str, Any]:
    """Switch video source for a specific camera pipeline."""
    manager = getattr(request.app.state, "camera_manager", None)
    if not manager:
        raise HTTPException(status_code=500, detail="CameraManager not initialized")

    success = await manager.switch_video(camera_id, payload.video_filename)
    if not success:
        raise HTTPException(status_code=400, detail=f"Failed to switch video to {payload.video_filename}")

    return {
        "success": True,
        "message": f"Camera {camera_id} video successfully switched to {payload.video_filename}",
        "video": payload.video_filename,
    }


@router.put("/cameras/{camera_id}/config")
async def configure_camera(
    camera_id: str, payload: CameraConfigRequest, request: Request
) -> Dict[str, Any]:
    """Configure or switch video / RTSP stream for a camera."""
    manager = getattr(request.app.state, "camera_manager", None)
    if not manager:
        raise HTTPException(status_code=500, detail="CameraManager not initialized")

    pipeline = manager.get_pipeline(camera_id)
    if not pipeline:
        raise HTTPException(status_code=404, detail="Camera not found")

    fps = payload.fps or 10

    if payload.mode == "RTSP" and payload.rtsp_url:
        import re
        url = payload.rtsp_url.strip()
        m = re.match(r"^(rtsp://)([^/:]+)(/.*)?$", url)
        if m:
            # If standard 554 not accessible, use 8080
            host = m.group(2)
            path = m.group(3) or ""
            url = f"rtsp://{host}:8080{path}"

        pipeline.stream.stop()
        new_stream = RTSPCameraStream(
            camera_id=camera_id,
            rtsp_url=url,
            target_fps=fps,
        )
        new_stream.start()
        pipeline.stream = new_stream
        pipeline.current_video_file = url
        return {"success": True, "message": f"Camera {camera_id} connected to RTSP stream: {url}"}

    elif payload.video_file_path:
        filename = payload.video_file_path.split("/")[-1]
        success = await manager.switch_video(camera_id, filename)
        if success:
            return {"success": True, "message": f"Camera {camera_id} switched to {filename}"}

    return {"success": True, "message": "Camera config updated"}


@router.post("/cameras/test-rtsp")
async def test_rtsp_connection(payload: TestRtspRequest) -> Dict[str, Any]:
    """Test connection to RTSP stream, with smart IP Webcam port 8080 fallback."""
    raw_url = payload.rtsp_url.strip()
    urls_to_try = [raw_url]

    # If user provided IP Webcam url without port, e.g. rtsp://10.80.113.163/h264_ulaw.sdp
    import re
    m = re.match(r"^(rtsp://)([^/:]+)(/.*)?$", raw_url)
    if m:
        host = m.group(2)
        path = m.group(3) or ""
        urls_to_try.append(f"rtsp://{host}:8080{path}")
        urls_to_try.append(f"http://{host}:8080/video")

    for u in urls_to_try:
        try:
            cap = cv2.VideoCapture(u)
            if cap.isOpened():
                ret, frame = cap.read()
                cap.release()
                if ret and frame is not None:
                    msg = "RTSP ulanishi muvaffaqiyatli! Video oqimi qabul qilindi."
                    if u != raw_url:
                        msg = f"RTSP muvaffaqiyatli ulandi! (IP Webcam port 8080 qo'shildi: {u})"
                    return {"success": True, "message": msg, "effective_url": u}
        except Exception:
            pass

    return {
        "success": False,
        "message": "RTSP serveriga ulanib bo'lmadi. Telefon IP Webcam uchun portni tekshiring: masalan, rtsp://10.80.113.163:8080/h264_ulaw.sdp",
    }
