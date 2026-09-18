import asyncio
from typing import Any, Dict
import cv2
from fastapi import APIRouter, File, HTTPException, Request, Response, UploadFile
from fastapi.responses import StreamingResponse
import numpy as np
from pydantic import BaseModel

from app.parking.event import ParkingEvent
from app.utils.image import encode_image_to_jpeg
from app.vision.normalizer import normalize_plate
from app.vision.preprocessing import preprocess_plate
from app.vision.quality import compute_plate_quality

router = APIRouter(prefix="/api/v1", tags=["Diagnostics & Control"])


class InjectPlateRequest(BaseModel):
    plate_number: str


@router.get("/cameras")
async def list_cameras(request: Request) -> Dict[str, Any]:
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
                "fps": fps,
                "target_fps": cfg.target_fps,
            }
        ],
    }


@router.get("/cameras/{camera_id}/snapshot")
async def get_camera_snapshot(camera_id: str, request: Request) -> Response:
    pipeline = getattr(request.app.state, "pipeline", None)
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
    pipeline = getattr(request.app.state, "pipeline", None)
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


@router.post("/cameras/{camera_id}/inject-plate")
async def inject_test_plate(
    camera_id: str, payload: InjectPlateRequest, request: Request
) -> Dict[str, Any]:
    """Inject a test plate for synthetic end-to-end testing."""
    if not payload.plate_number:
        raise HTTPException(status_code=400, detail="plate_number is required")

    cfg = request.app.state.config
    pipeline = getattr(request.app.state, "pipeline", None)
    if not pipeline:
        raise HTTPException(status_code=500, detail="Pipeline not initialized")

    norm = normalize_plate(payload.plate_number)
    event = ParkingEvent(
        camera_id=camera_id,
        direction=cfg.camera_direction,
        plate_number=norm.normalized if norm.is_valid else payload.plate_number,
        confidence=0.95,
        track_id="inject-sim",
        sample_count=3,
    )

    # Dispatch to backend
    await pipeline.backend.send_event(event)

    return {
        "success": True,
        "message": f"Injected plate {event.plate_number} and triggered backend event",
        "data": event.to_backend_payload(),
    }


@router.post("/cameras/{camera_id}/test-image")
async def test_image_ocr(
    camera_id: str, request: Request, image: UploadFile = File(...)
) -> Dict[str, Any]:
    """Upload an image to test quality check, preprocessing, and OCR directly."""
    pipeline = getattr(request.app.state, "pipeline", None)
    if not pipeline:
        raise HTTPException(status_code=500, detail="Pipeline not initialized")

    contents = await image.read()
    nparr = np.frombuffer(contents, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    if img is None:
        raise HTTPException(status_code=400, detail="Invalid image file")

    quality = compute_plate_quality(img)
    preprocessed = preprocess_plate(img)
    ocr_results = pipeline.ocr_engine.recognize(preprocessed, camera_id=camera_id)

    raw_text = ocr_results[0].text if ocr_results else ""
    confidence = ocr_results[0].confidence if ocr_results else 0.0
    norm = normalize_plate(raw_text)

    return {
        "success": True,
        "raw_ocr": raw_text,
        "ocr_confidence": round(confidence, 4),
        "normalized_plate": norm.normalized,
        "is_valid": norm.is_valid,
        "plate_type": norm.plate_type,
        "quality": {
            "sharpness": quality.sharpness,
            "brightness": quality.brightness,
            "contrast": quality.contrast,
            "is_acceptable": quality.is_acceptable,
            "rejection_reason": quality.rejection_reason,
        },
    }
