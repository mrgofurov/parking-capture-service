import asyncio
from contextlib import asynccontextmanager
from typing import AsyncGenerator
import torch
import uvicorn
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api.diagnostic import router as diagnostic_router
from app.api.health import router as health_router
from app.api.metrics import router as metrics_router
from app.backend.client import BackendClient
from app.camera.file import FileCameraStream
from app.camera.rtsp import RTSPCameraStream
from app.config import settings
from app.utils.image import cleanup_expired_captures
from app.utils.logging import logger, setup_logger
from app.vision.pipeline import VisionPipeline


async def retention_cleanup_loop(upload_dir: str, retention_days: int) -> None:
    """Run daily cleanup of expired snapshots."""
    while True:
        try:
            await asyncio.sleep(86400)  # Run once every 24 hours
            cleanup_expired_captures(upload_dir, retention_days)
        except asyncio.CancelledError:
            break
        except Exception as e:
            logger.error(f"Error in retention cleanup worker: {e}")


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    # 1. Setup Logging
    setup_logger(level=settings.log_level)
    logger.info("==================================================")
    logger.info("     Smart Parking Capture Service (Python AI)    ")
    logger.info("     YOLO + ByteTrack + PaddleOCR Multi-Frame     ")
    logger.info("==================================================")

    # 2. Check Hardware Acceleration
    gpu_available = torch.cuda.is_available()
    device_name = torch.cuda.get_device_name(0) if gpu_available else "CPU"
    logger.info(f"Inference Device: {device_name} (CUDA available: {gpu_available})")
    logger.info(
        f"Camera ID: {settings.camera_id}, Direction: {settings.camera_direction}, "
        f"Source: {settings.video_source}, Target FPS: {settings.target_fps}"
    )

    # 3. Create Backend Client
    backend_client = BackendClient(
        base_url=settings.backend_url,
        event_path=settings.backend_event_path,
        token=settings.backend_token,
        timeout=settings.backend_timeout_seconds,
        max_retries=settings.backend_max_retries,
        queue_size=settings.backend_queue_size,
    )
    await backend_client.start()

    # 4. Create Camera Stream
    if settings.video_source.lower() == "file":
        stream = FileCameraStream(
            camera_id=settings.camera_id,
            file_path=settings.video_file,
            target_fps=settings.target_fps,
            loop=True,
        )
    else:
        stream = RTSPCameraStream(
            camera_id=settings.camera_id,
            rtsp_url=settings.rtsp_url,
            target_fps=settings.target_fps,
            max_queue_size=settings.max_frame_queue,
        )

    # 5. Initialize Vision Pipeline
    pipeline = VisionPipeline(
        config=settings,
        camera_stream=stream,
        backend_client=backend_client,
    )
    await pipeline.start()

    # Save to app state
    app.state.config = settings
    app.state.pipeline = pipeline
    app.state.backend = backend_client

    # 6. Start background snapshot retention cleaner
    cleanup_task = asyncio.create_task(
        retention_cleanup_loop(settings.upload_dir, settings.capture_retention_days)
    )

    yield

    # Graceful Shutdown
    logger.info("Shutting down Smart Parking Capture Service...")
    cleanup_task.cancel()
    await pipeline.stop()
    await backend_client.stop()
    logger.info("Smart Parking Capture Service stopped cleanly.")


def create_app() -> FastAPI:
    app = FastAPI(
        title="Smart Parking Capture Service",
        description="Edge AI ANPR service using YOLO and PaddleOCR with Multi-Frame Consensus",
        version="2.0.0",
        lifespan=lifespan,
    )

    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    app.include_router(health_router)
    app.include_router(metrics_router)
    app.include_router(diagnostic_router)

    return app


app = create_app()

if __name__ == "__main__":
    uvicorn.run(
        "app.main:app",
        host="0.0.0.0",
        port=settings.service_port,
        log_level=settings.log_level.lower(),
    )
