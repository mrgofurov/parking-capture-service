from datetime import datetime, timezone
from typing import Optional
from pydantic import BaseModel, Field


import re

UUID_REGEX = re.compile(r"^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$", re.I)


class ParkingEvent(BaseModel):
    camera_id: str = Field(..., description="Unique camera identifier, e.g. entry-01")
    direction: str = Field(..., description="Direction: entry or exit")
    plate_number: str = Field(..., description="Confirmed vehicle license plate number")
    confidence: float = Field(..., description="Overall consensus confidence (0.0 to 1.0 or percentage)")
    detected_at: str = Field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat(),
        description="ISO 8601 timestamp of event detection",
    )
    track_id: Optional[str] = Field(default=None, description="Vehicle tracking ID")
    sample_count: Optional[int] = Field(default=1, description="Number of consensus frames")
    snapshot_path: Optional[str] = Field(default=None, description="Local path to saved snapshot")
    snapshot_base64: Optional[str] = Field(default=None, description="Base64 encoded best frame")

    def to_backend_payload(self) -> dict:
        """Format payload according to parking-backend requirements."""
        # parking-backend expects uppercase "ENTRY" or "EXIT"
        direction_upper = self.direction.upper() if self.direction else "ENTRY"
        # confidence as percentage if <= 1.0
        conf = self.confidence * 100.0 if self.confidence <= 1.0 else self.confidence

        payload = {
            "plate_number": self.plate_number.upper(),
            "direction": direction_upper,
            "confidence": round(conf, 2),
            "timestamp": self.detected_at,
            "detected_at": self.detected_at,
        }

        # parking-backend parses CameraID as *uuid.UUID, so only include if valid UUID
        if self.camera_id and UUID_REGEX.match(self.camera_id):
            payload["camera_id"] = self.camera_id

        if self.snapshot_base64:
            payload["snapshot_base64"] = self.snapshot_base64

        return payload
