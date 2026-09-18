from prometheus_client import Counter, Histogram, Gauge

# Frame & Stream Metrics
FRAMES_RECEIVED = Counter(
    "frames_received_total",
    "Total number of video frames received from camera stream",
    ["camera_id"],
)

FRAMES_PROCESSED = Counter(
    "frames_processed_total",
    "Total number of video frames processed through vision pipeline",
    ["camera_id"],
)

STREAM_RECONNECTS = Counter(
    "stream_reconnects_total",
    "Total number of camera stream reconnect attempts",
    ["camera_id"],
)

CAMERA_CONNECTED = Gauge(
    "camera_connected",
    "Camera connection status (1 = connected, 0 = disconnected)",
    ["camera_id"],
)

PROCESSING_FPS = Gauge(
    "processing_fps",
    "Current effective processing FPS of vision pipeline",
    ["camera_id"],
)

# Vision Detection Metrics
VEHICLE_DETECTIONS = Counter(
    "vehicle_detections_total",
    "Total number of vehicle detections",
    ["camera_id", "vehicle_class"],
)

PLATE_DETECTIONS = Counter(
    "plate_detections_total",
    "Total number of license plate detections",
    ["camera_id"],
)

# OCR Metrics
OCR_ATTEMPTS = Counter(
    "ocr_attempts_total",
    "Total number of OCR recognition attempts",
    ["camera_id"],
)

OCR_SUCCESS = Counter(
    "ocr_success_total",
    "Total number of successful OCR recognitions",
    ["camera_id"],
)

# Parking Event Metrics
PARKING_EVENTS = Counter(
    "parking_events_total",
    "Total number of validated parking events emitted",
    ["camera_id", "direction"],
)

DUPLICATE_EVENTS = Counter(
    "duplicate_events_total",
    "Total number of duplicate events suppressed by cooldown",
    ["camera_id"],
)

# Backend Client Metrics
BACKEND_REQUESTS = Counter(
    "backend_requests_total",
    "Total number of requests sent to parking backend",
    ["status"],
)

BACKEND_REQUEST_ERRORS = Counter(
    "backend_request_errors_total",
    "Total number of errors encountered when sending to backend",
    ["error_type"],
)

# Latency Histograms
PROCESSING_LATENCY = Histogram(
    "processing_latency_seconds",
    "Latency of processing a single frame through vision pipeline",
    ["camera_id"],
    buckets=[0.01, 0.025, 0.05, 0.075, 0.1, 0.25, 0.5, 1.0, 2.5],
)

OCR_LATENCY = Histogram(
    "ocr_latency_seconds",
    "Latency of PaddleOCR inference on license plate crop",
    ["camera_id"],
    buckets=[0.01, 0.025, 0.05, 0.1, 0.2, 0.3, 0.5, 1.0],
)
