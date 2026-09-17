package domain

import (
	"image"
	"time"
)

type Direction string

const (
	DirectionEntry Direction = "ENTRY"
	DirectionExit  Direction = "EXIT"
)

type CameraStatus string

const (
	CameraOnline       CameraStatus = "ONLINE"
	CameraOffline      CameraStatus = "OFFLINE"
	CameraReconnecting CameraStatus = "RECONNECTING"
)

type CameraConfig struct {
	ID          string       `json:"id"`
	Name        string       `json:"name"`
	ParkingID   string       `json:"parking_id"`
	Direction   Direction    `json:"direction"`
	RTSPURL     string       `json:"rtsp_url"`
	Enabled     bool         `json:"enabled"`
	FPS         int          `json:"fps"`
	Status      CameraStatus `json:"status"`
	BarrierID   string       `json:"barrier_id"`
	VirtualLine VirtualLine  `json:"virtual_line"`
}

type VirtualLine struct {
	X1 int `json:"x1"`
	Y1 int `json:"y1"`
	X2 int `json:"x2"`
	Y2 int `json:"y2"`
}

type Frame struct {
	CameraID   string
	SequenceID uint64
	Timestamp  time.Time
	Image      image.Image
	JPEGBytes  []byte
	Width      int
	Height     int
}

type BoundingBox struct {
	X      int `json:"x"`
	Y      int `json:"y"`
	Width  int `json:"width"`
	Height int `json:"height"`
}

type VehicleObservation struct {
	CameraID    string      `json:"camera_id"`
	TrackingID  string      `json:"tracking_id"`
	BoundingBox BoundingBox `json:"bounding_box"`
	Confidence  float64     `json:"confidence"`
	Timestamp   time.Time   `json:"timestamp"`
	CrossedLine bool        `json:"crossed_line"`
}

type PlateObservation struct {
	CameraID       string      `json:"camera_id"`
	TrackingID     string      `json:"tracking_id"`
	RawText        string      `json:"raw_text"`
	NormalizedText string      `json:"normalized_text"`
	Confidence     float64     `json:"confidence"`
	Sharpness      float64     `json:"sharpness"`
	Brightness     float64     `json:"brightness"`
	Contrast       float64     `json:"contrast"`
	BoundingBox    BoundingBox `json:"bounding_box"`
	Timestamp      time.Time   `json:"timestamp"`
}

type RecognitionResult struct {
	PlateNumber       string    `json:"plate_number"`
	RawText           string    `json:"raw_text"`
	OverallConfidence float64   `json:"overall_confidence"`
	SampleCount       int       `json:"sample_count"`
	SnapshotPath      string    `json:"snapshot_path"`
	SnapshotBase64    string    `json:"snapshot_base64"`
	Timestamp         time.Time `json:"timestamp"`
}

type StateType string

const (
	StateIdle              StateType = "IDLE"
	StateVehicleDetected   StateType = "VEHICLE_DETECTED"
	StateTracking          StateType = "TRACKING"
	StatePlateCandidate    StateType = "PLATE_CANDIDATE"
	StateRecognizing       StateType = "RECOGNIZING"
	StatePlateConfirmed    StateType = "PLATE_CONFIRMED"
	StateEventSent         StateType = "EVENT_SENT"
	StateWaitingForPassage StateType = "WAITING_FOR_PASSAGE"
	StatePassed            StateType = "PASSED"
	StateCooldown          StateType = "COOLDOWN"
)

type BarrierStatus string

const (
	BarrierOpen    BarrierStatus = "OPEN"
	BarrierClosed  BarrierStatus = "CLOSED"
	BarrierUnknown BarrierStatus = "UNKNOWN"
	BarrierError   BarrierStatus = "ERROR"
)

type BackendEvent struct {
	RequestID      string    `json:"request_id"`
	EventID        string    `json:"event_id"`
	CameraID       string    `json:"camera_id"`
	ParkingID      string    `json:"parking_id"`
	Direction      Direction `json:"direction"`
	PlateNumber    string    `json:"plate_number"`
	Confidence     float64   `json:"confidence"`
	Timestamp      time.Time `json:"timestamp"`
	SnapshotBase64 string    `json:"snapshot_base64,omitempty"`
}

type BackendResponse struct {
	Success bool `json:"success"`
	Data    struct {
		BarrierOpened   bool    `json:"barrier_opened"`
		RequiresPayment bool    `json:"requires_payment"`
		AmountDue       float64 `json:"amount_due"`
		Message         string  `json:"message"`
	} `json:"data"`
	Message string `json:"message"`
}
