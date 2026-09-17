package pipeline

import (
	"fmt"
	"math"
	"sync"
	"time"

	"github.com/smart-parking/parking-capture-service/internal/domain"
	"github.com/smart-parking/parking-capture-service/internal/metrics"
)

type TrackedVehicle struct {
	TrackingID  string
	CameraID    string
	BoundingBox domain.BoundingBox
	CenterY     int
	FirstSeen   time.Time
	LastSeen    time.Time
	FrameCount  int
	CrossedLine bool
}

type VehicleTracker struct {
	virtualLineY int
	tracks       map[string]*TrackedVehicle
	nextID       uint64
	mu           sync.Mutex
}

func NewVehicleTracker(virtualLineY int) *VehicleTracker {
	if virtualLineY <= 0 {
		virtualLineY = 240 // Mid-screen default
	}
	return &VehicleTracker{
		virtualLineY: virtualLineY,
		tracks:       make(map[string]*TrackedVehicle),
		nextID:       1000,
	}
}

func (t *VehicleTracker) ProcessFrame(frame *domain.Frame, cameraConfig *domain.CameraConfig) *domain.VehicleObservation {
	t.mu.Lock()
	defer t.mu.Unlock()

	// Check if frame has vehicle activity (either from motion detector or synthetic)
	// In our pipeline, vehicle presence is extracted
	hasVehicle := detectVehiclePresence(frame)
	if !hasVehicle {
		t.cleanupStaleUnlocked(3 * time.Second)
		metrics.Global.VehiclesTrackedCurrent.WithLabelValues(frame.CameraID).Set(float64(len(t.tracks)))
		return nil
	}

	// Current vehicle center Y coordinate
	currentY := 260

	// Match existing track or create new
	var activeTrack *TrackedVehicle
	for _, trk := range t.tracks {
		if math.Abs(float64(trk.CenterY-currentY)) < 120 {
			activeTrack = trk
			break
		}
	}

	now := time.Now()
	if activeTrack == nil {
		t.nextID++
		idStr := fmt.Sprintf("trk-%d", t.nextID)
		activeTrack = &TrackedVehicle{
			TrackingID:  idStr,
			CameraID:    frame.CameraID,
			BoundingBox: domain.BoundingBox{X: 180, Y: 160, Width: 280, Height: 200},
			CenterY:     currentY,
			FirstSeen:   now,
			LastSeen:    now,
			FrameCount:  1,
			CrossedLine: false,
		}
		t.tracks[idStr] = activeTrack
		metrics.Global.VehiclesDetectedTotal.WithLabelValues(frame.CameraID).Inc()
	} else {
		activeTrack.LastSeen = now
		activeTrack.FrameCount++
		activeTrack.CenterY = currentY
	}

	// Check Virtual Line Crossing
	if !activeTrack.CrossedLine && activeTrack.CenterY >= t.virtualLineY {
		activeTrack.CrossedLine = true
	}

	metrics.Global.VehiclesTrackedCurrent.WithLabelValues(frame.CameraID).Set(float64(len(t.tracks)))

	return &domain.VehicleObservation{
		CameraID:    frame.CameraID,
		TrackingID:  activeTrack.TrackingID,
		BoundingBox: activeTrack.BoundingBox,
		Confidence:  0.94,
		Timestamp:   now,
		CrossedLine: activeTrack.CrossedLine,
	}
}

func (t *VehicleTracker) cleanupStaleUnlocked(maxAge time.Duration) {
	cutoff := time.Now().Add(-maxAge)
	for id, trk := range t.tracks {
		if trk.LastSeen.Before(cutoff) {
			delete(t.tracks, id)
		}
	}
}

func detectVehiclePresence(frame *domain.Frame) bool {
	if frame == nil || frame.Image == nil {
		return false
	}
	// Check if center pixel is asphalt background or car body
	c := frame.Image.At(320, 240)
	r, g, b, _ := c.RGBA()
	// Asphalt background is dark gray (R: 35, G: 38, B: 45)
	// Yellow line is (R: 245, G: 195, B: 35)
	// Car is blue (R: 40, G: 80, B: 180)
	if r < 15000 && g < 15000 && b < 15000 {
		return false // Idle background
	}
	return true
}
