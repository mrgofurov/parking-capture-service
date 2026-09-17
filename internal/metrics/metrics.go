package metrics

import (
	"github.com/prometheus/client_golang/prometheus"
	"github.com/prometheus/client_golang/prometheus/promauto"
)

type Metrics struct {
	CameraConnected              *prometheus.GaugeVec
	CameraReconnectTotal         *prometheus.CounterVec
	FramesReceivedTotal          *prometheus.CounterVec
	FramesProcessedTotal         *prometheus.CounterVec
	FramesDroppedTotal           *prometheus.CounterVec
	VehiclesDetectedTotal        *prometheus.CounterVec
	VehiclesTrackedCurrent       *prometheus.GaugeVec
	PlateDetectionsTotal         *prometheus.CounterVec
	PlateRecognitionsTotal       *prometheus.CounterVec
	PlateRecognitionSuccessTotal *prometheus.CounterVec
	PlateRecognitionFailedTotal  *prometheus.CounterVec
	RecognitionLatencyMs         *prometheus.HistogramVec
	BarrierOpenTotal             *prometheus.CounterVec
	BarrierCloseTotal            *prometheus.CounterVec
	BarrierErrorsTotal           *prometheus.CounterVec
	EventDuplicatesTotal         *prometheus.CounterVec
}

var Global = NewMetrics()

func NewMetrics() *Metrics {
	return &Metrics{
		CameraConnected: promauto.NewGaugeVec(prometheus.GaugeOpts{
			Name: "camera_connected",
			Help: "Current connection status of camera (1 for online, 0 for offline)",
		}, []string{"camera_id"}),

		CameraReconnectTotal: promauto.NewCounterVec(prometheus.CounterOpts{
			Name: "camera_reconnect_total",
			Help: "Total reconnect attempts for camera",
		}, []string{"camera_id"}),

		FramesReceivedTotal: promauto.NewCounterVec(prometheus.CounterOpts{
			Name: "frames_received_total",
			Help: "Total raw frames received from stream",
		}, []string{"camera_id"}),

		FramesProcessedTotal: promauto.NewCounterVec(prometheus.CounterOpts{
			Name: "frames_processed_total",
			Help: "Total frames processed through pipeline",
		}, []string{"camera_id"}),

		FramesDroppedTotal: promauto.NewCounterVec(prometheus.CounterOpts{
			Name: "frames_dropped_total",
			Help: "Total frames dropped due to backpressure or rate limiting",
		}, []string{"camera_id"}),

		VehiclesDetectedTotal: promauto.NewCounterVec(prometheus.CounterOpts{
			Name: "vehicles_detected_total",
			Help: "Total vehicle movement triggers detected",
		}, []string{"camera_id"}),

		VehiclesTrackedCurrent: promauto.NewGaugeVec(prometheus.GaugeOpts{
			Name: "vehicles_tracked_current",
			Help: "Currently tracked active vehicles",
		}, []string{"camera_id"}),

		PlateDetectionsTotal: promauto.NewCounterVec(prometheus.CounterOpts{
			Name: "plate_detections_total",
			Help: "Total candidate license plate detections",
		}, []string{"camera_id"}),

		PlateRecognitionsTotal: promauto.NewCounterVec(prometheus.CounterOpts{
			Name: "plate_recognitions_total",
			Help: "Total completed OCR recognitions",
		}, []string{"camera_id"}),

		PlateRecognitionSuccessTotal: promauto.NewCounterVec(prometheus.CounterOpts{
			Name: "plate_recognition_success_total",
			Help: "Total successful high-confidence recognitions",
		}, []string{"camera_id"}),

		PlateRecognitionFailedTotal: promauto.NewCounterVec(prometheus.CounterOpts{
			Name: "plate_recognition_failed_total",
			Help: "Total failed or low-confidence plate recognitions",
		}, []string{"camera_id"}),

		RecognitionLatencyMs: promauto.NewHistogramVec(prometheus.HistogramOpts{
			Name:    "recognition_latency_ms",
			Help:    "End-to-end recognition pipeline latency in milliseconds",
			Buckets: []float64{50, 100, 200, 350, 500, 750, 1000, 1500, 2000},
		}, []string{"camera_id"}),

		BarrierOpenTotal: promauto.NewCounterVec(prometheus.CounterOpts{
			Name: "barrier_open_total",
			Help: "Total barrier open commands issued",
		}, []string{"barrier_id"}),

		BarrierCloseTotal: promauto.NewCounterVec(prometheus.CounterOpts{
			Name: "barrier_close_total",
			Help: "Total barrier close commands issued",
		}, []string{"barrier_id"}),

		BarrierErrorsTotal: promauto.NewCounterVec(prometheus.CounterOpts{
			Name: "barrier_errors_total",
			Help: "Total barrier hardware or network errors",
		}, []string{"barrier_id"}),

		EventDuplicatesTotal: promauto.NewCounterVec(prometheus.CounterOpts{
			Name: "event_duplicates_total",
			Help: "Total duplicate events suppressed",
		}, []string{"camera_id"}),
	}
}
