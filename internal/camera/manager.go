package camera

import (
	"bytes"
	"context"
	"encoding/base64"
	"fmt"
	"image"
	_ "image/jpeg"
	_ "image/png"
	"log"
	"sync"
	"time"

	"github.com/google/uuid"
	"github.com/smart-parking/parking-capture-service/internal/backend"
	"github.com/smart-parking/parking-capture-service/internal/barrier"
	"github.com/smart-parking/parking-capture-service/internal/capture"
	"github.com/smart-parking/parking-capture-service/internal/config"
	"github.com/smart-parking/parking-capture-service/internal/domain"
	"github.com/smart-parking/parking-capture-service/internal/metrics"
	"github.com/smart-parking/parking-capture-service/internal/ocr"
	"github.com/smart-parking/parking-capture-service/internal/pipeline"
	"github.com/smart-parking/parking-capture-service/internal/statemachine"
)

type CameraPipeline struct {
	Config          domain.CameraConfig
	Capturer        capture.FrameCapturer
	Tracker         *pipeline.VehicleTracker
	Consensus       *ocr.ConsensusAggregator
	StateMachine    *statemachine.RecognitionStateMachine
	Barrier         barrier.Controller
	frameChan       chan *domain.Frame
	cancelFunc      context.CancelFunc
	lastRecognition *domain.RecognitionResult
	lastFrame       *domain.Frame
	streamClients   map[chan []byte]bool
	mu              sync.RWMutex
}

type Manager struct {
	cfg        *config.Config
	backend    *backend.Client
	recognizer ocr.PlateRecognizer
	pipelines  map[string]*CameraPipeline
	mu         sync.RWMutex
}

func NewManager(cfg *config.Config, backendClient *backend.Client) *Manager {
	return &Manager{
		cfg:        cfg,
		backend:    backendClient,
		recognizer: ocr.NewNativeGoEngine(),
		pipelines:  make(map[string]*CameraPipeline),
	}
}

func (m *Manager) RegisterCamera(cam domain.CameraConfig, b barrier.Controller) error {
	m.mu.Lock()
	defer m.mu.Unlock()

	var capturer capture.FrameCapturer
	if m.cfg.SyntheticMode || cam.RTSPURL == "" {
		capturer = capture.NewSyntheticCapturer(cam.ID, cam.Direction, cam.FPS)
	} else {
		capturer = capture.NewFFmpegRTSPCapturer(cam.ID, cam.RTSPURL, cam.FPS)
	}

	tracker := pipeline.NewVehicleTracker(cam.VirtualLine.Y1)
	consensus := ocr.NewConsensusAggregator(ocr.ConsensusConfig{
		MinFrames:       m.cfg.ConsensusMinFrames,
		MaxWindowTime:   time.Duration(m.cfg.RecognitionTimeoutMs) * time.Millisecond,
		ConfidenceFloor: m.cfg.PlateConfidenceThreshold,
	})
	sm := statemachine.NewRecognitionStateMachine(cam.ID, time.Duration(m.cfg.BarrierPassageTimeoutSec)*time.Second)

	pipe := &CameraPipeline{
		Config:        cam,
		Capturer:      capturer,
		Tracker:       tracker,
		Consensus:     consensus,
		StateMachine:  sm,
		Barrier:       b,
		frameChan:     make(chan *domain.Frame, m.cfg.MaxFrameQueue),
		streamClients: make(map[chan []byte]bool),
	}

	m.pipelines[cam.ID] = pipe
	return nil
}

func (m *Manager) StartAll(ctx context.Context) {
	m.mu.Lock()
	defer m.mu.Unlock()

	for _, pipe := range m.pipelines {
		m.startPipeline(ctx, pipe)
	}
}

func (m *Manager) startPipeline(ctx context.Context, pipe *CameraPipeline) {
	pipeCtx, cancel := context.WithCancel(ctx)
	pipe.cancelFunc = cancel

	_ = pipe.Capturer.Start(pipeCtx, pipe.frameChan)

	// Worker loop for this camera pipeline
	go func(p *CameraPipeline) {
		for {
			select {
			case <-pipeCtx.Done():
				return
			case frame := <-p.frameChan:
				if frame == nil {
					continue
				}
				m.processFrame(pipeCtx, p, frame)
			}
		}
	}(pipe)

	log.Printf("[CameraManager] Camera %s (%s) pipeline started.\n", pipe.Config.Name, pipe.Config.Direction)
}

func (m *Manager) processFrame(ctx context.Context, pipe *CameraPipeline, frame *domain.Frame) {
	metrics.Global.FramesProcessedTotal.WithLabelValues(pipe.Config.ID).Inc()

	// Update last frame & broadcast to stream subscribers
	pipe.mu.Lock()
	pipe.lastFrame = frame
	for ch := range pipe.streamClients {
		select {
		case ch <- frame.JPEGBytes:
		default:
		}
	}
	pipe.mu.Unlock()

	// 1. Vehicle Tracking
	vehicle := pipe.Tracker.ProcessFrame(frame, &pipe.Config)
	if vehicle == nil {
		return
	}

	// 2. State machine on vehicle detected
	pipe.StateMachine.OnVehicleDetected(vehicle.TrackingID)

	// 3. Plate Recognition
	obsList, err := m.recognizer.Recognize(ctx, frame, vehicle)
	if err != nil || len(obsList) == 0 {
		return
	}

	pipe.StateMachine.OnPlateCandidate()
	for _, obs := range obsList {
		metrics.Global.PlateDetectionsTotal.WithLabelValues(pipe.Config.ID).Inc()
		pipe.Consensus.AddObservation(obs, frame)
	}

	// 4. Evaluate Consensus
	eval, err := pipe.Consensus.Evaluate(vehicle.TrackingID)
	if err != nil || !eval.Confirmed {
		return
	}

	// 5. Plate Confirmed!
	recResult := &domain.RecognitionResult{
		PlateNumber:       eval.PlateNumber,
		RawText:           eval.RawText,
		OverallConfidence: eval.Confidence,
		SampleCount:       eval.SampleCount,
		Timestamp:         time.Now(),
	}

	if eval.BestFrame != nil && len(eval.BestFrame.JPEGBytes) > 0 {
		recResult.SnapshotBase64 = base64.StdEncoding.EncodeToString(eval.BestFrame.JPEGBytes)
	}

	if !pipe.StateMachine.OnPlateConfirmed(eval.PlateNumber, recResult) {
		// Already handled or in cooldown
		return
	}

	pipe.mu.Lock()
	pipe.lastRecognition = recResult
	pipe.mu.Unlock()

	metrics.Global.PlateRecognitionsTotal.WithLabelValues(pipe.Config.ID).Inc()
	metrics.Global.PlateRecognitionSuccessTotal.WithLabelValues(pipe.Config.ID).Inc()

	log.Printf("[CameraManager] PLATE CONFIRMED: %s (Confidence: %.2f, Samples: %d) on Camera %s\n",
		eval.PlateNumber, eval.Confidence, eval.SampleCount, pipe.Config.ID)

	// 6. Send to Backend
	eventID := uuid.New().String()
	backendEvent := &domain.BackendEvent{
		RequestID:      uuid.New().String(),
		EventID:        eventID,
		CameraID:       pipe.Config.ID,
		ParkingID:      pipe.Config.ParkingID,
		Direction:      pipe.Config.Direction,
		PlateNumber:    eval.PlateNumber,
		Confidence:     eval.Confidence,
		Timestamp:      time.Now(),
		SnapshotBase64: recResult.SnapshotBase64,
	}

	pipe.StateMachine.OnEventSent()

	resp, err := m.backend.SendRecognitionEvent(ctx, backendEvent)
	if err != nil {
		log.Printf("[CameraManager] Warning: Backend communication failed: %v\n", err)
		return
	}

	// 7. If Backend authorizes entry/exit -> Open local Barrier!
	if resp != nil && resp.Data.BarrierOpened {
		cmdID := fmt.Sprintf("cmd_%s", eventID)
		reason := fmt.Sprintf("Backend ruxsati berildi: %s", eval.PlateNumber)
		_ = pipe.Barrier.Open(ctx, cmdID, reason)

		// Vehicle passage trigger simulation
		time.AfterFunc(3*time.Second, func() {
			pipe.StateMachine.OnVehiclePassed()
			_ = pipe.Barrier.Close(context.Background(), "pass_"+cmdID, "Avtomobil o'tdi")
		})
	}
}

func (m *Manager) InjectTestPlate(cameraID, plate string) error {
	m.mu.RLock()
	pipe, ok := m.pipelines[cameraID]
	m.mu.RUnlock()
	if !ok {
		return fmt.Errorf("camera %s not found", cameraID)
	}

	if synth, ok := pipe.Capturer.(*capture.SyntheticCapturer); ok {
		synth.InjectVehicle(plate)
		return nil
	}
	return fmt.Errorf("camera %s is not in synthetic mode", cameraID)
}

func (m *Manager) ListCameras() []map[string]interface{} {
	m.mu.RLock()
	defer m.mu.RUnlock()

	res := make([]map[string]interface{}, 0, len(m.pipelines))
	for _, p := range m.pipelines {
		p.mu.RLock()
		lastPlate := ""
		if p.lastRecognition != nil {
			lastPlate = p.lastRecognition.PlateNumber
		}
		p.mu.RUnlock()

		res = append(res, map[string]interface{}{
			"id":               p.Config.ID,
			"name":             p.Config.Name,
			"direction":        p.Config.Direction,
			"fps":              p.Config.FPS,
			"status":           p.Capturer.Status(),
			"state":            p.StateMachine.CurrentState(),
			"last_recognition": lastPlate,
		})
	}
	return res
}

func (m *Manager) ListBarriers() []map[string]interface{} {
	m.mu.RLock()
	defer m.mu.RUnlock()

	seen := make(map[string]bool)
	res := make([]map[string]interface{}, 0, len(m.pipelines))
	for _, p := range m.pipelines {
		if p.Barrier != nil {
			id := p.Barrier.BarrierID()
			if seen[id] {
				continue
			}
			seen[id] = true
			st, _ := p.Barrier.Status(context.Background())
			res = append(res, map[string]interface{}{
				"id":     id,
				"status": st,
			})
		}
	}
	return res
}

func (m *Manager) GetBarrier(id string) barrier.Controller {
	m.mu.RLock()
	defer m.mu.RUnlock()
	for _, p := range m.pipelines {
		if p.Barrier != nil && (p.Barrier.BarrierID() == id || p.Config.ID == id) {
			return p.Barrier
		}
	}
	return nil
}

func (m *Manager) GetLatestSnapshot(cameraID string) []byte {
	m.mu.RLock()
	pipe, ok := m.pipelines[cameraID]
	m.mu.RUnlock()
	if !ok {
		return nil
	}
	pipe.mu.RLock()
	defer pipe.mu.RUnlock()
	if pipe.lastFrame != nil {
		return pipe.lastFrame.JPEGBytes
	}
	return nil
}

func (m *Manager) SubscribeStream(cameraID string) (chan []byte, func()) {
	m.mu.RLock()
	pipe, ok := m.pipelines[cameraID]
	m.mu.RUnlock()
	if !ok {
		return nil, func() {}
	}

	ch := make(chan []byte, 10)
	pipe.mu.Lock()
	if pipe.streamClients == nil {
		pipe.streamClients = make(map[chan []byte]bool)
	}
	pipe.streamClients[ch] = true
	// Send latest frame immediately if available
	if pipe.lastFrame != nil {
		select {
		case ch <- pipe.lastFrame.JPEGBytes:
		default:
		}
	}
	pipe.mu.Unlock()

	unsub := func() {
		pipe.mu.Lock()
		delete(pipe.streamClients, ch)
		close(ch)
		pipe.mu.Unlock()
	}

	return ch, unsub
}

func (m *Manager) TestImageOCR(cameraID string, imgBytes []byte) (map[string]interface{}, error) {
	start := time.Now()
	img, _, err := image.Decode(bytes.NewReader(imgBytes))
	if err != nil {
		return nil, fmt.Errorf("rasmni dekodlab bo'lmadi: %w", err)
	}

	frame := &domain.Frame{
		CameraID:  cameraID,
		Timestamp: time.Now(),
		Image:     img,
		JPEGBytes: imgBytes,
		Width:     img.Bounds().Dx(),
		Height:    img.Bounds().Dy(),
	}

	dummyVeh := &domain.VehicleObservation{
		TrackingID: "test-upload-" + uuid.New().String()[:8],
		BoundingBox: domain.BoundingBox{
			X:      int(float64(img.Bounds().Dx()) * 0.2),
			Y:      int(float64(img.Bounds().Dy()) * 0.2),
			Width:  int(float64(img.Bounds().Dx()) * 0.6),
			Height: int(float64(img.Bounds().Dy()) * 0.6),
		},
	}

	obsList, err := m.recognizer.Recognize(context.Background(), frame, dummyVeh)
	if err != nil || len(obsList) == 0 {
		return nil, fmt.Errorf("davlat raqami aniqlanmadi")
	}

	obs := obsList[0]
	dur := time.Since(start).Milliseconds()

	return map[string]interface{}{
		"plate_number": obs.NormalizedText,
		"raw_text":     obs.RawText,
		"confidence":   obs.Confidence,
		"sharpness":    obs.Sharpness,
		"brightness":   obs.Brightness,
		"contrast":     obs.Contrast,
		"latency_ms":   dur,
		"timestamp":    obs.Timestamp,
	}, nil
}

func (m *Manager) StopAll() {
	m.mu.Lock()
	defer m.mu.Unlock()

	for _, pipe := range m.pipelines {
		if pipe.cancelFunc != nil {
			pipe.cancelFunc()
		}
		_ = pipe.Capturer.Stop()
	}
}
