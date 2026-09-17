package capture

import (
	"bytes"
	"context"
	"image"
	"image/color"
	"image/draw"
	"image/jpeg"
	"sync"
	"time"

	"github.com/smart-parking/parking-capture-service/internal/domain"
	"github.com/smart-parking/parking-capture-service/internal/metrics"
)

type SyntheticCapturer struct {
	cameraID    string
	direction   domain.Direction
	fps         int
	status      domain.CameraStatus
	injections  chan string
	cancelFunc  context.CancelFunc
	mu          sync.RWMutex
}

func NewSyntheticCapturer(cameraID string, direction domain.Direction, fps int) *SyntheticCapturer {
	if fps <= 0 {
		fps = 8
	}
	return &SyntheticCapturer{
		cameraID:   cameraID,
		direction:  direction,
		fps:        fps,
		status:     domain.CameraOnline,
		injections: make(chan string, 10),
	}
}

func (c *SyntheticCapturer) CameraID() string {
	return c.cameraID
}

func (c *SyntheticCapturer) Status() domain.CameraStatus {
	c.mu.RLock()
	defer c.mu.RUnlock()
	return c.status
}

func (c *SyntheticCapturer) InjectVehicle(plate string) {
	select {
	case c.injections <- plate:
	default:
	}
}

func (c *SyntheticCapturer) Start(ctx context.Context, out chan<- *domain.Frame) error {
	ctx, cancel := context.WithCancel(ctx)
	c.mu.Lock()
	c.cancelFunc = cancel
	c.status = domain.CameraOnline
	c.mu.Unlock()

	metrics.Global.CameraConnected.WithLabelValues(c.cameraID).Set(1)

	ticker := time.NewTicker(time.Duration(1000/c.fps) * time.Millisecond)
	go func() {
		defer ticker.Stop()
		defer func() {
			c.mu.Lock()
			c.status = domain.CameraOffline
			c.mu.Unlock()
			metrics.Global.CameraConnected.WithLabelValues(c.cameraID).Set(0)
		}()

		var seq uint64 = 0
		var currentPlate string
		var vehicleFramesRemaining = 0

		for {
			select {
			case <-ctx.Done():
				return

			case plate := <-c.injections:
				currentPlate = plate
				vehicleFramesRemaining = 12 // 12 frames of car in view

			case t := <-ticker.C:
				seq++
				metrics.Global.FramesReceivedTotal.WithLabelValues(c.cameraID).Inc()

				hasVehicle := false
				activePlate := ""
				if vehicleFramesRemaining > 0 {
					hasVehicle = true
					activePlate = currentPlate
					vehicleFramesRemaining--
				}

				img, jpegBytes := generateSyntheticFrame(c.cameraID, c.direction, seq, hasVehicle, activePlate)

				frame := &domain.Frame{
					CameraID:   c.cameraID,
					SequenceID: seq,
					Timestamp:  t,
					Image:      img,
					JPEGBytes:  jpegBytes,
					Width:      640,
					Height:     480,
				}

				select {
				case out <- frame:
				default:
					metrics.Global.FramesDroppedTotal.WithLabelValues(c.cameraID).Inc()
				}
			}
		}
	}()

	return nil
}

func (c *SyntheticCapturer) Stop() error {
	c.mu.Lock()
	defer c.mu.Unlock()
	if c.cancelFunc != nil {
		c.cancelFunc()
	}
	return nil
}

func generateSyntheticFrame(cameraID string, dir domain.Direction, seq uint64, hasVehicle bool, plate string) (image.Image, []byte) {
	img := image.NewRGBA(image.Rect(0, 0, 640, 480))

	// Background lane (asphalt dark gray)
	bgColor := color.RGBA{R: 35, G: 38, B: 45, A: 255}
	draw.Draw(img, img.Bounds(), &image.Uniform{bgColor}, image.Point{}, draw.Src)

	// Virtual line in middle (yellow lane line)
	lineColor := color.RGBA{R: 245, G: 195, B: 35, A: 255}
	for x := 0; x < 640; x++ {
		for y := 240; y < 244; y++ {
			img.Set(x, y, lineColor)
		}
	}

	if hasVehicle {
		// Vehicle body (Blue/metallic rectangle moving)
		carBox := image.Rect(180, 160, 460, 360)
		carColor := color.RGBA{R: 40, G: 80, B: 180, A: 255}
		draw.Draw(img, carBox, &image.Uniform{carColor}, image.Point{}, draw.Src)

		// License plate area (white rectangular plate)
		plateBox := image.Rect(250, 290, 390, 330)
		plateBg := color.RGBA{R: 250, G: 250, B: 250, A: 255}
		draw.Draw(img, plateBox, &image.Uniform{plateBg}, image.Point{}, draw.Src)

		// Plate border
		borderCol := color.RGBA{R: 15, G: 15, B: 15, A: 255}
		for x := 250; x <= 390; x++ {
			img.Set(x, 290, borderCol)
			img.Set(x, 330, borderCol)
		}
		for y := 290; y <= 330; y++ {
			img.Set(250, y, borderCol)
			img.Set(390, y, borderCol)
		}

		// Blue UZ strip on left
		uzStrip := image.Rect(250, 290, 270, 330)
		uzColor := color.RGBA{R: 20, G: 80, B: 220, A: 255}
		draw.Draw(img, uzStrip, &image.Uniform{uzColor}, image.Point{}, draw.Src)
	}

	var buf bytes.Buffer
	_ = jpeg.Encode(&buf, img, &jpeg.Options{Quality: 80})

	return img, buf.Bytes()
}
