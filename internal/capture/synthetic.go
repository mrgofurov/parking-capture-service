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
				var progress float64 = 0.0
				if vehicleFramesRemaining > 0 {
					hasVehicle = true
					activePlate = currentPlate
					if activePlate == "" {
						activePlate = "01A777AA"
					}
					progress = float64(14-vehicleFramesRemaining) / 14.0
					vehicleFramesRemaining--
				}

				img, jpegBytes := generateSyntheticFrame(c.cameraID, c.direction, seq, hasVehicle, activePlate, progress)

				frame := &domain.Frame{
					CameraID:   c.cameraID,
					SequenceID: seq,
					Timestamp:  t,
					Image:      img,
					JPEGBytes:  jpegBytes,
					Width:      640,
					Height:     480,
					PlateHint:  activePlate,
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

func generateSyntheticFrame(cameraID string, dir domain.Direction, seq uint64, hasVehicle bool, plate string, progress float64) (image.Image, []byte) {
	img := image.NewRGBA(image.Rect(0, 0, 640, 480))

	// 1. Asphalt roadway (textured dark gray)
	asphaltColor := color.RGBA{R: 36, G: 40, B: 48, A: 255}
	draw.Draw(img, img.Bounds(), &image.Uniform{asphaltColor}, image.Point{}, draw.Src)

	// Road curbs (left & right gray curb stripes)
	curbColor := color.RGBA{R: 70, G: 74, B: 82, A: 255}
	draw.Draw(img, image.Rect(0, 0, 40, 480), &image.Uniform{curbColor}, image.Point{}, draw.Src)
	draw.Draw(img, image.Rect(600, 0, 640, 480), &image.Uniform{curbColor}, image.Point{}, draw.Src)

	// White road lane boundary lines
	whiteLine := color.RGBA{R: 210, G: 215, B: 220, A: 255}
	for y := 0; y < 480; y++ {
		for x := 60; x <= 64; x++ {
			img.Set(x, y, whiteLine)
		}
		for x := 576; x <= 580; x++ {
			img.Set(x, y, whiteLine)
		}
	}

	// 2. Yellow Virtual Detection Line (Dashed ANPR trigger zone)
	virtualLineY := 260
	triggerColor := color.RGBA{R: 245, G: 190, B: 35, A: 255}
	for x := 64; x < 576; x++ {
		// Dashed pattern
		if (x/16)%2 == 0 {
			for dy := -2; dy <= 2; dy++ {
				img.Set(x, virtualLineY+dy, triggerColor)
			}
		}
	}

	// Draw Virtual Line Label
	drawText(img, 70, virtualLineY-12, "VIRTUAL DETECTION TRIGGER LINE", triggerColor, 1)

	// 3. Vehicle Rendering (if present)
	if hasVehicle {
		if plate == "" {
			plate = "01A777AA"
		}

		// Calculate car Y position based on progress (driving forward)
		// Moves from Y=80 to Y=280
		carCenterY := 90 + int(progress*180.0)
		carW := 280
		carH := 180
		carLeft := 320 - (carW / 2)
		carTop := carCenterY - (carH / 2)

		// Car Body (Deep Metallic Blue)
		carBodyColor := color.RGBA{R: 28, G: 64, B: 148, A: 255}
		carHoodColor := color.RGBA{R: 38, G: 82, B: 180, A: 255}
		carRoofColor := color.RGBA{R: 18, G: 48, B: 110, A: 255}

		// Main body
		draw.Draw(img, image.Rect(carLeft, carTop+40, carLeft+carW, carTop+carH), &image.Uniform{carBodyColor}, image.Point{}, draw.Src)
		// Hood
		draw.Draw(img, image.Rect(carLeft+20, carTop+70, carLeft+carW-20, carTop+carH), &image.Uniform{carHoodColor}, image.Point{}, draw.Src)
		// Cabin / Windshield
		windshieldCol := color.RGBA{R: 15, G: 25, B: 40, A: 255}
		draw.Draw(img, image.Rect(carLeft+40, carTop, carLeft+carW-40, carTop+65), &image.Uniform{carRoofColor}, image.Point{}, draw.Src)
		draw.Draw(img, image.Rect(carLeft+46, carTop+15, carLeft+carW-46, carTop+60), &image.Uniform{windshieldCol}, image.Point{}, draw.Src)

		// Headlights (Xenon White)
		headlightCol := color.RGBA{R: 240, G: 245, B: 255, A: 255}
		draw.Draw(img, image.Rect(carLeft+15, carTop+135, carLeft+50, carTop+155), &image.Uniform{headlightCol}, image.Point{}, draw.Src)
		draw.Draw(img, image.Rect(carLeft+carW-50, carTop+135, carLeft+carW-15, carTop+155), &image.Uniform{headlightCol}, image.Point{}, draw.Src)

		// Front Bumper / Grille
		grilleCol := color.RGBA{R: 18, G: 20, B: 24, A: 255}
		draw.Draw(img, image.Rect(carLeft+60, carTop+130, carLeft+carW-60, carTop+165), &image.Uniform{grilleCol}, image.Point{}, draw.Src)

		// License plate mounting
		plateW := 150
		plateH := 36
		plateX := 320 - (plateW / 2)
		plateY := carTop + 145

		// White Plate Background
		plateBg := color.RGBA{R: 252, G: 252, B: 254, A: 255}
		draw.Draw(img, image.Rect(plateX, plateY, plateX+plateW, plateY+plateH), &image.Uniform{plateBg}, image.Point{}, draw.Src)

		// Black border
		plateBorder := color.RGBA{R: 15, G: 15, B: 15, A: 255}
		for bx := plateX; bx < plateX+plateW; bx++ {
			img.Set(bx, plateY, plateBorder)
			img.Set(bx, plateY+plateH-1, plateBorder)
		}
		for by := plateY; by < plateY+plateH; by++ {
			img.Set(plateX, by, plateBorder)
			img.Set(plateX+plateW-1, by, plateBorder)
		}

		// Blue UZ flag banner on left
		uzBlue := color.RGBA{R: 22, G: 98, B: 220, A: 255}
		draw.Draw(img, image.Rect(plateX+1, plateY+1, plateX+22, plateY+plateH-1), &image.Uniform{uzBlue}, image.Point{}, draw.Src)
		drawText(img, plateX+3, plateY+12, "UZ", color.RGBA{R: 255, G: 255, B: 255, A: 255}, 1)

		// Render Plate Characters in crisp black text
		plateTextCol := color.RGBA{R: 10, G: 10, B: 10, A: 255}
		drawText(img, plateX+28, plateY+10, plate, plateTextCol, 2)

		// Green ANPR Tracking Bounding Box
		trackGreen := color.RGBA{R: 34, G: 197, B: 94, A: 255}
		drawBoundingBox(img, carLeft-6, carTop-6, carW+12, carH+20, trackGreen)
		drawText(img, carLeft-4, carTop-18, "VEHICLE TRACKED: "+plate+" (95.4%)", trackGreen, 1)
	}

	// 4. ANPR Camera HUD Overlay (Top & Bottom bars)
	hudBg := color.RGBA{R: 10, G: 14, B: 20, A: 210}
	draw.Draw(img, image.Rect(0, 0, 640, 32), &image.Uniform{hudBg}, image.Point{}, draw.Src)
	draw.Draw(img, image.Rect(0, 452, 640, 480), &image.Uniform{hudBg}, image.Point{}, draw.Src)

	// HUD Text
	hudGreen := color.RGBA{R: 52, G: 211, B: 153, A: 255}
	hudGray := color.RGBA{R: 200, G: 210, B: 225, A: 255}
	title := cameraID + " • " + string(dir) + " • 8 FPS"
	drawText(img, 12, 10, "[LIVE ANPR] "+title, hudGreen, 1)

	nowStr := time.Now().Format("2006-01-02 15:04:05")
	drawText(img, 460, 10, nowStr, hudGray, 1)

	drawText(img, 12, 460, "SMART PARKING EDGE CAPTURE SERVICE • MULTI-FRAME CONSENSUS OCR", hudGray, 1)

	var buf bytes.Buffer
	_ = jpeg.Encode(&buf, img, &jpeg.Options{Quality: 82})

	return img, buf.Bytes()
}

func drawBoundingBox(img *image.RGBA, x, y, w, h int, col color.RGBA) {
	bounds := img.Bounds()
	for dx := 0; dx < w; dx++ {
		px := x + dx
		if px >= bounds.Min.X && px < bounds.Max.X {
			if y >= bounds.Min.Y && y < bounds.Max.Y {
				img.Set(px, y, col)
				img.Set(px, y+1, col)
			}
			if y+h-1 >= bounds.Min.Y && y+h-1 < bounds.Max.Y {
				img.Set(px, y+h-1, col)
				img.Set(px, y+h-2, col)
			}
		}
	}
	for dy := 0; dy < h; dy++ {
		py := y + dy
		if py >= bounds.Min.Y && py < bounds.Max.Y {
			if x >= bounds.Min.X && x < bounds.Max.X {
				img.Set(x, py, col)
				img.Set(x+1, py, col)
			}
			if x+w-1 >= bounds.Min.X && x+w-1 < bounds.Max.X {
				img.Set(x+w-1, py, col)
				img.Set(x+w-2, py, col)
			}
		}
	}
}

// 5x7 Minimal Bitmap Font for clean rendering
var font5x7 = map[rune][7]uint8{
	'0': {0x0E, 0x11, 0x13, 0x15, 0x19, 0x11, 0x0E},
	'1': {0x04, 0x0C, 0x04, 0x04, 0x04, 0x04, 0x0E},
	'2': {0x0E, 0x11, 0x01, 0x02, 0x04, 0x08, 0x1F},
	'3': {0x1E, 0x01, 0x01, 0x0E, 0x01, 0x01, 0x1E},
	'4': {0x02, 0x06, 0x0A, 0x12, 0x1F, 0x02, 0x02},
	'5': {0x1F, 0x10, 0x1E, 0x01, 0x01, 0x11, 0x0E},
	'6': {0x06, 0x08, 0x10, 0x1E, 0x11, 0x11, 0x0E},
	'7': {0x1F, 0x01, 0x02, 0x04, 0x08, 0x08, 0x08},
	'8': {0x0E, 0x11, 0x11, 0x0E, 0x11, 0x11, 0x0E},
	'9': {0x0E, 0x11, 0x11, 0x0F, 0x01, 0x02, 0x0C},
	'A': {0x0E, 0x11, 0x11, 0x1F, 0x11, 0x11, 0x11},
	'B': {0x1E, 0x11, 0x11, 0x1E, 0x11, 0x11, 0x1E},
	'C': {0x0E, 0x11, 0x10, 0x10, 0x10, 0x11, 0x0E},
	'D': {0x1C, 0x12, 0x11, 0x11, 0x11, 0x12, 0x1C},
	'E': {0x1F, 0x10, 0x10, 0x1E, 0x10, 0x10, 0x1F},
	'F': {0x1F, 0x10, 0x10, 0x1E, 0x10, 0x10, 0x10},
	'G': {0x0E, 0x11, 0x10, 0x17, 0x11, 0x11, 0x0F},
	'H': {0x11, 0x11, 0x11, 0x1F, 0x11, 0x11, 0x11},
	'I': {0x0E, 0x04, 0x04, 0x04, 0x04, 0x04, 0x0E},
	'J': {0x07, 0x02, 0x02, 0x02, 0x02, 0x12, 0x0C},
	'K': {0x11, 0x12, 0x14, 0x18, 0x14, 0x12, 0x11},
	'L': {0x10, 0x10, 0x10, 0x10, 0x10, 0x10, 0x1F},
	'M': {0x11, 0x1B, 0x15, 0x15, 0x11, 0x11, 0x11},
	'N': {0x11, 0x11, 0x19, 0x15, 0x13, 0x11, 0x11},
	'O': {0x0E, 0x11, 0x11, 0x11, 0x11, 0x11, 0x0E},
	'P': {0x1E, 0x11, 0x11, 0x1E, 0x10, 0x10, 0x10},
	'Q': {0x0E, 0x11, 0x11, 0x11, 0x15, 0x12, 0x0D},
	'R': {0x1E, 0x11, 0x11, 0x1E, 0x14, 0x12, 0x11},
	'S': {0x0E, 0x11, 0x10, 0x0E, 0x01, 0x11, 0x0E},
	'T': {0x1F, 0x04, 0x04, 0x04, 0x04, 0x04, 0x04},
	'U': {0x11, 0x11, 0x11, 0x11, 0x11, 0x11, 0x0E},
	'V': {0x11, 0x11, 0x11, 0x11, 0x11, 0x0A, 0x04},
	'W': {0x11, 0x11, 0x11, 0x15, 0x15, 0x15, 0x0A},
	'X': {0x11, 0x11, 0x0A, 0x04, 0x0A, 0x11, 0x11},
	'Y': {0x11, 0x11, 0x0A, 0x04, 0x04, 0x04, 0x04},
	'Z': {0x1F, 0x01, 0x02, 0x04, 0x08, 0x10, 0x1F},
	' ': {0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00},
	':': {0x00, 0x0C, 0x0C, 0x00, 0x0C, 0x0C, 0x00},
	'-': {0x00, 0x00, 0x00, 0x1F, 0x00, 0x00, 0x00},
	'.': {0x00, 0x00, 0x00, 0x00, 0x00, 0x0C, 0x0C},
	'•': {0x00, 0x00, 0x0E, 0x0E, 0x0E, 0x00, 0x00},
	'[': {0x0E, 0x08, 0x08, 0x08, 0x08, 0x08, 0x0E},
	']': {0x0E, 0x02, 0x02, 0x02, 0x02, 0x02, 0x0E},
	'|': {0x04, 0x04, 0x04, 0x04, 0x04, 0x04, 0x04},
	'%': {0x19, 0x19, 0x02, 0x04, 0x08, 0x13, 0x13},
	'#': {0x0A, 0x0A, 0x1F, 0x0A, 0x1F, 0x0A, 0x0A},
}

func drawText(img *image.RGBA, startX, startY int, text string, col color.RGBA, scale int) {
	if scale < 1 {
		scale = 1
	}
	curX := startX
	for _, r := range text {
		glyph, ok := font5x7[r]
		if !ok {
			// Check lowercase
			if r >= 'a' && r <= 'z' {
				glyph, ok = font5x7[r-32]
			}
		}
		if !ok {
			glyph = font5x7[' ']
		}

		for row := 0; row < 7; row++ {
			bits := glyph[row]
			for colIdx := 0; colIdx < 5; colIdx++ {
				// Bit 4 is leftmost
				if (bits & (1 << (4 - colIdx))) != 0 {
					for sx := 0; sx < scale; sx++ {
						for sy := 0; sy < scale; sy++ {
							px := curX + (colIdx * scale) + sx
							py := startY + (row * scale) + sy
							if px >= 0 && px < 640 && py >= 0 && py < 480 {
								img.Set(px, py, col)
							}
						}
					}
				}
			}
		}
		curX += (5 * scale) + scale
	}
}
