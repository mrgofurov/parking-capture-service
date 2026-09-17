package capture

import (
	"bytes"
	"context"
	"fmt"
	"image/jpeg"
	"io"
	"log"
	"math/rand"
	"os/exec"
	"strings"
	"sync"
	"time"

	"github.com/smart-parking/parking-capture-service/internal/domain"
	"github.com/smart-parking/parking-capture-service/internal/metrics"
)

type FFmpegRTSPCapturer struct {
	cameraID   string
	rtspURL    string
	fps        int
	status     domain.CameraStatus
	cancelFunc context.CancelFunc
	mu         sync.RWMutex
}

func NewFFmpegRTSPCapturer(cameraID, rtspURL string, fps int) *FFmpegRTSPCapturer {
	if fps <= 0 {
		fps = 8
	}
	return &FFmpegRTSPCapturer{
		cameraID: cameraID,
		rtspURL:  rtspURL,
		fps:      fps,
		status:   domain.CameraOffline,
	}
}

func (c *FFmpegRTSPCapturer) CameraID() string {
	return c.cameraID
}

func (c *FFmpegRTSPCapturer) Status() domain.CameraStatus {
	c.mu.RLock()
	defer c.mu.RUnlock()
	return c.status
}

func (c *FFmpegRTSPCapturer) Start(ctx context.Context, out chan<- *domain.Frame) error {
	ctx, cancel := context.WithCancel(ctx)
	c.mu.Lock()
	c.cancelFunc = cancel
	c.status = domain.CameraReconnecting
	c.mu.Unlock()

	go c.runSupervisor(ctx, out)
	return nil
}

func (c *FFmpegRTSPCapturer) Stop() error {
	c.mu.Lock()
	defer c.mu.Unlock()
	if c.cancelFunc != nil {
		c.cancelFunc()
	}
	c.status = domain.CameraOffline
	return nil
}

func (c *FFmpegRTSPCapturer) runSupervisor(ctx context.Context, out chan<- *domain.Frame) {
	backoff := 1 * time.Second
	maxBackoff := 30 * time.Second

	for {
		select {
		case <-ctx.Done():
			return
		default:
		}

		c.mu.Lock()
		c.status = domain.CameraReconnecting
		c.mu.Unlock()
		metrics.Global.CameraReconnectTotal.WithLabelValues(c.cameraID).Inc()

		err := c.streamOnce(ctx, out)
		if err != nil {
			log.Printf("[FFmpeg] Camera %s disconnected: %v. Reconnecting in %v...\n", c.cameraID, err, backoff)
		}

		c.mu.Lock()
		c.status = domain.CameraOffline
		c.mu.Unlock()
		metrics.Global.CameraConnected.WithLabelValues(c.cameraID).Set(0)

		// Jitter
		jitter := time.Duration(rand.Intn(500)) * time.Millisecond
		select {
		case <-ctx.Done():
			return
		case <-time.After(backoff + jitter):
			backoff *= 2
			if backoff > maxBackoff {
				backoff = maxBackoff
			}
		}
	}
}

func (c *FFmpegRTSPCapturer) streamOnce(ctx context.Context, out chan<- *domain.Frame) error {
	var args []string
	if strings.HasPrefix(c.rtspURL, "rtsp://") || strings.HasPrefix(c.rtspURL, "rtsps://") {
		args = []string{
			"-hide_banner",
			"-loglevel", "error",
			"-rtsp_transport", "tcp",
			"-i", c.rtspURL,
			"-vf", fmt.Sprintf("fps=%d", c.fps),
			"-f", "image2pipe",
			"-vcodec", "mjpeg",
			"-",
		}
	} else {
		// Video file loop mode for real video testing
		args = []string{
			"-hide_banner",
			"-loglevel", "error",
			"-stream_loop", "-1",
			"-re",
			"-i", c.rtspURL,
			"-vf", fmt.Sprintf("fps=%d", c.fps),
			"-f", "image2pipe",
			"-vcodec", "mjpeg",
			"-",
		}
	}

	cmd := exec.CommandContext(ctx, "ffmpeg", args...)
	stdout, err := cmd.StdoutPipe()
	if err != nil {
		return err
	}

	if err := cmd.Start(); err != nil {
		return err
	}

	c.mu.Lock()
	c.status = domain.CameraOnline
	c.mu.Unlock()
	metrics.Global.CameraConnected.WithLabelValues(c.cameraID).Set(1)

	defer func() {
		_ = cmd.Process.Kill()
		_ = cmd.Wait()
	}()

	var seq uint64 = 0
	buf := make([]byte, 64*1024)
	var frameBuffer bytes.Buffer

	for {
		n, err := stdout.Read(buf)
		if n > 0 {
			frameBuffer.Write(buf[:n])
			for {
				data := frameBuffer.Bytes()
				// Find JPEG SOI (0xFF, 0xD8) and EOI (0xFF, 0xD9)
				soi := bytes.Index(data, []byte{0xFF, 0xD8})
				if soi == -1 {
					break
				}
				eoi := bytes.Index(data[soi:], []byte{0xFF, 0xD9})
				if eoi == -1 {
					break
				}
				fullEOI := soi + eoi + 2
				jpegData := data[soi:fullEOI]

				// Parse image
				img, err := jpeg.Decode(bytes.NewReader(jpegData))
				if err == nil {
					seq++
					metrics.Global.FramesReceivedTotal.WithLabelValues(c.cameraID).Inc()

					f := &domain.Frame{
						CameraID:   c.cameraID,
						SequenceID: seq,
						Timestamp:  time.Now(),
						Image:      img,
						JPEGBytes:  jpegData,
						Width:      img.Bounds().Dx(),
						Height:     img.Bounds().Dy(),
					}

					select {
					case out <- f:
					default:
						metrics.Global.FramesDroppedTotal.WithLabelValues(c.cameraID).Inc()
					}
				}

				// Trim frameBuffer
				frameBuffer.Next(fullEOI)
			}
		}

		if err != nil {
			if err == io.EOF {
				return fmt.Errorf("RTSP stream EOF")
			}
			return err
		}
	}
}
