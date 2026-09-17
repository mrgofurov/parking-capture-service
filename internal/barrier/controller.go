package barrier

import (
	"bytes"
	"context"
	"encoding/json"
	"log"
	"net/http"
	"sync"
	"time"

	"github.com/smart-parking/parking-capture-service/internal/domain"
	"github.com/smart-parking/parking-capture-service/internal/metrics"
)

type Controller interface {
	Open(ctx context.Context, commandID, reason string) error
	Close(ctx context.Context, commandID, reason string) error
	Status(ctx context.Context) (domain.BarrierStatus, error)
	BarrierID() string
}

type HTTPBarrierController struct {
	barrierID      string
	relayURL       string
	autoCloseSec   int
	status         domain.BarrierStatus
	lastCommandID  string
	lastOpenTime   time.Time
	autoCloseTimer *time.Timer
	client         *http.Client
	mu             sync.Mutex
}

func NewHTTPBarrierController(barrierID, relayURL string, autoCloseSec int) *HTTPBarrierController {
	if autoCloseSec <= 0 {
		autoCloseSec = 10
	}
	return &HTTPBarrierController{
		barrierID:    barrierID,
		relayURL:     relayURL,
		autoCloseSec: autoCloseSec,
		status:       domain.BarrierClosed,
		client:       &http.Client{Timeout: 3 * time.Second},
	}
}

func (c *HTTPBarrierController) BarrierID() string {
	return c.barrierID
}

func (c *HTTPBarrierController) Status(ctx context.Context) (domain.BarrierStatus, error) {
	c.mu.Lock()
	defer c.mu.Unlock()
	return c.status, nil
}

func (c *HTTPBarrierController) Open(ctx context.Context, commandID, reason string) error {
	c.mu.Lock()
	defer c.mu.Unlock()

	// Idempotency: avoid executing same commandID twice
	if c.lastCommandID == commandID {
		log.Printf("[Barrier] Duplicate command %s ignored for barrier %s\n", commandID, c.barrierID)
		metrics.Global.EventDuplicatesTotal.WithLabelValues(c.barrierID).Inc()
		return nil
	}

	// Avoid spamming OPEN if already open within last 3 seconds
	if c.status == domain.BarrierOpen && time.Since(c.lastOpenTime) < 3*time.Second {
		log.Printf("[Barrier] Barrier %s is already open, ignoring spam command\n", c.barrierID)
		return nil
	}

	c.lastCommandID = commandID
	c.lastOpenTime = time.Now()

	// Trigger hardware relay if URL configured
	if c.relayURL != "" {
		go func(url string) {
			payload, _ := json.Marshal(map[string]string{
				"action":     "open",
				"command_id": commandID,
				"reason":     reason,
			})
			req, _ := http.NewRequestWithContext(context.Background(), http.MethodPost, url, bytes.NewBuffer(payload))
			req.Header.Set("Content-Type", "application/json")
			resp, err := c.client.Do(req)
			if err != nil {
				log.Printf("[Barrier] Hardware relay open error for %s: %v\n", c.barrierID, err)
				metrics.Global.BarrierErrorsTotal.WithLabelValues(c.barrierID).Inc()
				return
			}
			defer resp.Body.Close()
		}(c.relayURL)
	}

	c.status = domain.BarrierOpen
	metrics.Global.BarrierOpenTotal.WithLabelValues(c.barrierID).Inc()
	log.Printf("[Barrier] Barrier %s OPENED. Reason: %s (Command: %s)\n", c.barrierID, reason, commandID)

	// Auto-close safety timer
	if c.autoCloseTimer != nil {
		c.autoCloseTimer.Stop()
	}
	c.autoCloseTimer = time.AfterFunc(time.Duration(c.autoCloseSec)*time.Second, func() {
		_ = c.Close(context.Background(), "auto_close_"+commandID, "Auto-close safety timeout")
	})

	return nil
}

func (c *HTTPBarrierController) Close(ctx context.Context, commandID, reason string) error {
	c.mu.Lock()
	defer c.mu.Unlock()

	if c.autoCloseTimer != nil {
		c.autoCloseTimer.Stop()
		c.autoCloseTimer = nil
	}

	if c.status == domain.BarrierClosed {
		return nil
	}

	if c.relayURL != "" {
		go func(url string) {
			payload, _ := json.Marshal(map[string]string{
				"action":     "close",
				"command_id": commandID,
				"reason":     reason,
			})
			req, _ := http.NewRequestWithContext(context.Background(), http.MethodPost, url, bytes.NewBuffer(payload))
			req.Header.Set("Content-Type", "application/json")
			resp, err := c.client.Do(req)
			if err != nil {
				log.Printf("[Barrier] Hardware relay close error for %s: %v\n", c.barrierID, err)
				metrics.Global.BarrierErrorsTotal.WithLabelValues(c.barrierID).Inc()
				return
			}
			defer resp.Body.Close()
		}(c.relayURL)
	}

	c.status = domain.BarrierClosed
	metrics.Global.BarrierCloseTotal.WithLabelValues(c.barrierID).Inc()
	log.Printf("[Barrier] Barrier %s CLOSED. Reason: %s\n", c.barrierID, reason)

	return nil
}
