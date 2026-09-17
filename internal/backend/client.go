package backend

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"log"
	"net/http"
	"sync"
	"time"

	"github.com/smart-parking/parking-capture-service/internal/domain"
	"github.com/smart-parking/parking-capture-service/internal/metrics"
)

type Client struct {
	baseURL      string
	token        string
	httpClient   *http.Client
	pendingQueue []*domain.BackendEvent
	queueMu      sync.Mutex
	maxQueueSize int
}

func NewClient(baseURL, token string, maxQueueSize int) *Client {
	if maxQueueSize <= 0 {
		maxQueueSize = 200
	}
	c := &Client{
		baseURL:      baseURL,
		token:        token,
		httpClient:   &http.Client{Timeout: 4 * time.Second},
		pendingQueue: make([]*domain.BackendEvent, 0, maxQueueSize),
		maxQueueSize: maxQueueSize,
	}
	go c.startQueueFlusher()
	return c
}

func (c *Client) SendRecognitionEvent(ctx context.Context, event *domain.BackendEvent) (*domain.BackendResponse, error) {
	url := fmt.Sprintf("%s/api/v1/anpr/event", c.baseURL)

	payload, err := json.Marshal(map[string]interface{}{
		"plate_number":    event.PlateNumber,
		"direction":       event.Direction,
		"confidence":      event.Confidence * 100.0,
		"timestamp":       event.Timestamp,
		"snapshot_base64": event.SnapshotBase64,
	})
	if err != nil {
		return nil, err
	}

	start := time.Now()
	req, err := http.NewRequestWithContext(ctx, http.MethodPost, url, bytes.NewBuffer(payload))
	if err != nil {
		return nil, err
	}
	req.Header.Set("Content-Type", "application/json")
	if c.token != "" {
		req.Header.Set("Authorization", "Bearer "+c.token)
	}

	resp, err := c.httpClient.Do(req)
	latency := time.Since(start).Milliseconds()
	metrics.Global.RecognitionLatencyMs.WithLabelValues(event.CameraID).Observe(float64(latency))

	if err != nil {
		log.Printf("[BackendClient] Request error: %v. Queuing event for retry.\n", err)
		c.enqueue(event)
		return nil, err
	}
	defer resp.Body.Close()

	if resp.StatusCode >= 500 {
		c.enqueue(event)
		return nil, fmt.Errorf("backend error: status %d", resp.StatusCode)
	}

	var backendResp domain.BackendResponse
	if err := json.NewDecoder(resp.Body).Decode(&backendResp); err != nil {
		return nil, err
	}

	return &backendResp, nil
}

func (c *Client) enqueue(event *domain.BackendEvent) {
	c.queueMu.Lock()
	defer c.queueMu.Unlock()

	if len(c.pendingQueue) >= c.maxQueueSize {
		// Drop oldest to maintain bound
		c.pendingQueue = c.pendingQueue[1:]
	}
	c.pendingQueue = append(c.pendingQueue, event)
}

func (c *Client) startQueueFlusher() {
	ticker := time.NewTicker(5 * time.Second)
	for range ticker.C {
		c.flushQueue()
	}
}

func (c *Client) flushQueue() {
	c.queueMu.Lock()
	if len(c.pendingQueue) == 0 {
		c.queueMu.Unlock()
		return
	}
	events := make([]*domain.BackendEvent, len(c.pendingQueue))
	copy(events, c.pendingQueue)
	c.pendingQueue = c.pendingQueue[:0]
	c.queueMu.Unlock()

	for _, ev := range events {
		ctx, cancel := context.WithTimeout(context.Background(), 3*time.Second)
		_, err := c.SendRecognitionEvent(ctx, ev)
		cancel()
		if err != nil {
			// Re-enqueue if still failing
			c.enqueue(ev)
			break
		}
	}
}
