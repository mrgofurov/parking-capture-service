package ocr

import (
	"errors"
	"math"
	"sync"
	"time"

	"github.com/smart-parking/parking-capture-service/internal/domain"
)

type ConsensusConfig struct {
	MinFrames       int
	MaxWindowTime   time.Duration
	ConfidenceFloor float64
}

type TrackedPlateSession struct {
	TrackingID   string
	Observations []domain.PlateObservation
	FirstSeen    time.Time
	LastSeen     time.Time
	BestFrame    *domain.Frame
	BestQuality  float64
}

type ConsensusAggregator struct {
	cfg      ConsensusConfig
	sessions map[string]*TrackedPlateSession
	mu       sync.Mutex
}

func NewConsensusAggregator(cfg ConsensusConfig) *ConsensusAggregator {
	if cfg.MinFrames <= 0 {
		cfg.MinFrames = 3
	}
	if cfg.MaxWindowTime <= 0 {
		cfg.MaxWindowTime = 1500 * time.Millisecond
	}
	if cfg.ConfidenceFloor <= 0 {
		cfg.ConfidenceFloor = 0.70
	}
	return &ConsensusAggregator{
		cfg:      cfg,
		sessions: make(map[string]*TrackedPlateSession),
	}
}

func (a *ConsensusAggregator) AddObservation(obs domain.PlateObservation, frame *domain.Frame) {
	a.mu.Lock()
	defer a.mu.Unlock()

	now := time.Now()
	sess, exists := a.sessions[obs.TrackingID]
	if !exists {
		sess = &TrackedPlateSession{
			TrackingID:   obs.TrackingID,
			Observations: make([]domain.PlateObservation, 0, 10),
			FirstSeen:    now,
			LastSeen:     now,
			BestFrame:    frame,
			BestQuality:  obs.Sharpness,
		}
		a.sessions[obs.TrackingID] = sess
	}

	sess.Observations = append(sess.Observations, obs)
	sess.LastSeen = now

	// Update best frame based on sharpness
	if frame != nil && obs.Sharpness > sess.BestQuality {
		sess.BestFrame = frame
		sess.BestQuality = obs.Sharpness
	}
}

type ConsensusResult struct {
	Confirmed     bool
	PlateNumber   string
	RawText       string
	Confidence    float64
	SampleCount   int
	BestFrame     *domain.Frame
	FormatValid   bool
	Reason        string
}

func (a *ConsensusAggregator) Evaluate(trackingID string) (*ConsensusResult, error) {
	a.mu.Lock()
	defer a.mu.Unlock()

	sess, exists := a.sessions[trackingID]
	if !exists || len(sess.Observations) == 0 {
		return nil, errors.New("no observations found for tracking ID")
	}

	// 1. Group observations by normalized plate
	type groupStats struct {
		rawTexts      []string
		count         int
		confSum       float64
		sharpnessSum  float64
		maxConfidence float64
		plateType     string
	}
	groups := make(map[string]*groupStats)

	for _, obs := range sess.Observations {
		norm := NormalizePlate(obs.NormalizedText)
		key := norm.Normalized
		if key == "" {
			continue
		}

		g, ok := groups[key]
		if !ok {
			g = &groupStats{
				rawTexts:      make([]string, 0, 5),
				plateType:     norm.PlateType,
				maxConfidence: obs.Confidence,
			}
			groups[key] = g
		}
		g.rawTexts = append(g.rawTexts, obs.RawText)
		g.count++
		g.confSum += obs.Confidence + norm.ScoreBonus
		g.sharpnessSum += obs.Sharpness
		if obs.Confidence > g.maxConfidence {
			g.maxConfidence = obs.Confidence
		}
	}

	if len(groups) == 0 {
		return &ConsensusResult{
			Confirmed:   false,
			SampleCount: len(sess.Observations),
			Reason:      "Barcha o'qilgan freymlar formati yaroqsiz",
		}, nil
	}

	// 2. Select winning candidate with highest weighted consensus score
	var bestPlate string
	var bestScore float64
	var bestGroup *groupStats

	totalObs := float64(len(sess.Observations))
	for plate, g := range groups {
		avgConf := g.confSum / float64(g.count)
		freqRatio := float64(g.count) / totalObs

		// Consensus Score Formula:
		// 60% average confidence + 30% frequency ratio + 10% sample volume factor
		sampleBonus := math.Min(0.10, float64(g.count)/10.0)
		score := (avgConf * 0.60) + (freqRatio * 0.30) + sampleBonus

		if score > bestScore {
			bestScore = score
			bestPlate = plate
			bestGroup = g
		}
	}

	// Normalize score into 0.0 - 1.0
	finalConfidence := math.Min(0.99, bestScore)

	// Check if meets threshold
	hasEnoughFrames := bestGroup.count >= a.cfg.MinFrames
	meetsConfidence := finalConfidence >= a.cfg.ConfidenceFloor

	raw := ""
	if len(bestGroup.rawTexts) > 0 {
		raw = bestGroup.rawTexts[0]
	}

	if hasEnoughFrames && meetsConfidence {
		return &ConsensusResult{
			Confirmed:   true,
			PlateNumber: bestPlate,
			RawText:     raw,
			Confidence:  finalConfidence,
			SampleCount: bestGroup.count,
			BestFrame:   sess.BestFrame,
			FormatValid: true,
			Reason:      "Ko'p freymli konsensus muvaffaqiyatli tasdiqlandi",
		}, nil
	}

	return &ConsensusResult{
		Confirmed:   false,
		PlateNumber: bestPlate,
		RawText:     raw,
		Confidence:  finalConfidence,
		SampleCount: bestGroup.count,
		BestFrame:   sess.BestFrame,
		FormatValid: false,
		Reason:      "Yetarli freymlar soni yoki ishonchlilik darajasi yetarli emas",
	}, nil
}

func (a *ConsensusAggregator) Cleanup(trackingID string) {
	a.mu.Lock()
	defer a.mu.Unlock()
	delete(a.sessions, trackingID)
}

func (a *ConsensusAggregator) CleanupStale(maxAge time.Duration) {
	a.mu.Lock()
	defer a.mu.Unlock()
	cutoff := time.Now().Add(-maxAge)
	for id, sess := range a.sessions {
		if sess.LastSeen.Before(cutoff) {
			delete(a.sessions, id)
		}
	}
}
