package ocr

import (
	"testing"
	"time"

	"github.com/smart-parking/parking-capture-service/internal/domain"
)

func TestNormalizePlate(t *testing.T) {
	tests := []struct {
		input       string
		expected    string
		valid       bool
		expectedTyp string
	}{
		{"01A777AA", "01A777AA", true, "INDIVIDUAL"},
		{"01 A 777 AA", "01A777AA", true, "INDIVIDUAL"},
		{"01-A-777-AA", "01A777AA", true, "INDIVIDUAL"},
		{"O1A777AA", "01A777AA", true, "INDIVIDUAL"}, // Ambiguity correction: O -> 0 at index 0
		{"01123ABC", "01123ABC", true, "LEGAL"},
		{"INVALID", "INVALID", false, "INVALID"},
		{"123", "123", false, "INVALID"},
	}

	for _, tt := range tests {
		res := NormalizePlate(tt.input)
		if res.IsValid != tt.valid {
			t.Errorf("NormalizePlate(%s): expected valid=%v, got=%v", tt.input, tt.valid, res.IsValid)
		}
		if res.Normalized != tt.expected {
			t.Errorf("NormalizePlate(%s): expected normalized=%s, got=%s", tt.input, tt.expected, res.Normalized)
		}
		if tt.valid && res.PlateType != tt.expectedTyp {
			t.Errorf("NormalizePlate(%s): expected type=%s, got=%s", tt.input, tt.expectedTyp, res.PlateType)
		}
	}
}

func TestConsensusAggregator_Evaluate(t *testing.T) {
	cfg := ConsensusConfig{
		MinFrames:       3,
		MaxWindowTime:   1 * time.Second,
		ConfidenceFloor: 0.70,
	}
	agg := NewConsensusAggregator(cfg)
	trackingID := "track-001"

	// Add observations with slight noise
	// 3 good observations of "01A777AA"
	// 1 noisy observation of "01A777AB"
	agg.AddObservation(domain.PlateObservation{
		TrackingID:     trackingID,
		NormalizedText: "01A777AA",
		Confidence:     0.88,
		Sharpness:      120.0,
	}, nil)

	agg.AddObservation(domain.PlateObservation{
		TrackingID:     trackingID,
		NormalizedText: "01A777AB", // minor glitch
		Confidence:     0.65,
		Sharpness:      90.0,
	}, nil)

	// Evaluate before reaching MinFrames=3 for winner
	res, err := agg.Evaluate(trackingID)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if res.Confirmed {
		t.Errorf("expected not confirmed yet because minFrames=3, got confirmed")
	}

	// Add 2 more high-quality frames for 01A777AA
	agg.AddObservation(domain.PlateObservation{
		TrackingID:     trackingID,
		NormalizedText: "01A777AA",
		Confidence:     0.94,
		Sharpness:      140.0,
	}, nil)
	agg.AddObservation(domain.PlateObservation{
		TrackingID:     trackingID,
		NormalizedText: "01A777AA",
		Confidence:     0.92,
		Sharpness:      135.0,
	}, nil)

	// Now 3 observations of 01A777AA
	res, err = agg.Evaluate(trackingID)
	if err != nil {
		t.Fatalf("unexpected error on second evaluate: %v", err)
	}

	if !res.Confirmed {
		t.Errorf("expected confirmed=true, got false. Reason: %s", res.Reason)
	}

	if res.PlateNumber != "01A777AA" {
		t.Errorf("expected winning plate 01A777AA, got %s", res.PlateNumber)
	}

	if res.SampleCount < 3 {
		t.Errorf("expected sampleCount >= 3, got %d", res.SampleCount)
	}

	// Test cleanup
	agg.Cleanup(trackingID)
	_, err = agg.Evaluate(trackingID)
	if err == nil {
		t.Errorf("expected error after cleanup, got nil")
	}
}
