package statemachine

import (
	"testing"
	"time"

	"github.com/smart-parking/parking-capture-service/internal/domain"
)

func TestRecognitionStateMachine_Transitions(t *testing.T) {
	sm := NewRecognitionStateMachine("cam-entry", 2*time.Second)

	if sm.CurrentState() != domain.StateIdle {
		t.Fatalf("expected initial state IDLE, got %s", sm.CurrentState())
	}

	// 1. Vehicle Detected
	ok := sm.OnVehicleDetected("track-101")
	if !ok {
		t.Fatalf("expected OnVehicleDetected to return true")
	}
	if sm.CurrentState() != domain.StateTracking {
		t.Errorf("expected state TRACKING, got %s", sm.CurrentState())
	}

	// Cannot trigger vehicle detected again while tracking
	if sm.OnVehicleDetected("track-102") {
		t.Errorf("expected OnVehicleDetected to return false while already active")
	}

	// 2. Plate Candidate
	ok = sm.OnPlateCandidate()
	if !ok {
		t.Fatalf("expected OnPlateCandidate to return true")
	}
	if sm.CurrentState() != domain.StateRecognizing {
		t.Errorf("expected state RECOGNIZING, got %s", sm.CurrentState())
	}

	// 3. Plate Confirmed
	res := &domain.RecognitionResult{
		PlateNumber:       "01A777AA",
		OverallConfidence: 0.95,
	}
	ok = sm.OnPlateConfirmed("01A777AA", res)
	if !ok {
		t.Fatalf("expected OnPlateConfirmed to return true")
	}
	if sm.CurrentState() != domain.StatePlateConfirmed {
		t.Errorf("expected state PLATE_CONFIRMED, got %s", sm.CurrentState())
	}

	// 4. Event Sent
	sm.OnEventSent()
	if sm.CurrentState() != domain.StateWaitingForPassage {
		t.Errorf("expected state WAITING_FOR_PASSAGE, got %s", sm.CurrentState())
	}

	// 5. Vehicle Passed -> enters Cooldown
	sm.OnVehiclePassed()
	if sm.CurrentState() != domain.StateCooldown {
		t.Errorf("expected state COOLDOWN, got %s", sm.CurrentState())
	}

	// 6. Reset to Idle
	sm.ResetToIdle("manual reset")
	if sm.CurrentState() != domain.StateIdle {
		t.Errorf("expected state IDLE after reset, got %s", sm.CurrentState())
	}
}
