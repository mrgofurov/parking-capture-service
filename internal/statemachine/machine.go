package statemachine

import (
	"log"
	"sync"
	"time"

	"github.com/smart-parking/parking-capture-service/internal/domain"
)

type StateListener func(cameraID string, oldState, newState domain.StateType, payload interface{})

type RecognitionStateMachine struct {
	cameraID       string
	currentState   domain.StateType
	activeTrackID  string
	confirmedPlate string
	lastStateTime  time.Time
	cooldownUntil  time.Time
	passageTimeout time.Duration
	listeners      []StateListener
	mu             sync.Mutex
}

func NewRecognitionStateMachine(cameraID string, passageTimeout time.Duration) *RecognitionStateMachine {
	if passageTimeout <= 0 {
		passageTimeout = 15 * time.Second
	}
	return &RecognitionStateMachine{
		cameraID:       cameraID,
		currentState:   domain.StateIdle,
		lastStateTime:  time.Now(),
		passageTimeout: passageTimeout,
		listeners:      make([]StateListener, 0),
	}
}

func (sm *RecognitionStateMachine) CurrentState() domain.StateType {
	sm.mu.Lock()
	defer sm.mu.Unlock()
	return sm.currentState
}

func (sm *RecognitionStateMachine) AddListener(l StateListener) {
	sm.mu.Lock()
	defer sm.mu.Unlock()
	sm.listeners = append(sm.listeners, l)
}

func (sm *RecognitionStateMachine) OnVehicleDetected(trackingID string) bool {
	sm.mu.Lock()
	defer sm.mu.Unlock()

	// Check Cooldown
	if time.Now().Before(sm.cooldownUntil) {
		return false
	}

	if sm.currentState == domain.StateIdle {
		sm.activeTrackID = trackingID
		sm.transitionUnlocked(domain.StateVehicleDetected, trackingID)
		sm.transitionUnlocked(domain.StateTracking, trackingID)
		return true
	}
	return false
}

func (sm *RecognitionStateMachine) OnPlateCandidate() bool {
	sm.mu.Lock()
	defer sm.mu.Unlock()

	if sm.currentState == domain.StateTracking {
		sm.transitionUnlocked(domain.StatePlateCandidate, nil)
		sm.transitionUnlocked(domain.StateRecognizing, nil)
		return true
	}
	return false
}

func (sm *RecognitionStateMachine) OnPlateConfirmed(plate string, result *domain.RecognitionResult) bool {
	sm.mu.Lock()
	defer sm.mu.Unlock()

	if sm.currentState == domain.StateRecognizing || sm.currentState == domain.StatePlateCandidate {
		sm.confirmedPlate = plate
		sm.transitionUnlocked(domain.StatePlateConfirmed, result)
		return true
	}
	return false
}

func (sm *RecognitionStateMachine) OnEventSent() {
	sm.mu.Lock()
	defer sm.mu.Unlock()

	if sm.currentState == domain.StatePlateConfirmed {
		sm.transitionUnlocked(domain.StateEventSent, nil)
		sm.transitionUnlocked(domain.StateWaitingForPassage, nil)
	}
}

func (sm *RecognitionStateMachine) OnVehiclePassed() {
	sm.mu.Lock()
	defer sm.mu.Unlock()

	if sm.currentState == domain.StateWaitingForPassage || sm.currentState == domain.StateEventSent {
		sm.transitionUnlocked(domain.StatePassed, nil)
		sm.startCooldownUnlocked(5 * time.Second)
	}
}

func (sm *RecognitionStateMachine) ResetToIdle(reason string) {
	sm.mu.Lock()
	defer sm.mu.Unlock()

	log.Printf("[StateMachine] Camera %s reset to IDLE. Reason: %s\n", sm.cameraID, reason)
	sm.activeTrackID = ""
	sm.confirmedPlate = ""
	sm.transitionUnlocked(domain.StateIdle, reason)
}

func (sm *RecognitionStateMachine) startCooldownUnlocked(duration time.Duration) {
	sm.cooldownUntil = time.Now().Add(duration)
	sm.transitionUnlocked(domain.StateCooldown, nil)

	time.AfterFunc(duration, func() {
		sm.mu.Lock()
		defer sm.mu.Unlock()
		if sm.currentState == domain.StateCooldown {
			sm.activeTrackID = ""
			sm.confirmedPlate = ""
			sm.transitionUnlocked(domain.StateIdle, nil)
		}
	})
}

func (sm *RecognitionStateMachine) transitionUnlocked(newState domain.StateType, payload interface{}) {
	oldState := sm.currentState
	sm.currentState = newState
	sm.lastStateTime = time.Now()

	for _, l := range sm.listeners {
		go l(sm.cameraID, oldState, newState, payload)
	}
}
