package config

import (
	"os"
	"strconv"

	"github.com/joho/godotenv"
)

type Config struct {
	ServicePort              string
	BackendURL               string
	BackendToken             string
	ParkingID                string
	SyntheticMode            bool
	DefaultFPS               int
	PlateConfidenceThreshold float64
	OCRConfidenceThreshold   float64
	ConsensusMinFrames       int
	RecognitionTimeoutMs     int
	CooldownSec              int
	BarrierPassageTimeoutSec int
	MaxFrameQueue            int
	InferenceWorkers         int
	StorageBestFrame         bool
	UploadDir                string
}

func LoadConfig() *Config {
	_ = godotenv.Load()

	servicePort := getEnv("SERVICE_PORT", "8086")
	backendURL := getEnv("BACKEND_URL", "http://localhost:8085")
	backendToken := getEnv("BACKEND_TOKEN", "")
	parkingID := getEnv("PARKING_ID", "main-parking")
	syntheticMode := getEnvAsBool("SYNTHETIC_MODE", true)
	defaultFPS := getEnvAsInt("DEFAULT_FPS", 8)
	plateConfThresh := getEnvAsFloat("PLATE_CONFIDENCE_THRESHOLD", 0.75)
	ocrConfThresh := getEnvAsFloat("OCR_CONFIDENCE_THRESHOLD", 0.80)
	consensusMinFrames := getEnvAsInt("CONSENSUS_MIN_FRAMES", 3)
	recognitionTimeoutMs := getEnvAsInt("RECOGNITION_TIMEOUT_MS", 1500)
	cooldownSec := getEnvAsInt("COOLDOWN_SEC", 5)
	barrierPassageTimeoutSec := getEnvAsInt("BARRIER_PASSAGE_TIMEOUT_SEC", 15)
	maxFrameQueue := getEnvAsInt("MAX_FRAME_QUEUE", 50)
	inferenceWorkers := getEnvAsInt("INFERENCE_WORKERS", 4)
	storageBestFrame := getEnvAsBool("STORAGE_BEST_FRAME", true)
	uploadDir := getEnv("UPLOAD_DIR", "./snapshots")

	_ = os.MkdirAll(uploadDir, 0755)

	return &Config{
		ServicePort:              servicePort,
		BackendURL:               backendURL,
		BackendToken:             backendToken,
		ParkingID:                parkingID,
		SyntheticMode:            syntheticMode,
		DefaultFPS:               defaultFPS,
		PlateConfidenceThreshold: plateConfThresh,
		OCRConfidenceThreshold:   ocrConfThresh,
		ConsensusMinFrames:       consensusMinFrames,
		RecognitionTimeoutMs:     recognitionTimeoutMs,
		CooldownSec:              cooldownSec,
		BarrierPassageTimeoutSec: barrierPassageTimeoutSec,
		MaxFrameQueue:            maxFrameQueue,
		InferenceWorkers:         inferenceWorkers,
		StorageBestFrame:         storageBestFrame,
		UploadDir:                uploadDir,
	}
}

func getEnv(key, defaultVal string) string {
	if val := os.Getenv(key); val != "" {
		return val
	}
	return defaultVal
}

func getEnvAsInt(key string, defaultVal int) int {
	if valStr := os.Getenv(key); valStr != "" {
		if v, err := strconv.Atoi(valStr); err == nil {
			return v
		}
	}
	return defaultVal
}

func getEnvAsFloat(key string, defaultVal float64) float64 {
	if valStr := os.Getenv(key); valStr != "" {
		if v, err := strconv.ParseFloat(valStr, 64); err == nil {
			return v
		}
	}
	return defaultVal
}

func getEnvAsBool(key string, defaultVal bool) bool {
	if valStr := os.Getenv(key); valStr != "" {
		if v, err := strconv.ParseBool(valStr); err == nil {
			return v
		}
	}
	return defaultVal
}
