package ocr

import (
	"context"

	"github.com/smart-parking/parking-capture-service/internal/domain"
)

type PlateRecognizer interface {
	Name() string
	Recognize(ctx context.Context, frame *domain.Frame, vehicle *domain.VehicleObservation) ([]domain.PlateObservation, error)
}
