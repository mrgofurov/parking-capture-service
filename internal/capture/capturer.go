package capture

import (
	"context"

	"github.com/smart-parking/parking-capture-service/internal/domain"
)

type FrameCapturer interface {
	CameraID() string
	Start(ctx context.Context, out chan<- *domain.Frame) error
	Stop() error
	Status() domain.CameraStatus
}
