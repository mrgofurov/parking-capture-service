package main

import (
	"context"
	"log"
	"os"
	"os/signal"
	"syscall"
	"time"

	"github.com/gofiber/fiber/v2"
	"github.com/smart-parking/parking-capture-service/internal/api"
	"github.com/smart-parking/parking-capture-service/internal/backend"
	"github.com/smart-parking/parking-capture-service/internal/barrier"
	"github.com/smart-parking/parking-capture-service/internal/camera"
	"github.com/smart-parking/parking-capture-service/internal/config"
	"github.com/smart-parking/parking-capture-service/internal/domain"
)

func main() {
	log.Println("==================================================")
	log.Println("     Starting Smart Parking Capture Service       ")
	log.Println("     Edge Computer Vision & Multi-Frame OCR       ")
	log.Println("==================================================")

	// 1. Config
	cfg := config.LoadConfig()
	log.Printf("[Config] Port: %s, Backend URL: %s, Synthetic mode: %v\n",
		cfg.ServicePort, cfg.BackendURL, cfg.SyntheticMode)

	// 2. Context with signal handling
	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()

	// 3. Backend Client
	backendClient := backend.NewClient(cfg.BackendURL, cfg.BackendToken, 200)

	// 4. Camera Manager
	camManager := camera.NewManager(cfg, backendClient)

	// Register Camera 1: Kirish (Lane 1 Entry)
	bEntry := barrier.NewHTTPBarrierController("entry-barrier", "", 10)
	camEntry := domain.CameraConfig{
		ID:        "cam-entry",
		Name:      "Kirish Yo'lagi Kamerasi (Lane 1)",
		ParkingID: cfg.ParkingID,
		Direction: domain.DirectionEntry,
		Enabled:   true,
		FPS:       cfg.DefaultFPS,
		BarrierID: "entry-barrier",
		VirtualLine: domain.VirtualLine{
			X1: 0, Y1: 240, X2: 640, Y2: 240,
		},
	}
	_ = camManager.RegisterCamera(camEntry, bEntry)

	// Register Camera 2: Chiqish (Lane 2 Exit)
	bExit := barrier.NewHTTPBarrierController("exit-barrier", "", 10)
	camExit := domain.CameraConfig{
		ID:        "cam-exit",
		Name:      "Chiqish Yo'lagi Kamerasi (Lane 2)",
		ParkingID: cfg.ParkingID,
		Direction: domain.DirectionExit,
		Enabled:   true,
		FPS:       cfg.DefaultFPS,
		BarrierID: "exit-barrier",
		VirtualLine: domain.VirtualLine{
			X1: 0, Y1: 240, X2: 640, Y2: 240,
		},
	}
	_ = camManager.RegisterCamera(camExit, bExit)

	// 5. Start camera capture pipelines
	camManager.StartAll(ctx)

	// 6. HTTP & Prometheus API
	app := fiber.New(fiber.Config{
		AppName:               "Smart Parking Capture Service v1.0",
		DisableStartupMessage: false,
	})

	api.SetupRouter(app, camManager)

	// Run Fiber in goroutine
	go func() {
		log.Printf("[Server] Capture service listening on :%s\n", cfg.ServicePort)
		if err := app.Listen(":" + cfg.ServicePort); err != nil {
			log.Printf("[Server] Listen ended: %v\n", err)
		}
	}()

	// 7. Wait for termination signals (Graceful Shutdown)
	sigChan := make(chan os.Signal, 1)
	signal.Notify(sigChan, os.Interrupt, syscall.SIGTERM)
	<-sigChan

	log.Println("[Server] Shutting down capture service gracefully...")
	cancel()
	camManager.StopAll()
	_ = app.ShutdownWithTimeout(3 * time.Second)
	log.Println("[Server] Capture service stopped.")
}
