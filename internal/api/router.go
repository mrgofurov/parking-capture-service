package api

import (
	"context"

	"github.com/gofiber/adaptor/v2"
	"github.com/gofiber/fiber/v2"
	"github.com/gofiber/fiber/v2/middleware/logger"
	"github.com/gofiber/fiber/v2/middleware/recover"
	"github.com/prometheus/client_golang/prometheus/promhttp"
	"github.com/smart-parking/parking-capture-service/internal/camera"
)

func SetupRouter(app *fiber.App, camManager *camera.Manager) {
	app.Use(recover.New())
	app.Use(logger.New(logger.Config{
		Format: "[CAPTURE] ${status} - ${latency} ${method} ${path}\n",
	}))

	// Health Endpoints
	app.Get("/health/live", func(c *fiber.Ctx) error {
		return c.JSON(fiber.Map{"status": "UP", "service": "parking-capture-service"})
	})

	app.Get("/health/ready", func(c *fiber.Ctx) error {
		return c.JSON(fiber.Map{"status": "READY", "service": "parking-capture-service"})
	})

	// Prometheus Metrics Endpoint
	app.Get("/metrics", adaptor.HTTPHandler(promhttp.Handler()))

	// Diagnostics and Device Control API
	v1 := app.Group("/api/v1")

	v1.Get("/cameras", func(c *fiber.Ctx) error {
		return c.JSON(fiber.Map{
			"success": true,
			"data":    camManager.ListCameras(),
		})
	})

	type InjectPlateRequest struct {
		PlateNumber string `json:"plate_number"`
	}

	v1.Post("/cameras/:id/inject-plate", func(c *fiber.Ctx) error {
		camID := c.Params("id")
		var req InjectPlateRequest
		if err := c.BodyParser(&req); err != nil || req.PlateNumber == "" {
			return c.Status(fiber.StatusBadRequest).JSON(fiber.Map{
				"success": false,
				"message": "Avtomobil raqami (plate_number) talab qilinadi",
			})
		}

		if err := camManager.InjectTestPlate(camID, req.PlateNumber); err != nil {
			return c.Status(fiber.StatusBadRequest).JSON(fiber.Map{
				"success": false,
				"message": err.Error(),
			})
		}

		return c.JSON(fiber.Map{
			"success": true,
			"message": "Avtomobil o'tishi muvaffaqiyatli simulyatsiya qilindi",
		})
	})

	v1.Get("/barriers", func(c *fiber.Ctx) error {
		return c.JSON(fiber.Map{
			"success": true,
			"data":    camManager.ListBarriers(),
		})
	})

	v1.Post("/barriers/:id/open", func(c *fiber.Ctx) error {
		barrierID := c.Params("id")
		b := camManager.GetBarrier(barrierID)
		if b == nil {
			return c.Status(fiber.StatusNotFound).JSON(fiber.Map{
				"success": false,
				"message": "Barrier topilmadi",
			})
		}

		_ = b.Open(context.Background(), "manual_api", "API orqali ochildi")
		return c.JSON(fiber.Map{
			"success": true,
			"message": "Shlagbaum ochildi",
		})
	})

	v1.Post("/barriers/:id/close", func(c *fiber.Ctx) error {
		barrierID := c.Params("id")
		b := camManager.GetBarrier(barrierID)
		if b == nil {
			return c.Status(fiber.StatusNotFound).JSON(fiber.Map{
				"success": false,
				"message": "Barrier topilmadi",
			})
		}

		_ = b.Close(context.Background(), "manual_api", "API orqali yopildi")
		return c.JSON(fiber.Map{
			"success": true,
			"message": "Shlagbaum yopildi",
		})
	})
}
