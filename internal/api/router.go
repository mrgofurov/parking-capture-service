package api

import (
	"bufio"
	"context"
	"fmt"
	"io"

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

	v1.Get("/cameras/:id/snapshot", func(c *fiber.Ctx) error {
		camID := c.Params("id")
		b := camManager.GetLatestSnapshot(camID)
		if len(b) == 0 {
			return c.Status(fiber.StatusNotFound).SendString("Snapshot mavjud emas")
		}
		c.Set("Content-Type", "image/jpeg")
		c.Set("Cache-Control", "no-cache, no-store, must-revalidate")
		return c.Send(b)
	})

	v1.Get("/cameras/:id/stream", func(c *fiber.Ctx) error {
		camID := c.Params("id")
		ch, unsub := camManager.SubscribeStream(camID)
		if ch == nil {
			return c.Status(fiber.StatusNotFound).SendString("Kamera topilmadi")
		}

		c.Set("Content-Type", "multipart/x-mixed-replace; boundary=frame")
		c.Set("Cache-Control", "no-cache")
		c.Set("Connection", "keep-alive")
		c.Context().SetBodyStreamWriter(func(w *bufio.Writer) {
			defer unsub()
			for frameBytes := range ch {
				if len(frameBytes) == 0 {
					continue
				}
				_, err := fmt.Fprintf(w, "--frame\r\nContent-Type: image/jpeg\r\nContent-Length: %d\r\n\r\n", len(frameBytes))
				if err != nil {
					return
				}
				if _, err := w.Write(frameBytes); err != nil {
					return
				}
				if _, err := fmt.Fprintf(w, "\r\n"); err != nil {
					return
				}
				if err := w.Flush(); err != nil {
					return
				}
			}
		})
		return nil
	})

	v1.Post("/cameras/:id/test-image", func(c *fiber.Ctx) error {
		camID := c.Params("id")
		file, err := c.FormFile("image")
		if err != nil {
			return c.Status(fiber.StatusBadRequest).JSON(fiber.Map{
				"success": false,
				"message": "Rasm fayli (image form-data) yuklanmadi",
			})
		}
		f, err := file.Open()
		if err != nil {
			return c.Status(fiber.StatusInternalServerError).JSON(fiber.Map{
				"success": false,
				"message": "Faylni ochishda xatolik",
			})
		}
		defer f.Close()

		data, err := io.ReadAll(f)
		if err != nil {
			return c.Status(fiber.StatusInternalServerError).JSON(fiber.Map{
				"success": false,
				"message": "Fayl ma'lumotlarini o'qishda xatolik",
			})
		}

		res, err := camManager.TestImageOCR(camID, data)
		if err != nil {
			return c.Status(fiber.StatusBadRequest).JSON(fiber.Map{
				"success": false,
				"message": err.Error(),
			})
		}

		return c.JSON(fiber.Map{
			"success": true,
			"data":    res,
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
