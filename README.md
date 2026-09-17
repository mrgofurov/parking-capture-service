# Smart Parking — Edge Capture Service (`parking-capture-service`)

Ushbu mikroxizmat oddiy IP (RTSP) kameralar orqali avtoturargohga kirish va chiqish yo'laklaridagi avtomobillarni aniqlash, O'zbekiston davlat raqamlarini ko'p freymli konsensus (multi-frame OCR) bilan o'qish, shlagbaumlarni xavfsiz boshqarish va hodisalarni markaziy `parking-backend` ga jo'natish uchun mo'ljallangan.

---

## Asosiy Imkoniyatlari

1. **RTSP & Synthetic Capture Adapterlari**:
   - Haqiqiy RTSP kamera oqimlarini FFmpeg orqali avtomatik qayta ulanish (reconnect with exponential backoff) bilan qabul qiladi.
   - RTSP kameralar bo'lmagan sharoitda test o'tkazish uchun **Synthetic / Replay Generator** mavjud (`SYNTHETIC_MODE=true`).
2. **Kadrlar Tanlash va Traektoriya (CV Pipeline)**:
   - 5-10 FPS bilan resurslarni tejagan holda freymlarni tahlil qiladi (bounded buffer).
   - Harakatdagi transport vositasini kuzatib boradi (`TrackingID`).
   - Virtual kirish/chiqish chizig'ini kesib o'tishni aniqlaydi.
   - Eng yuqori sifatli va aniq (sharpness/Laplacian/contrast) freymni avtomatik ajratib oladi.
3. **Multi-Frame Consensus OCR Engine**:
   - Bitta freymdagi tasodifiy xatoliklarga tayanmay, vaqtinchalik oyna (temporal window) ichida bir nechta freymni taqqoslaydi.
   - O'zbekiston davlat raqami formati bo'yicha simvollardagi noaniqliklarni avtomatik to'g'rilaydi (`0` vs `O`, `1` vs `I`, `5` vs `S`, `8` vs `B`).
4. **Xavfsiz Holat Mashinasi (State Machine)**:
   - `IDLE -> VEHICLE_DETECTED -> TRACKING -> PLATE_CANDIDATE -> RECOGNIZING -> PLATE_CONFIRMED -> EVENT_SENT -> WAITING_FOR_PASSAGE -> PASSED -> COOLDOWN -> IDLE`.
   - Qayta-qayta takroriy hodisa jo'natish va shlagbaumni qayta ochishni (flapping) qat'iy cheklaydi.
5. **Idempotent Shlagbaum Boshqaruvi**:
   - `OPEN`, `CLOSE` buyruqlari takroriy jo'natilmaydi.
   - Avtomobil o'tib ketgandan so'ng yoki timeout bo'lganda shlagbaum avtomatik va xavfsiz yopiladi.
6. **Tarmoq Chidamliligi (Offline Resilience)**:
   - Agar `parking-backend` vaqtincha uzilsa, hodisalar xotiradagi chegaralangan navbatda saqlanadi va qayta yuboriladi.
7. **Observability**:
   - Prometheus metrikalari: `/metrics` (`capture_fps`, `plate_detection_total`, `plate_ocr_latency_seconds`, `barrier_operations_total`).
   - Sog'liqni tekshirish: `/health/live` va `/health/ready`.

---

## O'rnatish va Ishga Tushirish

### 1. Mahalliy (Local Go) orqali ishga tushirish:
```bash
cd parking-capture-service

# Muhit o'zgaruvchilarini sozlash
cp .env.example .env

# Testlarni yurgazish
go test -v ./...

# Xizmatni ishga tushirish
go run cmd/server/main.go
```

Xizmat sukut bo'yicha `:8086` portida ishga tushadi.

### 2. Docker orqali ishga tushirish:
```bash
docker compose up -d --build
```

---

## API Endpoints

| Metod | Manzil | Tavsif |
| :--- | :--- | :--- |
| `GET` | `/health/live` | Liveness health check |
| `GET` | `/health/ready` | Readiness health check |
| `GET` | `/metrics` | Prometheus monitoring metrikalari |
| `GET` | `/api/v1/cameras` | Barcha sozlangan kameralar holati |
| `POST` | `/api/v1/cameras/:id/inject-plate` | Test uchun istalgan raqamni simulyatsiya qilish |
| `GET` | `/api/v1/barriers` | Shlagbaumlar holati |
| `POST` | `/api/v1/barriers/:id/trigger` | Shlagbaumni qo'lda ochish/yopish sinovi |
| `POST` | `/api/v1/replay/run` | Sinov bench-testini o'tkazish |

### Test uchun davlat raqamini kiritish (Manual Plate Injection):
```bash
curl -X POST http://localhost:8086/api/v1/cameras/cam-entry/inject-plate \
  -H "Content-Type: application/json" \
  -d '{"plate_number": "01A777AA"}'
```
