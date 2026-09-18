# Smart Parking — Edge Capture Service (`parking-capture-service`)

> **Python 3.11+ | FastAPI | Ultralytics YOLO | PaddleOCR | ByteTrack | PyTorch**

Ushbu mikroxizmat IP/RTSP kameralar yoki video fayllar orqali avtoturargohga kirish va chiqish yo'laklaridagi avtomobillarni real-vaqt rejimida aniqlash, O'zbekiston davlat raqamlarini ko'p freymli konsensus (multi-frame OCR) bilan o'qish, takroriy xabarlarni cheklash (debounce/cooldown) va hodisalarni `parking-backend` servisiga yuborish uchun mo'ljallangan.

---

## 1. Pipeline Architecture

```text
                    RTSP / IP Camera (yoki Video File)
                                  │
                                  ▼
                         Stream Reader Thread
                      (Zero-Latency Ring Buffer)
                                  │
                                  ▼
                            Frame Sampling
                             (Target FPS)
                                  │
                                  ▼
                           Lane ROI Filter
                                  │
                                  ▼
                       Vehicle Detection (YOLO)
                       (car, truck, bus, moto)
                                  │
                                  ▼
                           Vehicle Tracking
                       (ByteTrack / IoU Tracker)
                                  │
                                  ▼
                    License Plate Detection (YOLO)
                                  │
                                  ▼
                     Plate Crop & Quality Scoring
                    (Laplacian Sharpness, Size, Blur)
                                  │
                                  ▼
                         Image Preprocessing
                   (CLAHE, Sharpen, Deskew, Resize)
                                  │
                                  ▼
                             PaddleOCR
                                  │
                                  ▼
                       Uzbekistan Plate Normalizer
                   (Positional Ambiguity: O/0, I/1, B/8)
                                  │
                                  ▼
                      Multi-frame Consensus & Best Frame
                                  │
                                  ▼
                       Duplicate Protection (Cooldown)
                                  │
                                  ▼
                      Parking Backend Dispatch (HTTPX)
                       (Exponential Backoff & Offline Queue)
```

---

## 2. Asosiy Imkoniyatlar

1. **RTSP & Video File Abstraction**:
   - RTSP oqimlarini fon oqimida (background thread) qabul qiladi va kadrlar yig'ilib kechikish (latency) hosil bo'lishini oldini oladi.
   - Aloqa uzilganda eksponentsial kutish (exponential backoff with jitter) bilan avtomatik qayta ulanadi.
   - Benchmark va testlar uchun MP4/video fayllarni uzluksiz siklda (loop) o'qiy oladi (`VIDEO_SOURCE=file`).

2. **Computer Vision & Tracking**:
   - Ultralytics YOLO orqali transport vositalarini aniqlash.
   - Avtomobil bounding box'i ichidan davlat raqamini qirqib olish (hierarchical plate detection).
   - Har bir transport vositasining yo'lakdagi harakatini kuzatish (`track_id`).
   - Kamera oldida keraksiz sohalarni chiqarib tashlash uchun ROI (Region of Interest) zonasi.

3. **Plate Quality & Preprocessing**:
   - Kichik (`< 80x20`) yoki xira/blur kadrlarni OCR'ga yubormaydi (CPU/GPU resurslarini tejaydi).
   - Kontrastni kuchaytirish (CLAHE), ochartirish/to'g'rilash (deskew) va keskinlik berish (sharpening).

4. **PaddleOCR & Uzbekistan Davlat Raqamlari Normalizatsiyasi**:
   - PaddleOCR startup vaqtida bir marta yuklanadi (har freymda qayta yuklanmaydi).
   - O'zbekiston jismoniy (`01A123BC`) va yuridik (`01123ABC`) shaxslar formatini tekshiradi.
   - Pozitsion simvol xatoliklarini to'g'rilaydi: `O` $\leftrightarrow$ `0`, `I` $\leftrightarrow$ `1`, `B` $\leftrightarrow$ `8`, `S` $\leftrightarrow$ `5`, `Z` $\leftrightarrow$ `2`.

5. **Multi-Frame Consensus & Eng Yaxshi Freym (Best Frame)**:
   - Bitta freymga ishonib xato event chiqarmaydi; kamida `MIN_CONSENSUS_COUNT` ta freymdan tasdiq kutadi.
   - O'qish sifati, aniqlik darajasi va Laplacian aniqligi bo'yicha eng tiniq freymni ajratib oladi va saqlaydi.

6. **Deduplikatsiya va Backend Chidamliligi**:
   - Avtomobil kamera oldida turib qolsa, takroriy xabarlar yuborilmaydi (`EVENT_COOLDOWN_SECONDS=10`).
   - Markaziy `parking-backend` uzilib qolsa, hodisalar xotiradagi chegaralangan navbatda saqlanadi va aloqa tiklanganda jo'natiladi.

7. **Kuzatuv va Metrikalar**:
   - Prometheus metrikalari: `/metrics`.
   - Sog'liqni tekshirish: `/health`, `/ready`, `/health/live`, `/health/ready`.
   - Jonli kamera diagnostikasi: `/api/v1/cameras/{id}/snapshot` va `/stream` (MJPEG stream).

---

## 3. O'rnatish va Ishga Tushirish

### Talablar
- Python 3.11+
- FFmpeg o'rnatilgan bo'lishi kerak (`apt-get install ffmpeg libgl1`)
- NVIDIA GPU (agar mavjud bo'lsa, CUDA bilan avtomatik ishlaydi; aks holda CPU fallback)

### Mahalliy (Local Python)

```bash
# 1. Virtual environment yaratish
python3 -m venv .venv
source .venv/bin/activate

# 2. Bog'liqliklarni o'rnatish
pip install --upgrade pip
pip install -r requirements.txt

# 3. Konfiguratsiya
cp .env.example .env

# 4. Testlarni yurgazish
pytest tests/ -v

# 5. Xizmatni ishga tushirish
python -m app.main
```

Xizmat sukut bo'yicha `http://localhost:8086` manzilida ishga tushadi.

---

## 4. Docker orqali ishga tushirish

### CPU rejimi:
```bash
docker compose up -d --build
```

### NVIDIA GPU rejimi (CUDA):
`docker-compose.yml` faylida GPU bo'limini yoqing:
```yaml
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: all
              capabilities: [gpu]
```

---

## 5. Konfiguratsiya Parametrlari (`.env`)

| Parametr | Standart qiymat | Tavsif |
| :--- | :--- | :--- |
| `SERVICE_PORT` | `8086` | FastAPI server porti |
| `CAMERA_ID` | `entry-01` | Kamera identifikatori |
| `CAMERA_DIRECTION` | `entry` | Yo'nalish (`entry` yoki `exit`) |
| `VIDEO_SOURCE` | `rtsp` | Video manbasi (`rtsp` yoki `file`) |
| `RTSP_URL` | `rtsp://...` | Kamera RTSP URL manzili |
| `VIDEO_FILE` | `/data/test.mp4` | Agar `VIDEO_SOURCE=file` bo'lsa video fayl yo'li |
| `TARGET_FPS` | `10` | Freymlarni tahlil qilish tezligi |
| `ROI_ENABLED` | `false` | Qiziqish zonasini yoqish/o'chirish |
| `VEHICLE_MODEL` | `yolov8n.pt` | YOLO avtomobil detektori modeli |
| `PLATE_MODEL` | `yolov8n.pt` | YOLO davlat raqami detektori modeli |
| `MIN_PLATE_WIDTH` | `80` | Minimal raqam eni (px) |
| `MIN_PLATE_HEIGHT` | `20` | Minimal raqam balandligi (px) |
| `MIN_SHARPNESS_SCORE` | `25.0` | Laplacian blur filtri chegarasi |
| `OCR_INTERVAL_MS` | `200` | Bitta avtomobil uchun OCR chaqirish oraliq vaqti (ms) |
| `MIN_CONSENSUS_COUNT` | `2` | Konsensus uchun talab qilinadigan minimal freymlar |
| `MIN_FINAL_CONFIDENCE`| `0.78` | Yakuniy tasdiqlash ishonchlilik chegarasi |
| `EVENT_COOLDOWN_SECONDS` | `10` | Bir xil raqam uchun takrorlanishdan himoya vaqti |
| `BACKEND_URL` | `http://localhost:8085`| `parking-backend` manzili |
| `BACKEND_EVENT_PATH` | `/api/v1/parking/camera-events` | Hodisalarni qabul qiluvchi endpoint |

---

## 6. Video Benchmark va Aniqlik Sinovi (`scripts/test_video.py`)

Haqiqiy video fayl orqali pipeline aniqligini sinab ko'rish:

```bash
# Sinovni boshlash
python scripts/test_video.py --input /yo'l/video.mp4

# Natijani vizual debug video sifatida saqlash
python scripts/test_video.py --input /yo'l/video.mp4 --output output/debug.mp4
```

Chiqadigan hisobot:
```text
==================================================
                BENCHMARK RESULTS                 
==================================================
Frames processed:        450
Elapsed time:            28.40s (15.8 FPS)
Detected vehicles:       82
Detected plates:         79
Valid OCR:               74
Final confirmed plates:  71
Confirmed Plate Numbers: 01A123BC, 01B456DD, 10X777AA
==================================================
```

---

## 7. API Endpoints

| Metod | Manzil | Tavsif |
| :--- | :--- | :--- |
| `GET` | `/health` | To'liq sog'liq holati (kamera, FPS, GPU, so'nggi freym) |
| `GET` | `/ready` | Readiness tekshiruvi |
| `GET` | `/metrics` | Prometheus monitoring metrikalari |
| `GET` | `/api/v1/cameras` | Kameralar ro'yxati va joriy statusi |
| `GET` | `/api/v1/cameras/{id}/snapshot` | So'nggi olingan kadr (JPEG) |
| `GET` | `/api/v1/cameras/{id}/stream` | Jonli video oqimi (MJPEG stream brauzer uchun) |
| `POST`| `/api/v1/cameras/{id}/inject-plate` | Test uchun istalgan raqamni simulyatsiya qilish |
| `POST`| `/api/v1/cameras/{id}/test-image` | Yuklangan rasmda OCR va preprocessing testini o'tkazish |

---

## 8. Backendga Yuboriladigan Hodisa (Event Payload)

```json
{
  "camera_id": "entry-01",
  "direction": "entry",
  "plate_number": "01A123BC",
  "confidence": 0.9345,
  "detected_at": "2026-09-18T10:20:30Z",
  "timestamp": "2026-09-18T10:20:30Z",
  "snapshot_base64": "/9j/4AAQSkZJRgABAQ..."
}
```
