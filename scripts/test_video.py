#!/usr/bin/env python3
"""
Test Video Benchmark & Accuracy Tool
Runs the complete Vision Pipeline on an input video file and prints metrics:
- Detected vehicles
- Detected plates
- Valid OCR results
- Final confirmed plates
Optional: Saves annotated debug video with vehicle & plate bounding boxes, OCR text, and FPS.
"""

import argparse
import os
import sys
import time
from pathlib import Path
import cv2
import numpy as np

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.config import settings
from app.parking.zone import ROIZone
from app.utils.logging import setup_logger
from app.vision.consensus import ConsensusAggregator
from app.vision.detector import VehicleDetector
from app.vision.normalizer import normalize_plate
from app.vision.ocr import get_ocr_engine
from app.vision.plate_detector import LicensePlateDetector
from app.vision.preprocessing import preprocess_plate
from app.vision.quality import compute_plate_quality
from app.vision.tracker import VehicleTracker

logger = setup_logger("test-video")


def main():
    parser = argparse.ArgumentParser(description="Run ANPR Vision Pipeline on a video file")
    parser.add_argument("--input", "-i", type=str, required=True, help="Path to input MP4/video file")
    parser.add_argument("--output", "-o", type=str, default=None, help="Path to save annotated debug video")
    parser.add_argument("--max-frames", "-m", type=int, default=0, help="Max frames to process (0 = all)")
    parser.add_argument("--confidence", "-c", type=float, default=0.40, help="Vehicle detection confidence")
    args = parser.parse_args()

    if not os.path.exists(args.input):
        print(f"Error: Input video not found: {args.input}")
        sys.exit(1)

    cap = cv2.VideoCapture(args.input)
    if not cap.isOpened():
        print(f"Error: Cannot open video: {args.input}")
        sys.exit(1)

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    total_video_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    print("==================================================")
    print("      Smart Parking Video Accuracy Benchmark      ")
    print("==================================================")
    print(f"Input Video:   {args.input}")
    print(f"Resolution:    {width}x{height} @ {fps:.1f} FPS")
    print(f"Total Frames:  {total_video_frames}")
    if args.output:
        print(f"Debug Output:  {args.output}")
    print("==================================================")

    # Initialize components
    detector = VehicleDetector(model_path=settings.vehicle_model, confidence_threshold=args.confidence)
    plate_detector = LicensePlateDetector(model_path=settings.plate_model, confidence_threshold=0.45)
    ocr_engine = get_ocr_engine()
    tracker = VehicleTracker()
    consensus = ConsensusAggregator(
        min_frames=settings.min_consensus_count,
        min_confidence=settings.min_final_confidence,
    )
    roi = ROIZone(
        enabled=settings.roi_enabled,
        x1=settings.roi_x1,
        y1=settings.roi_y1,
        x2=settings.roi_x2,
        y2=settings.roi_y2,
    )

    writer = None
    if args.output:
        os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(args.output, fourcc, fps, (width, height))

    # Counters
    frames_processed = 0
    vehicles_detected_count = 0
    plates_detected_count = 0
    valid_ocr_count = 0
    confirmed_plates = set()
    track_last_ocr = {}

    start_benchmark = time.time()

    while True:
        ret, frame = cap.read()
        if not ret or frame is None:
            break

        frames_processed += 1
        if args.max_frames > 0 and frames_processed > args.max_frames:
            break

        # 1. Vehicle Detection
        veh_dets = detector.detect(frame)
        valid_veh = [d for d in veh_dets if roi.is_vehicle_in_zone(d.bbox)]
        vehicles_detected_count += len(valid_veh)

        # 2. Tracking
        tracked = tracker.update(valid_veh)

        # 3. Plate detection & OCR per vehicle track
        now = time.time()
        for trk in tracked:
            last_ocr_time = track_last_ocr.get(trk.track_id, 0.0)
            if (now - last_ocr_time) * 1000.0 < settings.ocr_interval_ms:
                continue
            track_last_ocr[trk.track_id] = now

            plate_dets = plate_detector.detect_in_vehicle_crop(frame, trk.bounding_box)
            if not plate_dets:
                continue

            plates_detected_count += len(plate_dets)
            best_plate = max(plate_dets, key=lambda p: p.confidence)

            # Quality filter
            q = compute_plate_quality(best_plate.crop)
            if not q.is_acceptable:
                continue

            preprocessed = preprocess_plate(best_plate.crop)
            ocr_res = ocr_engine.recognize(preprocessed)
            if not ocr_res:
                continue

            best_ocr = max(ocr_res, key=lambda o: o.confidence)
            norm = normalize_plate(best_ocr.text)
            if norm.is_valid:
                valid_ocr_count += 1

            consensus.add_candidate(
                track_id=trk.track_id,
                raw_text=best_ocr.text,
                ocr_confidence=best_ocr.confidence,
                plate_confidence=best_plate.confidence,
                sharpness=q.sharpness,
                frame=frame,
            )

            # Check consensus
            eval_res = consensus.evaluate(trk.track_id)
            if eval_res.confirmed and eval_res.plate_number:
                confirmed_plates.add(eval_res.plate_number)

            # Annotate frame for debug video
            if writer:
                # Draw vehicle box
                vx1, vy1, vx2, vy2 = trk.bounding_box
                cv2.rectangle(frame, (vx1, vy1), (vx2, vy2), (0, 255, 0), 2)
                cv2.putText(
                    frame,
                    f"ID:{trk.track_id} {eval_res.plate_number or '...'}",
                    (vx1, max(20, vy1 - 10)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (0, 255, 0),
                    2,
                )

                # Draw plate box
                px1, py1, px2, py2 = best_plate.bbox
                cv2.rectangle(frame, (px1, py1), (px2, py2), (0, 0, 255), 2)

        if writer:
            # Draw ROI if enabled
            if settings.roi_enabled:
                cv2.rectangle(
                    frame,
                    (settings.roi_x1, settings.roi_y1),
                    (settings.roi_x2, settings.roi_y2),
                    (255, 255, 0),
                    2,
                )
            writer.write(frame)

        if frames_processed % 50 == 0:
            elapsed = time.time() - start_benchmark
            print(f"Processed {frames_processed} frames ({frames_processed / elapsed:.1f} FPS)...")

    cap.release()
    if writer:
        writer.release()

    total_time = time.time() - start_benchmark
    effective_fps = frames_processed / max(0.001, total_time)

    print("\n==================================================")
    print("                BENCHMARK RESULTS                 ")
    print("==================================================")
    print(f"Frames processed:        {frames_processed}")
    print(f"Elapsed time:            {total_time:.2f}s ({effective_fps:.1f} FPS)")
    print(f"Detected vehicles:       {vehicles_detected_count}")
    print(f"Detected plates:         {plates_detected_count}")
    print(f"Valid OCR:               {valid_ocr_count}")
    print(f"Final confirmed plates:  {len(confirmed_plates)}")
    if confirmed_plates:
        print(f"Confirmed Plate Numbers: {', '.join(sorted(list(confirmed_plates)))}")
    print("==================================================")


if __name__ == "__main__":
    main()
