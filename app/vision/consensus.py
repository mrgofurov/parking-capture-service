import math
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional
import numpy as np
from app.vision.normalizer import normalize_plate, NormalizationResult


@dataclass
class PlateCandidate:
    raw_text: str
    normalized_text: str
    ocr_confidence: float
    plate_confidence: float
    sharpness: float
    frame: Optional[np.ndarray] = None
    timestamp: float = field(default_factory=time.time)
    plate_type: str = "UNKNOWN"
    score_bonus: float = 0.0


@dataclass
class TrackSession:
    track_id: str
    candidates: List[PlateCandidate] = field(default_factory=list)
    first_seen: float = field(default_factory=time.time)
    last_seen: float = field(default_factory=time.time)
    best_frame: Optional[np.ndarray] = None
    best_frame_score: float = -1.0
    confirmed: bool = False
    confirmed_plate: Optional[str] = None
    confirmed_confidence: float = 0.0


@dataclass
class ConsensusResult:
    confirmed: bool
    plate_number: Optional[str]
    raw_text: Optional[str]
    confidence: float
    sample_count: int
    best_frame: Optional[np.ndarray]
    format_valid: bool
    reason: str


class ConsensusAggregator:
    def __init__(
        self,
        min_frames: int = 2,
        min_confidence: float = 0.78,
        session_timeout: float = 5.0,
    ):
        self.min_frames = min_frames
        self.min_confidence = min_confidence
        self.session_timeout = session_timeout
        self.sessions: Dict[str, TrackSession] = {}

    def get_or_create_session(self, track_id: str) -> TrackSession:
        if track_id not in self.sessions:
            now = time.time()
            self.sessions[track_id] = TrackSession(
                track_id=track_id,
                first_seen=now,
                last_seen=now,
            )
        return self.sessions[track_id]

    def add_candidate(
        self,
        track_id: str,
        raw_text: str,
        ocr_confidence: float,
        plate_confidence: float,
        sharpness: float,
        frame: Optional[np.ndarray] = None,
    ) -> None:
        """Add an OCR observation to a track session and update best frame."""
        session = self.get_or_create_session(track_id)
        now = time.time()
        session.last_seen = now

        norm: NormalizationResult = normalize_plate(raw_text)
        candidate = PlateCandidate(
            raw_text=raw_text,
            normalized_text=norm.normalized,
            ocr_confidence=ocr_confidence,
            plate_confidence=plate_confidence,
            sharpness=sharpness,
            frame=frame,
            timestamp=now,
            plate_type=norm.plate_type,
            score_bonus=norm.score_bonus if norm.is_valid else 0.0,
        )
        session.candidates.append(candidate)

        # Calculate composite frame quality score
        # Combination of OCR confidence, plate detector confidence, and normalized sharpness
        sharpness_norm = min(1.0, sharpness / 100.0)
        frame_score = (
            ocr_confidence * 0.40
            + plate_confidence * 0.30
            + sharpness_norm * 0.20
            + (0.10 if norm.is_valid else 0.0)
        )

        if frame is not None and frame_score > session.best_frame_score:
            session.best_frame = frame
            session.best_frame_score = frame_score

    def evaluate(self, track_id: str) -> ConsensusResult:
        """
        Evaluate multi-frame consensus for a track using weighted voting.
        """
        if track_id not in self.sessions:
            return ConsensusResult(
                confirmed=False,
                plate_number=None,
                raw_text=None,
                confidence=0.0,
                sample_count=0,
                best_frame=None,
                format_valid=False,
                reason="Track session topilmadi",
            )

        session = self.sessions[track_id]
        if not session.candidates:
            return ConsensusResult(
                confirmed=False,
                plate_number=None,
                raw_text=None,
                confidence=0.0,
                sample_count=0,
                best_frame=session.best_frame,
                format_valid=False,
                reason="Nomzodlar mavjud emas",
            )

        # If already confirmed, return cached confirmation
        if session.confirmed and session.confirmed_plate:
            return ConsensusResult(
                confirmed=True,
                plate_number=session.confirmed_plate,
                raw_text=session.confirmed_plate,
                confidence=session.confirmed_confidence,
                sample_count=len(session.candidates),
                best_frame=session.best_frame,
                format_valid=True,
                reason="Oldin tasdiqlangan consensus",
            )

        # Group candidates by valid normalized plate
        groups: Dict[str, Dict] = {}
        total_valid = 0

        for cand in session.candidates:
            norm = normalize_plate(cand.normalized_text)
            if not norm.is_valid:
                continue

            total_valid += 1
            key = norm.normalized
            if key not in groups:
                groups[key] = {
                    "count": 0,
                    "conf_sum": 0.0,
                    "plate_conf_sum": 0.0,
                    "sharpness_sum": 0.0,
                    "max_conf": 0.0,
                    "raw_texts": [],
                    "plate_type": norm.plate_type,
                    "score_bonus": norm.score_bonus,
                }
            g = groups[key]
            g["count"] += 1
            g["conf_sum"] += cand.ocr_confidence
            g["plate_conf_sum"] += cand.plate_confidence
            g["sharpness_sum"] += cand.sharpness
            g["max_conf"] = max(g["max_conf"], cand.ocr_confidence)
            g["raw_texts"].append(cand.raw_text)

        if not groups:
            return ConsensusResult(
                confirmed=False,
                plate_number=None,
                raw_text=session.candidates[-1].raw_text,
                confidence=session.candidates[-1].ocr_confidence,
                sample_count=len(session.candidates),
                best_frame=session.best_frame,
                format_valid=False,
                reason="Barcha olingan freymlardagi raqamlar formati yaroqsiz",
            )

        # Select winning candidate based on weighted score:
        # 50% avg confidence + 25% frequency ratio + 15% detector conf + 10% sample count bonus
        best_plate = None
        best_score = -1.0
        best_group = None

        for plate, g in groups.items():
            count = g["count"]
            avg_ocr_conf = g["conf_sum"] / count
            avg_plate_conf = g["plate_conf_sum"] / count
            freq_ratio = count / total_valid
            sample_bonus = min(0.10, count * 0.02)

            score = (
                avg_ocr_conf * 0.50
                + freq_ratio * 0.25
                + avg_plate_conf * 0.15
                + sample_bonus
                + g["score_bonus"]
            )
            # Bound score to [0.0, 0.99]
            score = min(0.99, max(0.0, score))

            if score > best_score:
                best_score = score
                best_plate = plate
                best_group = g

        final_confidence = round(best_score, 4)
        has_enough_frames = best_group["count"] >= self.min_frames
        meets_confidence = final_confidence >= self.min_confidence
        raw = best_group["raw_texts"][0] if best_group["raw_texts"] else best_plate

        if has_enough_frames and meets_confidence:
            session.confirmed = True
            session.confirmed_plate = best_plate
            session.confirmed_confidence = final_confidence
            return ConsensusResult(
                confirmed=True,
                plate_number=best_plate,
                raw_text=raw,
                confidence=final_confidence,
                sample_count=best_group["count"],
                best_frame=session.best_frame,
                format_valid=True,
                reason="Multi-frame konsensus muvaffaqiyatli tasdiqlandi",
            )

        return ConsensusResult(
            confirmed=False,
            plate_number=best_plate,
            raw_text=raw,
            confidence=final_confidence,
            sample_count=best_group["count"],
            best_frame=session.best_frame,
            format_valid=True,
            reason="Yetarli freymlar soni yoki confidence threshold yetarli emas",
        )

    def cleanup_stale_sessions(self) -> int:
        """Remove sessions that have been inactive longer than session_timeout."""
        now = time.time()
        stale_keys = [
            k for k, s in self.sessions.items()
            if now - s.last_seen > self.session_timeout
        ]
        for k in stale_keys:
            del self.sessions[k]
        return len(stale_keys)
