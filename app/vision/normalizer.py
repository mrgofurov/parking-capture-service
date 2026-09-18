import re
from dataclasses import dataclass
from typing import Optional


# Regex for Uzbekistan License Plates
# 1. Standard Individual: 2 digits (01-95) + 1 letter (A-Z) + 3 digits + 2 letters (A-Z) -> 01A123BC (8 chars)
REGEX_INDIVIDUAL = re.compile(r"^(\d{2})([A-Z])(\d{3})([A-Z]{2})$")

# 2. Legal Entity: 2 digits (01-95) + 3 digits + 3 letters (A-Z) -> 01123ABC (8 chars)
REGEX_LEGAL = re.compile(r"^(\d{2})(\d{3})([A-Z]{3})$")

# 3. General Fallback: 2 digits + 5-6 alphanumeric characters (7-8 chars)
REGEX_GENERAL = re.compile(r"^(\d{2})[A-Z0-9]{5,6}$")


@dataclass
class NormalizationResult:
    original: str
    normalized: str
    is_valid: bool
    plate_type: str  # "INDIVIDUAL", "LEGAL", "GENERAL", "INVALID"
    score_bonus: float = 0.0


# Confusions mapping
LETTER_TO_DIGIT = {
    "O": "0", "Q": "0", "D": "0",
    "I": "1", "L": "1", "J": "1",
    "Z": "2",
    "S": "5",
    "B": "8",
    "G": "6",
}

DIGIT_TO_LETTER = {
    "0": "O",
    "1": "I",
    "2": "Z",
    "5": "S",
    "8": "B",
    "6": "G",
}


def force_digit(char: str) -> str:
    """Convert commonly confused letters to digits if position requires a digit."""
    return LETTER_TO_DIGIT.get(char, char)


def force_letter(char: str) -> str:
    """Convert commonly confused digits to letters if position requires a letter."""
    return DIGIT_TO_LETTER.get(char, char)


def clean_plate_text(raw: str) -> str:
    """Remove spaces, hyphens, dots and special characters, and convert to uppercase."""
    if not raw:
        return ""
    upper = raw.upper().strip()
    return "".join(c for c in upper if c.isalnum())


def normalize_plate(raw_text: str) -> NormalizationResult:
    """
    Normalize and validate Uzbekistan license plate text with position-aware OCR correction.
    """
    cleaned = clean_plate_text(raw_text)

    if len(cleaned) < 7 or len(cleaned) > 9:
        return NormalizationResult(
            original=raw_text,
            normalized=cleaned,
            is_valid=False,
            plate_type="INVALID",
            score_bonus=0.0,
        )

    # 1. Check direct match for Standard Individual Plate (01A123BC)
    if REGEX_INDIVIDUAL.match(cleaned):
        return NormalizationResult(
            original=raw_text,
            normalized=cleaned,
            is_valid=True,
            plate_type="INDIVIDUAL",
            score_bonus=0.15,
        )

    # 2. Check direct match for Legal Entity Plate (01123ABC)
    if REGEX_LEGAL.match(cleaned):
        return NormalizationResult(
            original=raw_text,
            normalized=cleaned,
            is_valid=True,
            plate_type="LEGAL",
            score_bonus=0.15,
        )

    chars = list(cleaned)

    # 3. Positional ambiguity correction for 8-character plates
    if len(chars) == 8:
        # Candidate A: Try Individual format (01 A 123 BC)
        # indices 0,1 -> digit; index 2 -> letter; indices 3,4,5 -> digit; indices 6,7 -> letter
        indiv = [
            force_digit(chars[0]),
            force_digit(chars[1]),
            force_letter(chars[2]),
            force_digit(chars[3]),
            force_digit(chars[4]),
            force_digit(chars[5]),
            force_letter(chars[6]),
            force_letter(chars[7]),
        ]
        candidate_indiv = "".join(indiv)
        if REGEX_INDIVIDUAL.match(candidate_indiv):
            return NormalizationResult(
                original=raw_text,
                normalized=candidate_indiv,
                is_valid=True,
                plate_type="INDIVIDUAL",
                score_bonus=0.10,
            )

        # Candidate B: Try Legal format (01 123 ABC)
        # indices 0,1 -> digit; indices 2,3,4 -> digit; indices 5,6,7 -> letter
        legal = [
            force_digit(chars[0]),
            force_digit(chars[1]),
            force_digit(chars[2]),
            force_digit(chars[3]),
            force_digit(chars[4]),
            force_letter(chars[5]),
            force_letter(chars[6]),
            force_letter(chars[7]),
        ]
        candidate_legal = "".join(legal)
        if REGEX_LEGAL.match(candidate_legal):
            return NormalizationResult(
                original=raw_text,
                normalized=candidate_legal,
                is_valid=True,
                plate_type="LEGAL",
                score_bonus=0.10,
            )

    # 4. Fallback General format (e.g. 7-character regional or trailers)
    if REGEX_GENERAL.match(cleaned):
        return NormalizationResult(
            original=raw_text,
            normalized=cleaned,
            is_valid=True,
            plate_type="GENERAL",
            score_bonus=0.05,
        )

    # 5. Invalid / Non-plate text (e.g. "ABC", "123", "HELLO", "XXXXXX")
    return NormalizationResult(
        original=raw_text,
        normalized=cleaned,
        is_valid=False,
        plate_type="INVALID",
        score_bonus=0.0,
    )
