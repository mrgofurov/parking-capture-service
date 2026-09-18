import pytest
from app.vision.normalizer import (
    clean_plate_text,
    force_digit,
    force_letter,
    normalize_plate,
)


def test_clean_plate_text():
    assert clean_plate_text(" 01 a 123 bc ") == "01A123BC"
    assert clean_plate_text("01-A-123-BC") == "01A123BC"
    assert clean_plate_text("01.123.ABC") == "01123ABC"
    assert clean_plate_text("") == ""


def test_individual_plate_direct_match():
    res = normalize_plate("01A123BC")
    assert res.is_valid is True
    assert res.normalized == "01A123BC"
    assert res.plate_type == "INDIVIDUAL"
    assert res.score_bonus > 0


def test_legal_plate_direct_match():
    res = normalize_plate("01123ABC")
    assert res.is_valid is True
    assert res.normalized == "01123ABC"
    assert res.plate_type == "LEGAL"
    assert res.score_bonus > 0


def test_positional_character_confusion_correction():
    # 'O' instead of '0' in region code (index 0), '0' instead of 'O' in letter (index 2)
    # OCR reads: "O1O123BC" -> Should become "01O123BC"
    res1 = normalize_plate("O1O123BC")
    assert res1.is_valid is True
    assert res1.normalized == "01O123BC"
    assert res1.plate_type == "INDIVIDUAL"

    # 'B' instead of '8' in digits, '8' instead of 'B' in letter
    # OCR reads: "01A12B8C" -> Should correct 'B' at index 5 to '8', and '8' at index 6 to 'B'
    res2 = normalize_plate("01A12B8C")
    assert res2.is_valid is True
    assert res2.normalized == "01A128BC"

    # 'I' instead of '1' in region code
    res3 = normalize_plate("I0A777AA")
    assert res3.is_valid is True
    assert res3.normalized == "10A777AA"

    # 'S' instead of '5' in digits
    res4 = normalize_plate("01A1S3BC")
    assert res4.is_valid is True
    assert res4.normalized == "01A153BC"

    # 'Z' instead of '2' in digits
    res5 = normalize_plate("01A1Z3BC")
    assert res5.is_valid is True
    assert res5.normalized == "01A123BC"


def test_invalid_plates_filtered():
    invalid_cases = [
        "ABC",
        "123",
        "HELLO",
        "XXXXXX",
        "1234567890123",
        "",
        "PARKING",
    ]
    for case in invalid_cases:
        res = normalize_plate(case)
        assert res.is_valid is False, f"Expected {case} to be invalid"
        assert res.plate_type == "INVALID"
