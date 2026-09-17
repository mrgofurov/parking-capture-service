package ocr

import (
	"regexp"
	"strings"
	"unicode"
)

// Uzbek Standard Plate: 2 digits (region) + 1 letter + 3 digits + 2 letters, e.g. 01A123BC
var regexIndividualPlate = regexp.MustCompile(`^(\d{2})([A-Z])(\d{3})([A-Z]{2})$`)

// Uzbek Legal Entity Plate: 2 digits + 3 digits + 3 letters, e.g. 01123ABC
var regexLegalPlate = regexp.MustCompile(`^(\d{2})(\d{3})([A-Z]{3})$`)

// General fallback plate
var regexGeneralPlate = regexp.MustCompile(`^(\d{2})[A-Z0-9]{5,6}$`)

type NormalizationResult struct {
	Original   string
	Normalized string
	IsValid    bool
	PlateType  string // "INDIVIDUAL", "LEGAL", "GENERAL"
	ScoreBonus float64
}

func NormalizePlate(raw string) NormalizationResult {
	clean := strings.ToUpper(strings.TrimSpace(raw))
	// Remove all whitespace, hyphens, dots
	var sb strings.Builder
	for _, r := range clean {
		if unicode.IsLetter(r) || unicode.IsDigit(r) {
			sb.WriteRune(r)
		}
	}
	cleaned := sb.String()

	if len(cleaned) < 7 || len(cleaned) > 9 {
		return NormalizationResult{
			Original:   raw,
			Normalized: cleaned,
			IsValid:    false,
			PlateType:  "INVALID",
			ScoreBonus: 0.0,
		}
	}

	// 1. Try direct match
	if regexIndividualPlate.MatchString(cleaned) {
		return NormalizationResult{
			Original:   raw,
			Normalized: cleaned,
			IsValid:    true,
			PlateType:  "INDIVIDUAL",
			ScoreBonus: 0.15,
		}
	}
	if regexLegalPlate.MatchString(cleaned) {
		return NormalizationResult{
			Original:   raw,
			Normalized: cleaned,
			IsValid:    true,
			PlateType:  "LEGAL",
			ScoreBonus: 0.15,
		}
	}

	// 2. Position-aware character ambiguity correction (0 vs O, 1 vs I, 5 vs S, 8 vs B)
	runes := []rune(cleaned)
	if len(runes) == 8 {
		// Individual format: [0,1] digits, [2] letter, [3,4,5] digits, [6,7] letters
		corr := make([]rune, 8)
		copy(corr, runes)

		corr[0] = forceDigit(corr[0])
		corr[1] = forceDigit(corr[1])
		corr[2] = forceLetter(corr[2])
		corr[3] = forceDigit(corr[3])
		corr[4] = forceDigit(corr[4])
		corr[5] = forceDigit(corr[5])
		corr[6] = forceLetter(corr[6])
		corr[7] = forceLetter(corr[7])

		candidate := string(corr)
		if regexIndividualPlate.MatchString(candidate) {
			return NormalizationResult{
				Original:   raw,
				Normalized: candidate,
				IsValid:    true,
				PlateType:  "INDIVIDUAL",
				ScoreBonus: 0.10,
			}
		}
	}

	if regexGeneralPlate.MatchString(cleaned) {
		return NormalizationResult{
			Original:   raw,
			Normalized: cleaned,
			IsValid:    true,
			PlateType:  "GENERAL",
			ScoreBonus: 0.05,
		}
	}

	return NormalizationResult{
		Original:   raw,
		Normalized: cleaned,
		IsValid:    false,
		PlateType:  "UNKNOWN",
		ScoreBonus: 0.0,
	}
}

func forceDigit(r rune) rune {
	switch r {
	case 'O', 'Q', 'D':
		return '0'
	case 'I', 'L':
		return '1'
	case 'Z':
		return '2'
	case 'S':
		return '5'
	case 'B':
		return '8'
	}
	return r
}

func forceLetter(r rune) rune {
	switch r {
	case '0':
		return 'O'
	case '1':
		return 'I'
	case '5':
		return 'S'
	case '8':
		return 'B'
	}
	return r
}
