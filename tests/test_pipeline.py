"""Sanity tests for the triage pipeline.

These don't load the transformer model by default — they validate the
lexicon extractor, data structures, and edge cases. To run the full
end-to-end test that exercises the NLI model, set RUN_E2E=1.

    pytest tests/                  # fast tests only
    RUN_E2E=1 pytest tests/        # includes model-loading test
"""

from __future__ import annotations

import os

import pytest

from src.classifier import TriageResult, TIER_HYPOTHESES, _build_rationale
from src.data import EXAMPLE_NOTES
from src.extractor import SYMPTOM_CATEGORIES, SymptomExtractor, SymptomResult


# ---------------------------------------------------------------------------
# Data integrity
# ---------------------------------------------------------------------------

def test_tier_hypotheses_have_all_three_tiers():
    assert set(TIER_HYPOTHESES.keys()) == {"ROUTINE", "SOON", "URGENT"}


def test_symptom_categories_have_lexicons():
    for cat, config in SYMPTOM_CATEGORIES.items():
        assert config["lexicon"], f"{cat} has empty lexicon"
        assert config["hypothesis"], f"{cat} has empty hypothesis"


def test_example_notes_cover_all_tiers():
    tiers = {note.expected_tier for note in EXAMPLE_NOTES}
    assert tiers == {"ROUTINE", "SOON", "URGENT"}


# ---------------------------------------------------------------------------
# Lexicon-only extraction (no model load)
# ---------------------------------------------------------------------------

@pytest.fixture
def lex_only_extractor():
    """Extractor with NLI disabled — runs without downloading models."""
    return SymptomExtractor(use_nli=False)


def test_extractor_finds_explicit_chest_pain(lex_only_extractor):
    text = "Patient reports sudden onset chest pain and shortness of breath."
    result = lex_only_extractor.extract(text)
    cats = {m.category for m in result.explicit}
    assert "cardiovascular" in cats
    assert "respiratory" in cats


def test_extractor_handles_empty_input(lex_only_extractor):
    result = lex_only_extractor.extract("")
    assert result.explicit == []
    assert result.matched_categories == set()


def test_extractor_is_case_insensitive(lex_only_extractor):
    result = lex_only_extractor.extract("HEADACHE and DIZZINESS started yesterday.")
    cats = {m.category for m in result.explicit}
    assert "neurological" in cats


def test_extractor_finds_matched_categories_set(lex_only_extractor):
    text = "Reports nausea and vomiting along with fever."
    result = lex_only_extractor.extract(text)
    assert "gastrointestinal" in result.matched_categories
    assert "constitutional" in result.matched_categories


# ---------------------------------------------------------------------------
# Negation handling (NegEx-lite)
# ---------------------------------------------------------------------------

def test_negation_simple_no(lex_only_extractor):
    """A negated symptom should not appear in explicit matches."""
    text = "Patient reports no chest pain."
    result = lex_only_extractor.extract(text)
    assert "cardiovascular" not in result.matched_categories
    assert len(result.negated) >= 1
    assert any(m.category == "cardiovascular" for m in result.negated)


def test_negation_multiple_symptoms_same_clause(lex_only_extractor):
    """A single 'No X, Y, or Z' clause should negate all listed symptoms."""
    text = (
        "Patient is feeling generally well. No chest pain, shortness of breath, "
        "or recent illness. Sleep and appetite normal."
    )
    result = lex_only_extractor.extract(text)
    # All three should be NEGATED, not explicit.
    assert "cardiovascular" not in result.matched_categories
    assert "respiratory" not in result.matched_categories
    negated_cats = {m.category for m in result.negated}
    assert "cardiovascular" in negated_cats
    assert "respiratory" in negated_cats


def test_negation_denies(lex_only_extractor):
    """The 'denies' trigger should be recognized."""
    text = "Patient denies shortness of breath."
    result = lex_only_extractor.extract(text)
    assert "respiratory" not in result.matched_categories


def test_negation_does_not_cross_sentence_boundary(lex_only_extractor):
    """Negation in one sentence should not affect symptoms in the next."""
    text = "No prior history of cardiac issues. Patient has chest pain now."
    result = lex_only_extractor.extract(text)
    assert "cardiovascular" in result.matched_categories


def test_positive_symptom_still_caught_after_negation_fix(lex_only_extractor):
    """Make sure we didn't break positive detection."""
    text = "Patient reports crushing chest pain radiating to left arm."
    result = lex_only_extractor.extract(text)
    assert "cardiovascular" in result.matched_categories
    assert len(result.explicit) >= 1


# ---------------------------------------------------------------------------
# Rationale rendering
# ---------------------------------------------------------------------------

def test_rationale_for_strong_signal():
    dist = {"URGENT": 0.85, "SOON": 0.10, "ROUTINE": 0.05}
    rationale = _build_rationale(dist)
    assert "Strong signal" in rationale
    assert "URGENT" in rationale


def test_rationale_for_borderline():
    dist = {"SOON": 0.45, "ROUTINE": 0.40, "URGENT": 0.15}
    rationale = _build_rationale(dist)
    assert "borderline" in rationale.lower()


def test_triage_result_is_confident():
    r = TriageResult(tier="URGENT", confidence=0.82, distribution={})
    assert r.is_confident()
    r2 = TriageResult(tier="SOON", confidence=0.42, distribution={})
    assert not r2.is_confident()


# ---------------------------------------------------------------------------
# End-to-end (gated; loads the transformer)
# ---------------------------------------------------------------------------

@pytest.mark.skipif(
    os.environ.get("RUN_E2E") != "1",
    reason="Set RUN_E2E=1 to run end-to-end tests that download the NLI model.",
)
def test_e2e_urgent_note_classifies_as_urgent():
    from src.classifier import TriageClassifier

    clf = TriageClassifier()
    urgent_note = next(n for n in EXAMPLE_NOTES if n.expected_tier == "URGENT")
    result = clf.classify(urgent_note.text)
    assert result.tier == "URGENT"
    assert result.confidence > 0.5
