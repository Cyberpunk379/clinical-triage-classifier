"""Symptom category extraction.

Hybrid pipeline:
  1. A curated clinical lexicon flags explicit symptom mentions across six
     high-level categories.
  2. The same NLI model used for triage scores each category's presence on
     the note as a whole, surfacing implicit findings the lexicon misses
     (e.g. "I get winded walking to the mailbox" → respiratory).
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from functools import lru_cache
from typing import Optional


# NegEx-lite: triggers and a clause-bounded window. Catches "no chest pain",
# "denies SOB", "without dyspnea", etc. without pulling in a full clinical
# NLP framework. Negation scope is bounded by sentence-ending punctuation.
NEGATION_TRIGGERS = [
    "no",
    "not",
    "without",
    "denies",
    "denying",
    "denied",
    "negative for",
    "absent",
    "ruled out",
    "no history of",
    "no evidence of",
    "free of",
]

NEGATION_WINDOW_WORDS = 6


SYMPTOM_CATEGORIES: dict[str, dict] = {
    "cardiovascular": {
        "lexicon": [
            "chest pain", "chest pressure", "chest tightness", "palpitations",
            "racing heart", "irregular heartbeat", "syncope", "fainting",
            "swelling in legs", "edema",
        ],
        "hypothesis": "The patient is describing heart or circulation symptoms such as chest pain, palpitations, or fainting.",
    },
    "respiratory": {
        "lexicon": [
            "shortness of breath", "dyspnea", "cough", "wheezing", "wheeze",
            "winded", "trouble breathing", "can't catch my breath", "sob",
            "coughing up blood", "hemoptysis",
        ],
        "hypothesis": "The patient is describing breathing-related symptoms such as shortness of breath, cough, or wheezing.",
    },
    "neurological": {
        "lexicon": [
            "headache", "migraine", "dizziness", "vertigo", "numbness",
            "tingling", "weakness", "slurred speech", "vision change",
            "double vision", "loss of consciousness", "seizure", "confusion",
        ],
        "hypothesis": "The patient is describing nervous-system symptoms such as headache, numbness, slurred speech, weakness, or vision changes.",
    },
    "gastrointestinal": {
        "lexicon": [
            "nausea", "vomiting", "diarrhea", "constipation", "abdominal pain",
            "stomach pain", "belly pain", "bloody stool", "melena",
            "heartburn", "reflux",
        ],
        "hypothesis": "The patient is describing digestive symptoms such as nausea, vomiting, diarrhea, or abdominal pain.",
    },
    "constitutional": {
        "lexicon": [
            "fever", "chills", "night sweats", "fatigue", "tired all the time",
            "weight loss", "loss of appetite", "malaise",
        ],
        "hypothesis": "The patient is describing constitutional symptoms such as fever, chills, fatigue, or weight loss.",
    },
    "musculoskeletal": {
        "lexicon": [
            "back pain", "joint pain", "muscle pain", "stiffness", "swollen joint",
            "limited range of motion", "sprain", "strain",
        ],
        "hypothesis": "The patient is describing pain, stiffness, or injury affecting joints, muscles, bones, or the back.",
    },
}


def _is_negated(text: str, match_start: int, window_words: int = NEGATION_WINDOW_WORDS) -> bool:
    """Return True if the match at `match_start` is in the scope of a negation trigger.

    Scope is bounded by the nearest preceding sentence-ending punctuation
    (`.`, `;`, `?`, `!`) and a `window_words` word distance between the
    trigger and the match. This catches:

        "No chest pain, shortness of breath, or recent illness."  → both negated
        "Patient denies SOB."                                     → negated
        "Without dyspnea on exertion."                            → negated

    But correctly does *not* negate:

        "No prior history. Patient has chest pain."  → NOT negated
    """
    clause_start = max(text.rfind(c, 0, match_start) for c in ".;?!\n")
    clause_start = clause_start + 1 if clause_start >= 0 else 0
    preceding = text[clause_start:match_start].lower()

    if not preceding.strip():
        return False

    for trigger in NEGATION_TRIGGERS:
        for trigger_match in re.finditer(rf"\b{re.escape(trigger)}\b", preceding):
            between = preceding[trigger_match.end():]
            word_count = len(between.split())
            if word_count <= window_words:
                return True
    return False


@dataclass
class SymptomMatch:
    """A single explicit symptom mention."""

    category: str
    term: str
    span: tuple[int, int]


@dataclass
class SymptomResult:
    """Output of the symptom extractor."""

    explicit: list[SymptomMatch] = field(default_factory=list)
    negated: list[SymptomMatch] = field(default_factory=list)
    implicit_scores: dict[str, float] = field(default_factory=dict)
    matched_categories: set[str] = field(default_factory=set)

    def all_categories(self, implicit_threshold: float = 0.55) -> set[str]:
        """Union of explicitly matched and implicitly-scored-high categories."""
        implicit = {
            cat for cat, score in self.implicit_scores.items()
            if score >= implicit_threshold
        }
        return self.matched_categories | implicit


class SymptomExtractor:
    """Extracts symptom categories from a clinical note."""

    def __init__(
        self,
        nli_model: str = "valhalla/distilbart-mnli-12-3",
        device: int | str | None = -1,
        use_nli: bool = True,
    ):
        self.nli_model = nli_model
        self.device = device
        self.use_nli = use_nli
        self._pipe = None
        self._compiled = {
            cat: [(term, re.compile(rf"\b{re.escape(term)}\b", re.IGNORECASE))
                  for term in config["lexicon"]]
            for cat, config in SYMPTOM_CATEGORIES.items()
        }

    def _load(self):
        if self._pipe is None and self.use_nli:
            from transformers import pipeline  # lazy import
            self._pipe = pipeline(
                "zero-shot-classification",
                model=self.nli_model,
                device=self.device,
            )
        return self._pipe

    def extract(self, text: str) -> SymptomResult:
        """Run hybrid extraction on a note.

        Lexicon matches inside a negation scope are tracked in `result.negated`
        and NOT counted as explicit positive findings. This matters: a note
        that says "no chest pain" should not be flagged as cardiovascular.
        """
        if not text or not text.strip():
            return SymptomResult()

        explicit: list[SymptomMatch] = []
        negated: list[SymptomMatch] = []
        matched: set[str] = set()

        for category, patterns in self._compiled.items():
            for term, pattern in patterns:
                for match in pattern.finditer(text):
                    sm = SymptomMatch(
                        category=category,
                        term=match.group(0),
                        span=(match.start(), match.end()),
                    )
                    if _is_negated(text, match.start()):
                        negated.append(sm)
                    else:
                        explicit.append(sm)
                        matched.add(category)

        implicit_scores: dict[str, float] = {}
        if self.use_nli:
            pipe = self._load()
            hypotheses = [config["hypothesis"] for config in SYMPTOM_CATEGORIES.values()]
            categories = list(SYMPTOM_CATEGORIES.keys())
            result = pipe(
                text,
                candidate_labels=hypotheses,
                hypothesis_template="{}",
                multi_label=True,
            )
            hyp_to_cat = {
                config["hypothesis"]: cat
                for cat, config in SYMPTOM_CATEGORIES.items()
            }
            for label, score in zip(result["labels"], result["scores"]):
                implicit_scores[hyp_to_cat[label]] = float(score)

        return SymptomResult(
            explicit=explicit,
            negated=negated,
            implicit_scores=implicit_scores,
            matched_categories=matched,
        )


@lru_cache(maxsize=3)
def get_extractor(nli_model: str = "valhalla/distilbart-mnli-12-3") -> SymptomExtractor:
    """Return a cached extractor instance for the given NLI model."""
    return SymptomExtractor(nli_model=nli_model)


def default_extractor() -> SymptomExtractor:
    """Backwards-compatible alias for the default-model extractor."""
    return get_extractor()
