"""Zero-shot triage classifier.

Frames three-tier triage (ROUTINE / SOON / URGENT) as a zero-shot NLI task
using facebook/bart-large-mnli. Each note is scored against natural-language
hypotheses for each tier; probabilities are softmaxed to yield a calibrated
distribution.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from functools import lru_cache
from typing import Optional


TIER_HYPOTHESES = {
    "ROUTINE": "The patient has no acute or new symptoms, only routine maintenance concerns such as an annual checkup, medication refill, or stable chronic condition.",
    "SOON": "The patient has new or worsening symptoms that need clinical evaluation within a few days but are not immediately life-threatening.",
    "URGENT": "The patient has acute, sudden-onset, or rapidly developing severe symptoms suggesting a medical emergency such as stroke, heart attack, or severe acute distress.",
}

DEFAULT_MODEL = "valhalla/distilbart-mnli-12-3"


@dataclass
class TriageResult:
    """Output of the triage classifier."""

    tier: str
    confidence: float
    distribution: dict[str, float] = field(default_factory=dict)
    rationale: Optional[str] = None

    def is_confident(self, threshold: float = 0.6) -> bool:
        """Whether the top-class probability clears a usable threshold."""
        return self.confidence >= threshold


class TriageClassifier:
    """Zero-shot triage classifier.

    Lazy-loads the underlying NLI pipeline on first call. Safe to instantiate
    cheaply at module-import time.

    Parameters
    ----------
    model_name : str
        HuggingFace model identifier. Defaults to facebook/bart-large-mnli.
    device : int | str | None
        Passed through to transformers.pipeline. -1 forces CPU.
    """

    def __init__(self, model_name: str = DEFAULT_MODEL, device: int | str | None = -1):
        self.model_name = model_name
        self.device = device
        self._pipe = None

    def _load(self):
        if self._pipe is None:
            from transformers import pipeline  # lazy import
            self._pipe = pipeline(
                "zero-shot-classification",
                model=self.model_name,
                device=self.device,
            )
        return self._pipe

    def classify(self, text: str) -> TriageResult:
        """Classify a clinical note into a triage tier.

        Parameters
        ----------
        text : str
            Free-text clinical note or patient-reported symptom narrative.

        Returns
        -------
        TriageResult
            Predicted tier, confidence, full distribution, and rationale.
        """
        if not text or not text.strip():
            return TriageResult(
                tier="ROUTINE",
                confidence=0.0,
                distribution={k: 0.0 for k in TIER_HYPOTHESES},
                rationale="Empty input.",
            )

        pipe = self._load()
        labels = list(TIER_HYPOTHESES.keys())
        hypotheses = [TIER_HYPOTHESES[label] for label in labels]

        # We use the hypotheses themselves as candidate labels so the entailment
        # scores reflect the full natural-language framing, then map back.
        result = pipe(
            text,
            candidate_labels=hypotheses,
            hypothesis_template="{}",
            multi_label=False,
        )

        # result["labels"] and result["scores"] are sorted by score desc.
        hyp_to_tier = {v: k for k, v in TIER_HYPOTHESES.items()}
        distribution = {
            hyp_to_tier[label]: float(score)
            for label, score in zip(result["labels"], result["scores"])
        }

        top_tier = max(distribution, key=distribution.get)
        top_score = distribution[top_tier]

        rationale = _build_rationale(distribution)

        return TriageResult(
            tier=top_tier,
            confidence=top_score,
            distribution=distribution,
            rationale=rationale,
        )


def _build_rationale(distribution: dict[str, float]) -> str:
    """Render a short natural-language rationale for the chosen tier."""
    sorted_tiers = sorted(distribution.items(), key=lambda kv: kv[1], reverse=True)
    top, runner_up = sorted_tiers[0], sorted_tiers[1]
    gap = top[1] - runner_up[1]

    if gap < 0.10:
        return (
            f"Confidence is borderline between {top[0]} ({top[1]:.0%}) and "
            f"{runner_up[0]} ({runner_up[1]:.0%}). Consider clinician review."
        )
    if top[1] >= 0.75:
        return f"Strong signal for {top[0]} ({top[1]:.0%})."
    return f"Moderate signal for {top[0]} ({top[1]:.0%})."


@lru_cache(maxsize=3)
def get_classifier(model_name: str = DEFAULT_MODEL) -> TriageClassifier:
    """Return a cached classifier instance for the given model.

    Caching by model_name lets the Streamlit app swap between models
    (e.g., distilbart ↔ bart-large) without reloading weights every call.
    """
    return TriageClassifier(model_name=model_name)


def default_classifier() -> TriageClassifier:
    """Backwards-compatible alias for the default-model classifier."""
    return get_classifier(DEFAULT_MODEL)
