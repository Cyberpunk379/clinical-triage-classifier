"""Example notes and MTSamples loading utilities."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import pandas as pd


@dataclass
class ExampleNote:
    """A bundled example note covering a known triage tier."""

    label: str
    expected_tier: str
    text: str


EXAMPLE_NOTES: list[ExampleNote] = [
    ExampleNote(
        label="Routine — annual follow-up",
        expected_tier="ROUTINE",
        text=(
            "Patient is a 42-year-old presenting for routine annual physical. "
            "Reports feeling generally well. Mild seasonal allergies controlled "
            "with over-the-counter antihistamines. No chest pain, shortness of "
            "breath, or recent illness. Sleep and appetite normal. Last labs "
            "six months ago were within normal limits."
        ),
    ),
    ExampleNote(
        label="Routine — medication refill",
        expected_tier="ROUTINE",
        text=(
            "Patient called requesting refill of lisinopril. Reports blood "
            "pressure readings at home averaging 124/78 over the last two weeks. "
            "No side effects, no dizziness, no cough. Asking if appointment is "
            "needed before next refill in three months."
        ),
    ),
    ExampleNote(
        label="Soon — worsening symptoms over days",
        expected_tier="SOON",
        text=(
            "Patient reports four days of progressively worsening cough with "
            "yellow-green sputum production. Mild fever (100.8°F) yesterday "
            "evening. Some fatigue and decreased appetite. No shortness of "
            "breath at rest, but feels winded climbing one flight of stairs, "
            "which is new. No chest pain. Wants to know if she should be seen."
        ),
    ),
    ExampleNote(
        label="Soon — new neurological symptom",
        expected_tier="SOON",
        text=(
            "55-year-old male reports two days of intermittent numbness and "
            "tingling in left hand, lasting 10-15 minutes at a time. No "
            "weakness, no slurred speech, no vision changes. No history of "
            "similar episodes. Otherwise feels well. Taking aspirin daily."
        ),
    ),
    ExampleNote(
        label="Urgent — acute chest pain",
        expected_tier="URGENT",
        text=(
            "62-year-old male presents with sudden onset crushing chest "
            "pressure radiating to left arm and jaw, started 45 minutes ago "
            "while at rest. Associated with shortness of breath, nausea, and "
            "diaphoresis. History of hypertension and hyperlipidemia. Family "
            "history of MI in father at age 58. Currently still symptomatic."
        ),
    ),
    ExampleNote(
        label="Urgent — neurological emergency",
        expected_tier="URGENT",
        text=(
            "Family member reports patient developed sudden right-sided facial "
            "droop and slurred speech approximately 30 minutes ago. Patient is "
            "unable to lift right arm against gravity. No prior similar "
            "episodes. Last known well 45 minutes ago. Patient is 71 years old "
            "with history of atrial fibrillation."
        ),
    ),
]


def load_mtsamples(path: str | Path) -> pd.DataFrame:
    """Load the MTSamples CSV if present.

    MTSamples is a publicly available collection of de-identified medical
    transcription samples. See data/README.md for fetch instructions.

    Parameters
    ----------
    path : str | Path
        Path to the MTSamples CSV.

    Returns
    -------
    pd.DataFrame
        With at minimum columns: 'description', 'medical_specialty',
        'sample_name', 'transcription', 'keywords'.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"MTSamples CSV not found at {path}. "
            f"See data/README.md for download instructions."
        )
    df = pd.read_csv(path)
    df = df.dropna(subset=["transcription"]).reset_index(drop=True)
    return df


def sample_mtsamples(
    df: pd.DataFrame,
    n: int = 10,
    specialty: Optional[str] = None,
    seed: int = 42,
) -> pd.DataFrame:
    """Draw a small sample of notes, optionally filtered by specialty."""
    if specialty is not None:
        df = df[df["medical_specialty"].str.strip().str.lower() == specialty.lower()]
    return df.sample(min(n, len(df)), random_state=seed).reset_index(drop=True)
