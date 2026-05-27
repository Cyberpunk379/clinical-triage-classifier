"""Evaluate the zero-shot triage classifier against bundled example notes.

The six notes in `src/data.py:EXAMPLE_NOTES` are hand-labeled with their
expected triage tier. This script runs the classifier against each note
and reports per-note and aggregate accuracy.

This is a small eval — six notes is not a benchmark. It is the
reproducible quality floor: a baseline anyone who clones the repo can
verify, and a starting point for the larger hand-labeled MTSamples
evaluation described in `data/README.md`.

Usage:
    python scripts/eval.py
    python scripts/eval.py --model facebook/bart-large-mnli
"""

from __future__ import annotations

import argparse
import sys
from collections import Counter
from pathlib import Path

# Make src importable when running this script directly.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.classifier import DEFAULT_MODEL, TriageClassifier
from src.data import EXAMPLE_NOTES


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model",
        default=DEFAULT_MODEL,
        help=f"HuggingFace model identifier (default: {DEFAULT_MODEL})",
    )
    args = parser.parse_args()

    print(f"\n{'=' * 72}")
    print(f"Triage eval — model: {args.model}")
    print(f"Notes: {len(EXAMPLE_NOTES)}")
    print(f"{'=' * 72}\n")

    classifier = TriageClassifier(model_name=args.model)

    correct = 0
    by_tier: dict[str, Counter] = {"ROUTINE": Counter(), "SOON": Counter(), "URGENT": Counter()}

    for note in EXAMPLE_NOTES:
        result = classifier.classify(note.text)
        is_correct = result.tier == note.expected_tier
        if is_correct:
            correct += 1
        by_tier[note.expected_tier][result.tier] += 1

        marker = "✓" if is_correct else "✗"
        print(f"{marker} {note.label}")
        print(f"  expected:  {note.expected_tier}")
        print(f"  predicted: {result.tier} ({result.confidence:.0%})")
        print(f"  rationale: {result.rationale}")
        print()

    accuracy = correct / len(EXAMPLE_NOTES)
    print(f"{'=' * 72}")
    print(f"Accuracy: {correct}/{len(EXAMPLE_NOTES)} = {accuracy:.0%}")
    print(f"{'=' * 72}\n")

    print("Confusion (rows = expected, cols = predicted):")
    tiers = ["ROUTINE", "SOON", "URGENT"]
    header = "          " + "  ".join(f"{t:>8}" for t in tiers)
    print(header)
    for expected in tiers:
        row = "  ".join(f"{by_tier[expected].get(t, 0):>8}" for t in tiers)
        print(f"  {expected:<8}{row}")
    print()

    return 0 if accuracy == 1.0 else 1


if __name__ == "__main__":
    sys.exit(main())
