"""Clinical NLP Triage Classifier.

Submodules are imported lazily so that lightweight uses (lexicon-only symptom
extraction, data utilities, tests) don't pull in `transformers` and `torch`
unless actually needed.
"""

__version__ = "0.1.0"
__all__ = ["TriageClassifier", "TriageResult", "SymptomExtractor", "SymptomResult"]
