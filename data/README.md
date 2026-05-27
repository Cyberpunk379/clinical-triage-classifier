# Data

This folder is where downloaded corpora live. Nothing is checked in.

## MTSamples

MTSamples is a publicly available collection of ~5,000 de-identified medical
transcription samples across multiple specialties. It is not labeled for
triage urgency, so it cannot be used directly for supervised training of
this classifier — but it is excellent for stress-testing the zero-shot
baseline, building a held-out evaluation set by hand-labeling a subset, and
exploring symptom-extraction edge cases.

### Fetch

The cleanest hosted copy is on Kaggle:

> https://www.kaggle.com/datasets/tboyle10/medicaltranscriptions

Download the CSV (`mtsamples.csv`) into this folder:

```
data/mtsamples.csv
```

### Use

```python
from src.data import load_mtsamples, sample_mtsamples

df = load_mtsamples("data/mtsamples.csv")
print(df.shape)                                   # (~4999, 5)
print(df["medical_specialty"].value_counts())     # specialty distribution

# Draw a small sample to label by hand for evaluation
sample = sample_mtsamples(df, n=30, specialty="Emergency Room Reports")
```

### Building a labeled evaluation set

Recommended workflow:

1. Sample 30-50 notes across specialties most relevant to triage (Emergency
   Room Reports, Consult — History and Phy., General Medicine).
2. Hand-label each with one of `ROUTINE` / `SOON` / `URGENT`.
3. Save as `data/labeled_eval.csv` with columns `note_id, text, tier`.
4. Update the results table in the top-level README with measured numbers
   on that set.

## Other corpora worth knowing

- **n2c2 / i2b2 challenges** — gold-standard NLP corpora for clinical tasks.
  Access via DBMI Portal: https://portal.dbmi.hms.harvard.edu/projects/n2c2-nlp/
  Requires registration and DUA but is free.
- **MIMIC-III / MIMIC-IV** — ICU clinical notes. Requires CITI training and
  credentialing through PhysioNet. Higher friction but much larger.
- **MTSamples** (this folder) — the right place to start for a portfolio
  project, no credentialing required.
