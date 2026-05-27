# Clinical NLP Triage Classifier

> A demonstrator for zero-shot triage of free-text clinical notes and patient-reported symptom narratives, producing structured symptom extractions with NegEx negation handling and urgency tiers with calibrated confidence.

<p align="center">
  <a href="#demo">Demo</a> ·
  <a href="#approach">Approach</a> ·
  <a href="#evaluation">Evaluation</a> ·
  <a href="#limitations">Limitations</a> ·
  <a href="#roadmap">Roadmap</a>
</p>

---

## Why this exists

Most patient-clinician contact is mediated by free-text: chief complaints, symptom narratives, asynchronous messages. Triaging that text — *how urgent is this, and what symptoms are actually being described* — is one of the highest-leverage applications of clinical NLP, and one of the easiest to get wrong.

This project is a transparent, runnable baseline for that problem. It shows:

1. A zero-shot transformer pipeline that works out of the box on unseen notes.
2. A hybrid lexicon + NLI symptom extractor that correctly handles negation (e.g., does *not* flag "no chest pain" as cardiovascular).
3. A Streamlit demo recruiters, clinicians, and engineers can run locally in under five minutes — with live model swapping between a fast distilled model and a more accurate larger one.

It is not a clinical product. See [Limitations](#limitations).

## Demo

```bash
git clone https://github.com/Cyberpunk379/clinical-triage-classifier
cd clinical-triage-classifier
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
streamlit run app.py
```

Paste a clinical note (or pick one of the six bundled examples spanning all three tiers) and read off the urgency tier, extracted symptom categories, and per-class confidence. The sidebar selector lets you swap between models live.

Hosted version: https://huggingface.co/spaces/Osafobz/clinical-triage-classifier

## Approach

**Triage classification.** Three-tier triage (`ROUTINE` / `SOON` / `URGENT`) is framed as a zero-shot natural language inference task. Each note is scored against natural-language hypotheses for each tier (e.g., *"The patient has acute, sudden-onset, or rapidly developing severe symptoms suggesting a medical emergency such as stroke, heart attack, or severe acute distress."*) and entailment probabilities are softmaxed to yield a calibrated distribution.

This avoids the cold-start problem of supervised triage classifiers — no labeled training corpus is required to ship a useful baseline. It also gives an honest confidence signal: a note that genuinely sits between tiers produces a flat distribution rather than a confidently-wrong hard label, and the system surfaces this as a "consider clinician review" rationale.

**Model choice.** The default is `valhalla/distilbart-mnli-12-3` (~300MB), a distilled BART-MNLI that runs comfortably on a laptop. The UI sidebar lets you live-swap to `facebook/bart-large-mnli` (~1.6GB) for higher accuracy on borderline cases. Either model is loaded lazily on first call and cached per-process.

**Symptom extraction.** A hybrid pipeline combining (a) a curated clinical lexicon mapped to six high-level symptom categories — cardiovascular, respiratory, neurological, gastrointestinal, constitutional, musculoskeletal — and (b) span-level confidence from the same NLI model. Categories with no lexical hits but strong NLI entailment are surfaced as *implicit* findings, which matters for narratives that describe symptoms without naming them (`"I get winded walking to the mailbox"` → respiratory).

**Negation handling (NegEx-lite).** A common failure mode of naive symptom extractors is matching keywords inside negation scope — flagging "no chest pain" as a positive cardiovascular finding. The extractor uses a NegEx-style approach: it identifies negation triggers (`no`, `not`, `denies`, `without`, `negative for`, …) and suppresses lexicon matches within the same clause and a bounded word window. Negated matches are tracked separately on `SymptomResult.negated` rather than discarded, which preserves the audit trail.

**Why not just fine-tune Clinical BERT?** Because (a) MTSamples-style public corpora are unlabeled for triage urgency, (b) MIMIC requires credentialing that gates reproducibility, and (c) a zero-shot baseline you can read and audit beats a black-box classifier of unknown provenance for a portfolio piece. A fine-tuning path is included under `src/finetune.py` for the supervised follow-up.

## Repo layout

```
clinical-triage-classifier/
├── app.py                     # Streamlit demo with live model selector
├── src/
│   ├── classifier.py          # Zero-shot triage, model-aware cached singleton
│   ├── extractor.py           # Symptom extraction + NegEx-lite negation
│   ├── data.py                # Bundled example notes + MTSamples loader
│   └── finetune.py            # Supervised fine-tuning skeleton (roadmap)
├── scripts/
│   └── eval.py                # Reproducible eval against bundled examples
├── tests/
│   └── test_pipeline.py       # 15 tests including negation handling
├── data/
│   └── README.md              # How to fetch MTSamples
├── requirements.txt
└── README.md
```

## Evaluation

Against the six hand-labeled example notes bundled in `src/data.py`, the default `valhalla/distilbart-mnli-12-3` zero-shot baseline achieves:

```
Accuracy: 6/6 = 100%

Confusion (rows = expected, cols = predicted):
           ROUTINE      SOON    URGENT
  ROUTINE        2         0         0
  SOON           0         2         0
  URGENT         0         0         2
```

Confidence per note ranges from 52% (moderate, on a borderline new-symptom case) to 81% (strong). The system honestly surfaces the moderate calls in its rationale string rather than presenting them as confident hard labels.

Reproduce this:

```bash
python scripts/eval.py                                  # default distilbart
python scripts/eval.py --model facebook/bart-large-mnli # larger model
```

Six notes is intentionally a small floor — the reproducible baseline anyone who clones the repo can verify, not a benchmark. A larger labeled evaluation on a MTSamples subset is the next deliverable; see `data/README.md` for the labeling workflow.

The `tests/test_pipeline.py` suite (15 tests, runs in under one second without model downloads) covers the lexicon extractor, NegEx negation handling — including the specific *"No chest pain, shortness of breath, or recent illness"* multi-symptom case — and the rationale-rendering logic. Run with `pytest tests/`.

## Limitations

- **Not a medical device.** This is a research and engineering demonstrator. It is not validated for clinical use, has not been reviewed by clinicians, and must not be used to make care decisions.
- **English-only.** The underlying NLI models are trained on English text.
- **No de-identification.** Inputs to the demo are processed by the loaded model. Do not paste real patient data into the hosted version. Run locally for any text containing PHI.
- **No calibration on real triage outcomes.** Confidence scores reflect model entailment probabilities, not actual triage accuracy in deployment.
- **Bias.** The base NLI models carry the biases of their training corpora. Triage performance has not been audited across demographic strata.
- **Edge cases.** Borderline notes between tiers will sometimes produce confidently-wrong labels with the distilled model. Switching to BART-large via the UI selector materially improves these cases at the cost of memory.

## Roadmap

- [ ] Hand-label a 50-note MTSamples subset and report measured precision/recall/F1 per tier.
- [ ] Fine-tune Clinical BERT on the labeled corpus and benchmark against the zero-shot baselines.
- [ ] Add temporal reasoning: extract symptom onset, duration, trajectory.
- [ ] Add structured output schema (FHIR Observation resources).
- [ ] Deploy a HuggingFace Spaces version with rate-limited inference.

## Related work

Related published work by the author on the surrounding infrastructure for systems like this:

- *A Patient-Owned Health Data Layer for AI-Enabled Healthcare Systems.* SSRN, 2026.
- *Patient-Centered Health Data Architectures for Interoperable Care in Emerging and Underserved Markets.* SSRN, 2026.
- *Governing Patient-Centered Health Data Systems: Trust, Consent, and AI Accountability in Emerging Digital Health Ecosystems.* SSRN, 2026.

## License

MIT. See `LICENSE`.

## Citation

If you reference this work:

```bibtex
@software{afobhokhan2026triage,
  author  = {Afobhokhan, Humphrey},
  title   = {Clinical NLP Triage Classifier},
  year    = {2026},
  url     = {https://github.com/<your-handle>/clinical-triage-classifier}
}
```