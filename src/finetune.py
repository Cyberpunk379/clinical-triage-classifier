"""Optional supervised fine-tuning skeleton.

This module is a starting point for the supervised follow-up to the zero-shot
baseline. Fill in once you have a hand-labeled corpus (n >= 500 recommended).

Suggested base model: emilyalsentzer/Bio_ClinicalBERT
Suggested split: 70/15/15 train/val/test, stratified by tier.

Run with:
    python -m src.finetune \\
        --train-csv data/labeled_train.csv \\
        --val-csv data/labeled_val.csv \\
        --output-dir models/triage-clinicalbert
"""

from __future__ import annotations

import argparse
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description="Fine-tune a triage classifier.")
    parser.add_argument("--train-csv", required=True, type=Path)
    parser.add_argument("--val-csv", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--base-model", default="emilyalsentzer/Bio_ClinicalBERT")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=2e-5)
    args = parser.parse_args()

    # TODO: Implement once a labeled corpus exists.
    #   1. Load CSVs with columns: text, tier (ROUTINE|SOON|URGENT).
    #   2. Tokenize with AutoTokenizer.from_pretrained(args.base_model).
    #   3. Build AutoModelForSequenceClassification with num_labels=3.
    #   4. Train with transformers.Trainer.
    #   5. Save model + tokenizer to args.output_dir.
    #   6. Report macro-F1 on val set.

    raise NotImplementedError(
        "Supervised fine-tuning skeleton. Implement when labeled data is available."
    )


if __name__ == "__main__":
    main()
