"""Classify new headlines with a fine-tuned model.

  python src/predict.py --model-dir models/bert-base-multilingual-cased_train-en-es/best \
      "Ransomware attack paralyses hospital systems" "Filtrados los datos de 2 millones de clientes"
"""
import argparse

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-dir", required=True)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("texts", nargs="+")
    args = parser.parse_args()

    tok = AutoTokenizer.from_pretrained(args.model_dir)
    model = AutoModelForSequenceClassification.from_pretrained(args.model_dir).eval()
    batch = tok(args.texts, return_tensors="pt", padding=True, truncation=True, max_length=160)
    with torch.no_grad():
        probs = torch.sigmoid(model(**batch).logits)

    for text, p in zip(args.texts, probs):
        labels = [f"{model.config.id2label[i]} ({v:.2f})" for i, v in enumerate(p.tolist()) if v >= args.threshold]
        print(f"{text}\n  -> {', '.join(labels) or 'no threat type above threshold'}")


if __name__ == "__main__":
    main()
