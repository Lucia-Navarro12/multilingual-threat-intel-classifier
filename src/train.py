"""Fine-tune multilingual BERT (multi-label).

Examples:
  python src/train.py --train-langs en es   # English + Spanish
  python src/train.py --train-langs en      # only English
"""
import argparse
import json
from pathlib import Path

import numpy as np
import torch
from datasets import Dataset
from transformers import (AutoModelForSequenceClassification, AutoTokenizer,
                          DataCollatorWithPadding, Trainer, TrainingArguments)

from common import load, metrics
from data import LABELS


def to_dataset(df, tok, max_len: int) -> Dataset:
    ds = Dataset.from_dict({
        "text": df["text"].tolist(),
        "labels": df[LABELS].values.astype(np.float32).tolist(),
    })
    return ds.map(lambda b: tok(b["text"], truncation=True, max_length=max_len),
                  batched=True, remove_columns=["text"])


def predict(trainer: Trainer, ds: Dataset, threshold: float) -> np.ndarray:
    logits = trainer.predict(ds).predictions
    return (torch.sigmoid(torch.tensor(logits)).numpy() >= threshold).astype(int)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="bert-base-multilingual-cased")
    parser.add_argument("--train-langs", nargs="+", default=["en", "es"])
    parser.add_argument("--test-langs", nargs="+", default=["en", "es"])
    parser.add_argument("--epochs", type=int, default=4)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--max-len", type=int, default=160)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    tag = f"{args.model.split('/')[-1]}_train-{'-'.join(args.train_langs)}"
    tok = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForSequenceClassification.from_pretrained(
        args.model,
        num_labels=len(LABELS),
        problem_type="multi_label_classification",
        id2label=dict(enumerate(LABELS)),
        label2id={l: i for i, l in enumerate(LABELS)},
    )

    train_ds = to_dataset(load("train", args.train_langs), tok, args.max_len)
    val_ds = to_dataset(load("val", args.train_langs), tok, args.max_len)

    def compute_metrics(pred):
        probs = torch.sigmoid(torch.tensor(pred.predictions)).numpy()
        m = metrics(pred.label_ids.astype(int), (probs >= args.threshold).astype(int))
        return {"f1_micro": m["f1_micro"], "f1_macro": m["f1_macro"]}

    training_args = TrainingArguments(
        output_dir=f"models/{tag}",
        num_train_epochs=args.epochs,
        learning_rate=args.lr,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=64,
        weight_decay=0.01,
        warmup_ratio=0.1,
        eval_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=1,
        load_best_model_at_end=True,
        metric_for_best_model="f1_macro",
        fp16=torch.cuda.is_available(),
        report_to="none",
        seed=args.seed,
    )
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        data_collator=DataCollatorWithPadding(tok),
        compute_metrics=compute_metrics,
    )
    trainer.train()

    results = {}
    for lang in args.test_langs:
        test_df = load("test", [lang])
        y_pred = predict(trainer, to_dataset(test_df, tok, args.max_len), args.threshold)
        results[lang] = metrics(test_df[LABELS].values, y_pred)
        print(f"[{lang}] F1 micro={results[lang]['f1_micro']:.3f} macro={results[lang]['f1_macro']:.3f}")

    Path("results").mkdir(exist_ok=True)
    Path(f"results/{tag}.json").write_text(json.dumps(results, indent=2))
    trainer.save_model(f"models/{tag}/best")
    tok.save_pretrained(f"models/{tag}/best")
    print(f"Saved results/{tag}.json and models/{tag}/best")


if __name__ == "__main__":
    main()
