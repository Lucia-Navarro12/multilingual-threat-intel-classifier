"""Shared helpers: loading splits and computing multi-label metrics."""
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import classification_report, f1_score

from data import LABELS


def load(split: str, langs: list[str], root: str = "data") -> pd.DataFrame:
    frames = []
    for lang in langs:
        df = pd.read_csv(Path(root) / lang / f"{split}.csv")
        df["lang"] = lang
        frames.append(df)
    return pd.concat(frames, ignore_index=True)


def metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    report = classification_report(y_true, y_pred, target_names=LABELS, zero_division=0, output_dict=True)
    return {
        "f1_micro": f1_score(y_true, y_pred, average="micro", zero_division=0),
        "f1_macro": f1_score(y_true, y_pred, average="macro", zero_division=0),
        "per_label_f1": {l: report[l]["f1-score"] for l in LABELS},
    }
