"""Download the dataset and make train/val/test splits.

Source: NewsDataHub/cybersecurity-news-dataset-english-3000 (Hugging Face, CC BY 4.0).
The six mentions_* columns are used as labels.
"""
import argparse
import urllib.request
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

PARQUET_URL = (
    "https://huggingface.co/api/datasets/NewsDataHub/cybersecurity-news-dataset-english-3000"
    "/parquet/default/train/0.parquet"
)

# from rarest to most common (used for stratifying)
LABELS = ["ddos", "vulnerability", "ransomware", "hacking", "cyber_attack", "data_breach"]


def build_text(row: pd.Series) -> str:
    title = str(row["title"]).strip()
    desc = str(row["description"]).strip()
    if not desc or desc.lower() == title.lower():
        return title
    return f"{title}. {desc}"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default="data/en")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    raw = Path("data/raw.parquet")
    if not raw.exists():
        raw.parent.mkdir(parents=True, exist_ok=True)
        print("Downloading dataset...")
        urllib.request.urlretrieve(PARQUET_URL, raw)

    df = pd.read_parquet(raw)
    df["text"] = df.apply(build_text, axis=1)
    for label in LABELS:
        df[label] = df[f"mentions_{label}"].astype(str).eq("True").astype(int)

    df = df.drop_duplicates(subset="text")
    df = df[df[LABELS].sum(axis=1) > 0]  # remove articles with no label

    # stratify by the rarest label so every split has some ddos / vulnerability
    df["primary"] = df[LABELS].idxmax(axis=1)

    train, rest = train_test_split(df, test_size=0.30, stratify=df["primary"], random_state=args.seed)
    val, test = train_test_split(rest, test_size=0.50, stratify=rest["primary"], random_state=args.seed)

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    cols = ["id", "text", *LABELS]
    for name, split in [("train", train), ("val", val), ("test", test)]:
        split[cols].to_csv(out / f"{name}.csv", index=False)
        print(f"{name}: {len(split)} rows | " + ", ".join(f"{l}={split[l].sum()}" for l in LABELS))


if __name__ == "__main__":
    main()
