"""Machine-translate the English splits into Spanish with MarianMT (Helsinki-NLP/opus-mt-en-es).

Each split is translated on its own, so a translated article never leaks into another split.
Run on a GPU (e.g. Google Colab): ~2,900 short texts take a few minutes.
"""
import argparse
from pathlib import Path

import pandas as pd
import torch
from transformers import MarianMTModel, MarianTokenizer

MODEL = "Helsinki-NLP/opus-mt-en-es"


def translate(texts: list[str], tok, model, device, batch_size: int) -> list[str]:
    out = []
    for i in range(0, len(texts), batch_size):
        batch = tok(texts[i : i + batch_size], return_tensors="pt", padding=True,
                    truncation=True, max_length=512).to(device)
        with torch.no_grad():
            gen = model.generate(**batch, max_new_tokens=512, num_beams=4)
        out.extend(tok.batch_decode(gen, skip_special_tokens=True))
        print(f"  {min(i + batch_size, len(texts))}/{len(texts)}", end="\r")
    print()
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--src", default="data/en")
    parser.add_argument("--out", default="data/es")
    parser.add_argument("--batch-size", type=int, default=32)
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    tok = MarianTokenizer.from_pretrained(MODEL)
    model = MarianMTModel.from_pretrained(MODEL).to(device).eval()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    for split in ["train", "val", "test"]:
        df = pd.read_csv(Path(args.src) / f"{split}.csv")
        print(f"Translating {split} ({len(df)} rows) on {device}...")
        df["text"] = translate(df["text"].tolist(), tok, model, device, args.batch_size)
        df.to_csv(out / f"{split}.csv", index=False)


if __name__ == "__main__":
    main()
