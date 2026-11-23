"""Translate the English splits to Spanish with opus-mt-en-es.

Each split is translated separately so nothing leaks between splits.
Texts are translated sentence by sentence: with several sentences at once
the model sometimes drops the first one (usually the title).
Better on GPU.
"""
import argparse
import re
from pathlib import Path

import pandas as pd
import torch
from transformers import MarianMTModel, MarianTokenizer

MODEL = "Helsinki-NLP/opus-mt-en-es"


ABBREV = re.compile(r"(\b(?:[A-Z]\.){1,3}|\b(?:Mr|Mrs|Ms|Dr|Inc|Corp|Ltd|Co|No|vs|mil|St)\.)$")


def split_sentences(text: str) -> list[str]:
    # split after . ! ? when the next sentence starts with a capital letter or a quote
    parts = re.split(r"(?<=[.!?])\s+(?=[A-Z\"“'])", text.strip())
    sents = []
    for p in parts:
        if sents and ABBREV.search(sents[-1]):  # "U.S. man..." is not a new sentence
            sents[-1] += " " + p
        else:
            sents.append(p)
    return [s for s in sents if s]


def translate_sentences(sents: list[str], tok, model, device, batch_size: int) -> list[str]:
    # sort by length so each batch has similar lengths (less padding, faster)
    order = sorted(range(len(sents)), key=lambda i: len(sents[i]))
    out = [""] * len(sents)
    for i in range(0, len(order), batch_size):
        idx = order[i : i + batch_size]
        batch = tok([sents[j] for j in idx], return_tensors="pt", padding=True,
                    truncation=True, max_length=512).to(device)
        with torch.no_grad():
            gen = model.generate(**batch, num_beams=4)
        for j, t in zip(idx, tok.batch_decode(gen, skip_special_tokens=True)):
            out[j] = t
        print(f"  {min(i + batch_size, len(order))}/{len(order)} sentences", end="\r")
    print()
    return out


def translate(texts: list[str], tok, model, device, batch_size: int) -> list[str]:
    split = [split_sentences(t) for t in texts]
    flat = [s for sents in split for s in sents]
    translated = iter(translate_sentences(flat, tok, model, device, batch_size))
    return [" ".join(next(translated) for _ in sents) for sents in split]


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
