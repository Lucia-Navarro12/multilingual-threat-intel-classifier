"""Classical baseline: TF-IDF + one-vs-rest logistic regression, evaluated per language."""
import argparse
import json
from pathlib import Path

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.multiclass import OneVsRestClassifier
from sklearn.pipeline import make_pipeline

from common import load, metrics
from data import LABELS


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-langs", nargs="+", default=["en", "es"])
    parser.add_argument("--test-langs", nargs="+", default=["en", "es"])
    args = parser.parse_args()

    train = load("train", args.train_langs)
    model = make_pipeline(
        TfidfVectorizer(ngram_range=(1, 2), min_df=2, sublinear_tf=True),
        OneVsRestClassifier(LogisticRegression(max_iter=2000, class_weight="balanced")),
    )
    model.fit(train["text"], train[LABELS].values)

    results = {}
    for lang in args.test_langs:
        test = load("test", [lang])
        results[lang] = metrics(test[LABELS].values, model.predict(test["text"]))
        print(f"[{lang}] F1 micro={results[lang]['f1_micro']:.3f} macro={results[lang]['f1_macro']:.3f}")

    Path("results").mkdir(exist_ok=True)
    name = f"results/baseline_train-{'-'.join(args.train_langs)}.json"
    Path(name).write_text(json.dumps(results, indent=2))
    print(f"Saved {name}")


if __name__ == "__main__":
    main()
