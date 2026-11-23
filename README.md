# Multilingual Threat Intelligence Classifier

Multi-label classification of cybersecurity news by threat type, in **English and Spanish**, by fine-tuning multilingual BERT.

[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Lucia-Navarro12/multilingual-threat-intel-classifier/blob/main/notebooks/colab.ipynb)

## The question

Threat-intelligence teams monitor news in many languages, but labelled data is mostly in English.
**Can a model trained on English security news classify Spanish news it has never seen?**

## Data

| | |
|---|---|
| Source | [NewsDataHub/cybersecurity-news-dataset-english-3000](https://huggingface.co/datasets/NewsDataHub/cybersecurity-news-dataset-english-3000) (CC BY 4.0) |
| Text | Article title + description |
| Labels (multi-label) | `ransomware`, `data_breach`, `vulnerability`, `ddos`, `hacking`, `cyber_attack` |
| Size | 2,920 articles after de-duplication and dropping unlabelled rows |
| Splits | 70 / 15 / 15, stratified on the rarest label present |
| Spanish | Machine-translated per split with [Helsinki-NLP/opus-mt-en-es](https://huggingface.co/Helsinki-NLP/opus-mt-en-es), so no article leaks across splits |

Label counts in the training split: data_breach 942 · cyber_attack 807 · hacking 559 · ransomware 354 · vulnerability 59 · ddos 42.

## Approach

1. **Baseline**: TF-IDF (1–2-grams) + one-vs-rest logistic regression with class weighting.
2. **Transformer**: `bert-base-multilingual-cased` with a sigmoid multi-label head (BCE loss), 4 epochs, best epoch picked on validation macro-F1.
3. **Two training regimes**:
   - `en`: English only, then tested on Spanish → **zero-shot cross-lingual transfer**
   - `en es`: both languages

## Results

Test set, F1 (macro averages over the 6 labels, so the rare classes count as much as the common ones).

| Model | Trained on | Test EN macro-F1 | Test ES macro-F1 |
|---|---|---|---|
| TF-IDF + LogReg | en | 0.888 | _run notebook_ |
| TF-IDF + LogReg | en + es | _run notebook_ | _run notebook_ |
| mBERT | en | _run notebook_ | _run notebook_ |
| mBERT | en + es | _run notebook_ | _run notebook_ |

## Limitations

- **Weak labels**: the dataset's flags are keyword-derived ("mentions ransomware"), which is why a bag-of-words baseline is already strong in English. The real test is the language shift.
- **Translated Spanish**: the Spanish test set is machine-translated, not native Spanish news. A small hand-labelled set of real Spanish headlines would be the next step.
- **Rare classes**: DDoS and vulnerability have under 60 training examples each; their F1 is the least stable.

## Run it

```bash
pip install -r requirements.txt
python src/data.py                       # download + splits      -> data/en
python src/translate.py                  # EN -> ES (GPU advised)  -> data/es
python src/baseline.py --train-langs en
python src/train.py --train-langs en     # zero-shot transfer to Spanish
python src/train.py --train-langs en es
python src/predict.py --model-dir models/bert-base-multilingual-cased_train-en-es/best "Ataque DDoS contra webs del Gobierno"
```

Or open `notebooks/colab.ipynb` in Google Colab with a T4 GPU (about 30–45 minutes end to end).

## Structure

```
src/
  data.py        download, labels, stratified splits
  translate.py   MarianMT English -> Spanish
  baseline.py    TF-IDF + logistic regression
  train.py       multilingual BERT fine-tuning (Hugging Face Trainer)
  predict.py     classify new headlines
  common.py      data loading + metrics
notebooks/colab.ipynb
results/         JSON metrics per run
```
