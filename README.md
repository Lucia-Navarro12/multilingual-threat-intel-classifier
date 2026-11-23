# Multilingual Threat Intelligence Classifier

A small project to classify cybersecurity news by threat type (ransomware, data breach, DDoS...) in English and Spanish using multilingual BERT.

I wanted to see if a model trained only on English news can also classify Spanish news, since most labelled security data is in English.

[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Lucia-Navarro12/multilingual-threat-intel-classifier/blob/main/notebooks/colab.ipynb)

## Data

- [NewsDataHub/cybersecurity-news-dataset-english-3000](https://huggingface.co/datasets/NewsDataHub/cybersecurity-news-dataset-english-3000) (CC BY 4.0). I use the title + description of each article.
- 6 labels, an article can have more than one: `ransomware`, `data_breach`, `vulnerability`, `ddos`, `hacking`, `cyber_attack`.
- 2,920 articles after removing duplicates and articles with no label. Split 70/15/15.
- The Spanish version is machine translated with [opus-mt-en-es](https://huggingface.co/Helsinki-NLP/opus-mt-en-es).

## What I did

1. Baseline with TF-IDF + logistic regression.
2. Fine-tuned `bert-base-multilingual-cased` for multi-label classification.
3. Trained it twice: only with English (and tested on Spanish), and with both languages.

## Results

Macro F1 on the test set:

| Model | Trained on | EN | ES |
|---|---|---|---|
| TF-IDF + LogReg | en | 0.888 | 0.454 |
| TF-IDF + LogReg | en + es | 0.891 | 0.882 |
| mBERT | en | 0.919 | 0.769 |
| mBERT | en + es | 0.914 | 0.893 |

- Trained only on English, TF-IDF gets 0.45 on Spanish and mBERT gets 0.77, without seeing any Spanish example.
- Trained on both languages, the two models are close on Spanish (0.88 and 0.89).

Some examples with mBERT trained on both languages:

```
Ransomware gang encrypts hospital servers and demands bitcoin
  -> ransomware (0.98)
Filtrados los datos personales de dos millones de clientes de una aerolínea
  -> data_breach (0.94)
Ataque DDoS deja sin servicio las webs de varios ministerios
  -> ddos (0.91), cyber_attack (0.98)
```

## Notes

- The labels in the dataset come from keywords, so the TF-IDF baseline already does well in English.
- The Spanish data is translated, not real Spanish news.
- DDoS and vulnerability have very few examples (less than 60 each).

## How to run

Easiest is the Colab notebook (`notebooks/colab.ipynb`) with a T4 GPU. Or locally:

```bash
pip install -r requirements.txt
python src/data.py
python src/translate.py
python src/baseline.py --train-langs en
python src/train.py --train-langs en
python src/train.py --train-langs en es
python src/predict.py --model-dir models/bert-base-multilingual-cased_train-en-es/best "Ataque DDoS contra webs del Gobierno"
```
