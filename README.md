# Deep Learning Assignment – 2

This repository contains two deep learning projects developed as part of a sequence modeling assignment. Each explores a different approach to working with sequential data using modern neural architectures.

---

## 1. Seq2Seq Transliteration Model

- Implements a **character-level encoder-decoder RNN** for transliterating words from **Latin script to Devanagari script**
- Supports **LSTM**, **GRU**, or vanilla RNN cells
- Uses the [Dakshina dataset](https://github.com/google-research-datasets/dakshina)
- Includes training, evaluation, and interactive prediction scripts
- Reports accuracy and estimates parameter/computation costs

 See `seq2seq/README.md` for architecture details and usage instructions

---

## 2. GPT-2 Lyrics Generator

- Fine-tunes a **GPT-2 language model** to generate creative English song lyrics
- Built using HuggingFace's `transformers` library
- Trains on a custom lyrics dataset (e.g., [Kaggle Poetry](https://www.kaggle.com/datasets/paultimothymooney/poetry))
- Supports saving/loading models and generating multiple lyric variations

📄 See `GPT2/README.md` for the full workflow, generation examples, and references

---

## Tools & Libraries

- Python 3.8+
- TensorFlow 2.x (for Seq2Seq)
- PyTorch (for GPT-2)
- HuggingFace Transformers
- NumPy, Pandas

---



