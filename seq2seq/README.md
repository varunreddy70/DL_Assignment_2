# Sequence-to-Sequence Transliteration Model

This repository contains an RNN-based Seq2Seq model for transliterating words from **Latin script to Devanagari script**, built as part of a Deep Learning assignment using the [Dakshina dataset](https://github.com/google-research-datasets/dakshina).

---

## Model Architecture

- Encoder-Decoder architecture using **LSTM** or **GRU** cells
- Character-level **embeddings**
- **Configurable hyperparameters**: embedding dim, hidden units, dropout, RNN type
- Uses **teacher forcing** during training
- Trained on word-level transliteration pairs

---

## Requirements

- Python 3.8+
- TensorFlow 2.x
- pandas
- numpy

---

## Files

| File | Description |
|------|-------------|
| `train.py` | Training script (includes param/computation estimation) |
| `evaluate.py` | Test set evaluation (optional) |
| `predict.py` | Interactive predictions for sample words |
| `model.py` | Configurable model architecture |
| `preprocess.py` | Preprocesses the dataset and builds vocab |
| `inference_setup.py` | Builds and saves encoder/decoder models for inference |
| `transliteration_model.h5` | Saved trained model |
| `*.npy` | Preprocessed input/output sequences and vocab |

---

## Usage

### 1. Train the Model

python train.py
Outputs:

Trained model (transliteration_model.h5)

Training/validation accuracy

Parameter count and computation cost estimation

### 2. Evaluate on Test Set (Optional)

python evaluate.py
Reports:

Test accuracy

Sample predictions

### 3. Predict Custom Words

python predict.py

Generates:
Transliteration of predefined sample words (e.g., namaste, pyaar, dil)


## Results

| Metric          | Value        |
|-----------------|--------------|
| Training Accuracy | 95.66%       |
| Validation Accuracy | 95.12%      |
| Test Accuracy   | 93.91%       |
| Model Parameters | ~12.7M       |

### Performance

Below are the outputs from our training, evaluation, and prediction scripts:

#### Training Output
![Training Output](![train 3](https://github.com/user-attachments/assets/49a6d667-b2d0-4f1d-ad39-b2a53483d0b5)

*Figure 1: Output of train.py showing training and validation accuracy/loss over epochs*

#### Evaluation Output
![Evaluation Output](![evaluate](https://github.com/user-attachments/assets/90c31d80-3ea1-46de-a73c-441afcebd70f)

*Figure 2: Output of evaluate.py showing test accuracy and loss metrics*

#### Prediction Samples
![Prediction Samples](![predict 1](https://github.com/user-attachments/assets/44763bf7-58e5-48b9-a07d-fcfad6dd88a6)
*Figure 3: Sample predictions from predict.py for predefined test words*

### Sample Predictions

| Input Word | True Transliteration | Predicted Transliteration |
|------------|----------------------|---------------------------|
| namaste    | नमस्ते              | नमस्ते                   |
| pyaar      | प्रेम                | प्रया                    |
| dil        | दिल                 | हिकल                    |


## References
-Dakshina Dataset  
-Keras Seq2Seq Guide
