# Sequence-to-Sequence Transliteration Model

This repository contains an RNN-based seq2seq model for transliterating Latin script to Devanagari script, implemented as part of a Deep Learning assignment using the Dakshina dataset.

## Model Architecture
- **Encoder-Decoder** with LSTM/GRU cells
- Character-level embeddings
- Configurable hyperparameters (embedding dim, hidden units, etc.)
- Teacher forcing during training

## Requirements
- Python 3.8+
- TensorFlow 2.x
- pandas
- numpy

## Files
.
train.py # Training script
evaluate.py # Test set evaluation
predict.py # Prediction interface
model.py # Model architecture
preprocess.py # Data preprocessing
inference_setup.py # Encoder/decoder setup for inference
transliteration_model.h5 # Saved model
*.npy # Preprocessed data files


## Usage

### 1. Training
bash
python train.py
Outputs:

Model file (transliteration_model.h5)

Training/validation metrics

Parameter and computation counts

### 2. Evaluation
bash
python evaluate.py
Reports:

Test loss

Test accuracy

### 3. Prediction
bash
python predict.py
Generates:

Predictions for custom words (namaste, pyaar, etc.)

Random samples from test set

## Results
Metric	Value
Training Acc	95.66%
Validation Acc	95.12%
Test Accuracy	93.91%
Parameters	12.7M
Sample predictions:

namaste → समास्त्री
pyaar → प्रया
dil → हिकल

## References
Dakshina Dataset

Keras Seq2Seq Guide


### Key features:
1. Clear structure matching your project
2. Documents all key scripts
3. Shows sample results
4. Includes configuration options
5. Provides reference links
