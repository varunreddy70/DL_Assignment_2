import numpy as np
from tensorflow.keras.models import load_model

# Load model and vocab
model = load_model("seq2seq_model.h5")
char_to_id = np.load("char_to_id.npy", allow_pickle=True).item()
id_to_char = {v: k for k, v in char_to_id.items()}

# Predict function
def predict(word):
    # Convert input to IDs
    input_ids = np.array([[char_to_id[c] for c in word.lower()]])
    # Predict Devanagari characters
    pred = model.predict([input_ids, np.zeros((1, 1))])  # Dummy decoder input
    pred_ids = np.argmax(pred[0], axis=-1)
    return ''.join([id_to_char[i] for i in pred_ids if i in id_to_char])

# Test
print(predict("namaste"))  # Expected: "नमस्ते"