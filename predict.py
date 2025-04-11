import numpy as np
from tensorflow.keras.models import load_model

# Load model and vocab
model = load_model("seq2seq_model.h5")
char_to_id = np.load("char_to_id.npy", allow_pickle=True).item()
id_to_char = {v: k for k, v in char_to_id.items()}

# Debug: Print loaded vocab (first 10 items)
print("Sample vocabulary:", dict(list(char_to_id.items())[:10]))

# Predict function
def predict(word):
    try:
        # Convert input to IDs
        input_ids = np.array([[char_to_id[c] for c in word.lower()]])
        # Predict Devanagari characters (dummy decoder input)
        pred = model.predict([input_ids, np.zeros((1, len(word)))])
        pred_ids = np.argmax(pred[0], axis=-1)
        # Filter out padding (0) and unknown IDs
        output = ''.join([id_to_char[i] for i in pred_ids if i != 0 and i in id_to_char])
        return output
    except KeyError as e:
        return f"Error: Character '{e.args[0]}' not in vocabulary."

# Test with example words
test_words = ["namaste", "ghar", "pyaar", "dhanyavad"]
for word in test_words:
    print(f"Input: '{word}' → Output: '{predict(word)}'")