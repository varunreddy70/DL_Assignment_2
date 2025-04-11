import numpy as np
from tensorflow.keras.models import load_model

# Load assets
model = load_model("seq2seq_model.h5")
char_to_id = np.load("char_to_id.npy", allow_pickle=True).item()
id_to_char = np.load("id_to_char.npy", allow_pickle=True).item()

def predict(word):
    try:
        # Convert input to sequence
        input_seq = np.array([[char_to_id[c] for c in word.lower()]])
        
        # Initialize target sequence
        target_seq = np.zeros((1, len(word) + 10))  # Extra space for output
        
        # Generate predictions
        for i in range(len(word) + 10 - 1):
            predictions = model.predict([input_seq, target_seq], verbose=0)
            next_id = np.argmax(predictions[0, i])
            if next_id == 0:  # Stop at padding
                break
            target_seq[0, i+1] = next_id
        
        # Convert to characters
        output = "".join([id_to_char.get(int(i), "") for i in target_seq[0] if i > 0])
        return output
    except Exception as e:
        return f"Error: {str(e)}"

# Test cases
test_words = ["namaste", "ghar", "pyaar", "dhanyavad", "hello"]
for word in test_words:
    print(f"Input: '{word}' → Output: '{predict(word)}'")

# Vocabulary verification
print("\nVocabulary check:")
print(f"Latin 'a' exists: {'a' in char_to_id}")
print(f"Devanagari 'न' exists: {'न' in char_to_id}")