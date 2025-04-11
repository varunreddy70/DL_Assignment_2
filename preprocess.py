import pandas as pd
import numpy as np
from tensorflow.keras.preprocessing.sequence import pad_sequences

# Load data and drop rows with missing values
data = pd.read_csv("hi.translit.sampled.train.tsv", sep="\t", header=None, 
                   names=["latin", "devanagari", "count"])
data = data.dropna()  # Remove rows with NaN values

# Clean and split into characters
data["latin_chars"] = data["latin"].str.lower().apply(list)
data["devanagari_chars"] = data["devanagari"].astype(str).apply(list)  # Ensure string type

# Create vocabulary
all_chars = set(char for word in data["latin_chars"] for char in word) | \
            set(char for word in data["devanagari_chars"] for char in word)
char_to_id = {char: i+1 for i, char in enumerate(all_chars)}  # 0 reserved for padding
vocab_size = len(char_to_id) + 1

# Convert words to IDs and pad sequences
X_train = [[char_to_id[c] for c in word] for word in data["latin_chars"]]
y_train = [[char_to_id[c] for c in word] for word in data["devanagari_chars"]]
X_train = pad_sequences(X_train, padding='post')
y_train = pad_sequences(y_train, padding='post')

# Save processed data
np.save("X_train.npy", X_train)
np.save("y_train.npy", y_train)
np.save("char_to_id.npy", char_to_id)

print("Preprocessing completed! Files saved: X_train.npy, y_train.npy, char_to_id.npy")