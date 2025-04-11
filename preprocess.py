import pandas as pd
import numpy as np
from tensorflow.keras.preprocessing.sequence import pad_sequences

# Load and clean data
data = pd.read_csv("hi.translit.sampled.train.tsv", sep="\t", header=None,
                  names=["latin", "devanagari", "count"])

# Filter out non-string and empty values
data = data[(data["latin"].apply(lambda x: isinstance(x, str))) & 
            (data["devanagari"].apply(lambda x: isinstance(x, str)))]

# Create character-level mappings
latin_chars = set(char for word in data["latin"].str.lower() for char in word)
devanagari_chars = set(char for word in data["devanagari"] for char in word)

# Ensure essential Latin characters are included
essential_latin = set('abcdefghijklmnopqrstuvwxyz')
all_chars = latin_chars.union(devanagari_chars).union(essential_latin)

# Create vocabulary with Latin characters first
sorted_chars = sorted(all_chars, key=lambda x: (not x.isascii(), x))
char_to_id = {char: i+1 for i, char in enumerate(sorted_chars)}
id_to_char = {v: k for k, v in char_to_id.items()}

# Convert words to character sequences
data["latin_seq"] = data["latin"].str.lower().apply(lambda x: [char_to_id[c] for c in x])
data["devanagari_seq"] = data["devanagari"].apply(lambda x: [char_to_id[c] for c in x])

# Pad sequences
max_len = max(max(len(seq) for seq in data["latin_seq"]),
              max(len(seq) for seq in data["devanagari_seq"]))

X_train = pad_sequences(data["latin_seq"], maxlen=max_len, padding="post")
y_train = pad_sequences(data["devanagari_seq"], maxlen=max_len, padding="post")

# Save processed data
np.save("X_train.npy", X_train)
np.save("y_train.npy", y_train)
np.save("char_to_id.npy", char_to_id)
np.save("id_to_char.npy", id_to_char)

print("Preprocessing complete!")
print(f"Vocabulary size: {len(char_to_id)}")
print(f"Sample Latin mappings: {dict(list(char_to_id.items())[:10])}")