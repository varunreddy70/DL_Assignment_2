import pandas as pd
import numpy as np
from tensorflow.keras.preprocessing.sequence import pad_sequences

# Devanagari character order
DEVANAGARI_ORDER = [
    'अ','आ','इ','ई','उ','ऊ','ऋ','ए','ऐ','ओ','औ',
    'क','ख','ग','घ','ङ','च','छ','ज','झ','ञ',
    'ट','ठ','ड','ढ','ण','त','थ','द','ध','न',
    'प','फ','ब','भ','म','य','र','ल','व','श',
    'ष','स','ह',
    'ा','ि','ी','ु','ू','ृ','े','ै','ो','ौ','ं','ः','ँ'
]

def devanagari_sort_key(char):
    try:
        return DEVANAGARI_ORDER.index(char)
    except ValueError:
        return len(DEVANAGARI_ORDER)

# Load and clean data
data = pd.read_csv("hi.translit.sampled.train.tsv", sep="\t", header=None,
                  names=["devanagari", "latin", "count"])

# Data cleaning
data = data.dropna()
data = data[(data["latin"].apply(lambda x: isinstance(x, str))) &
            (data["devanagari"].apply(lambda x: isinstance(x, str)))]

# Reverse direction
data = data.rename(columns={"latin": "source", "devanagari": "target"})

# Create vocabulary
latin_chars = set(char for word in data["source"].str.lower() for char in word)
devanagari_chars = set(char for word in data["target"] for char in word)
all_chars = latin_chars.union(devanagari_chars).union(set('abcdefghijklmnopqrstuvwxyz'))

# Special tokens
special_tokens = ['<start>', '<end>', '<pad>', '<unk>']
for token in special_tokens:
    all_chars.add(token)

# Sorted mappings
sorted_chars = (
    sorted([c for c in all_chars if c.isascii() and c not in special_tokens]) +
    sorted([c for c in all_chars if not c.isascii() and c not in special_tokens], 
           key=devanagari_sort_key) +
    special_tokens
)
char_to_id = {char: i for i, char in enumerate(sorted_chars)}

# Sequence conversion
data["source_seq"] = data["source"].apply(
    lambda x: [char_to_id[c] for c in x.lower() if c in char_to_id])
data["target_seq"] = data["target"].apply(
    lambda x: [char_to_id['<start>']] + 
              [char_to_id[c] for c in x if c in char_to_id] +
              [char_to_id['<end>']])

# Padding
max_len = max(max(len(s) for s in data["source_seq"]),
              max(len(s) for s in data["target_seq"]))
X_train = pad_sequences(data["source_seq"], maxlen=max_len, padding="post", 
                       value=char_to_id['<pad>'])
y_train = pad_sequences(data["target_seq"], maxlen=max_len, padding="post",
                       value=char_to_id['<pad>'])

# Save files
np.save("X_train.npy", X_train)
np.save("y_train.npy", y_train)
np.save("char_to_id.npy", char_to_id)
np.save("id_to_char.npy", {v:k for k,v in char_to_id.items()})

print("Preprocessing completed successfully!")
print(f"Vocabulary size: {len(char_to_id)}")
print(f"Sample Latin: {sorted_chars[:5]}")
print(f"Sample Devanagari: {[c for c in sorted_chars if not c.isascii()][:5]}")