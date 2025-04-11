import numpy as np
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, LSTM, Dense, Embedding

# Load preprocessed data
X_train = np.load("X_train.npy")
y_train = np.load("y_train.npy")
char_to_id = np.load("char_to_id.npy", allow_pickle=True).item()
vocab_size = len(char_to_id) + 1

# Model parameters
embedding_size = 64  # 'm' in assignment
hidden_units = 128   # 'k' in assignment

# Encoder
encoder_inputs = Input(shape=(None,))
encoder_embedding = Embedding(vocab_size, embedding_size)(encoder_inputs)
encoder_lstm = LSTM(hidden_units, return_state=True)
_, state_h, state_c = encoder_lstm(encoder_embedding)
encoder_states = [state_h, state_c]

# Decoder
decoder_inputs = Input(shape=(None,))
decoder_embedding = Embedding(vocab_size, embedding_size)(decoder_inputs)
decoder_lstm = LSTM(hidden_units, return_sequences=True, return_state=True)
decoder_outputs, _, _ = decoder_lstm(decoder_embedding, initial_state=encoder_states)
decoder_dense = Dense(vocab_size, activation='softmax')
output = decoder_dense(decoder_outputs)

# Compile and train
model = Model([encoder_inputs, decoder_inputs], output)
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy')
model.fit([X_train, y_train[:, :-1]], y_train[:, 1:], epochs=10, batch_size=32)

# Save model
model.save("seq2seq_model.h5")