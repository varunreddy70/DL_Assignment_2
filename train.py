import numpy as np
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, LSTM, Dense, Embedding
from tensorflow.keras.callbacks import EarlyStopping

# Load preprocessed data
X_train = np.load("X_train.npy")
y_train = np.load("y_train.npy")
char_to_id = np.load("char_to_id.npy", allow_pickle=True).item()
vocab_size = len(char_to_id)

# Model parameters
embedding_dim = 64
latent_dim = 128
batch_size = 32
epochs = 30

# Encoder
encoder_inputs = Input(shape=(None,))
encoder_embedding = Embedding(vocab_size, embedding_dim)(encoder_inputs)
encoder_lstm = LSTM(latent_dim, return_state=True)
_, state_h, state_c = encoder_lstm(encoder_embedding)
encoder_states = [state_h, state_c]

# Decoder
decoder_inputs = Input(shape=(None,))
decoder_embedding = Embedding(vocab_size, embedding_dim)(decoder_inputs)
decoder_lstm = LSTM(latent_dim, return_sequences=True, return_state=True)
decoder_outputs, _, _ = decoder_lstm(decoder_embedding, initial_state=encoder_states)
decoder_dense = Dense(vocab_size, activation='softmax')
decoder_outputs = decoder_dense(decoder_outputs)

# Model
model = Model([encoder_inputs, decoder_inputs], decoder_outputs)
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy')

# Training
history = model.fit(
    [X_train, y_train[:, :-1]],
    np.expand_dims(y_train[:, 1:], -1),
    batch_size=batch_size,
    epochs=epochs,
    validation_split=0.2,
    callbacks=[EarlyStopping(patience=3)]
)

model.save("seq2seq_model.h5")
print("Training complete! Model saved.")