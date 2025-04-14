import numpy as np
from tensorflow.keras.models import load_model, Model
from tensorflow.keras.layers import Input, LSTM, Embedding, Dense

# Load the full trained model
model = load_model("transliteration_model.h5")
char_to_id = np.load("char_to_id.npy", allow_pickle=True).item()
vocab_size = len(char_to_id)

# Extract embedding_dim and lstm_units from the model
embedding_dim = model.get_layer('embedding').output_dim  # Access the output_dim directly
lstm_units = model.get_layer('lstm').units

# === Rebuild encoder ===
# Encoder setup
encoder_inputs = Input(shape=(None,), name="encoder_inputs")
encoder_embedding_layer = Embedding(input_dim=vocab_size, output_dim=embedding_dim, name="encoder_embedding")
encoder_embedding = encoder_embedding_layer(encoder_inputs)
encoder_lstm_layer = LSTM(lstm_units, return_state=True, name="encoder_lstm")
_, state_h, state_c = encoder_lstm_layer(encoder_embedding)
encoder_model = Model(encoder_inputs, [state_h, state_c])
encoder_model.save("encoder_model.h5")

# === Rebuild decoder ===
decoder_inputs = Input(shape=(None,), name="decoder_inputs")
decoder_state_input_h = Input(shape=(lstm_units,), name="decoder_h")
decoder_state_input_c = Input(shape=(lstm_units,), name="decoder_c")

decoder_embedding_layer = Embedding(input_dim=vocab_size, output_dim=embedding_dim, name="decoder_embedding")
decoder_embedding = decoder_embedding_layer(decoder_inputs)

decoder_lstm_layer = LSTM(lstm_units, return_sequences=True, return_state=True, name="decoder_lstm")
decoder_outputs, state_h_dec, state_c_dec = decoder_lstm_layer(
    decoder_embedding, initial_state=[decoder_state_input_h, decoder_state_input_c]
)

decoder_dense_layer = Dense(vocab_size, activation='softmax', name="decoder_dense")
decoder_outputs = decoder_dense_layer(decoder_outputs)

decoder_model = Model(
    [decoder_inputs, decoder_state_input_h, decoder_state_input_c],
    [decoder_outputs, state_h_dec, state_c_dec]
)
decoder_model.save("decoder_model.h5")

print("✅ Saved encoder_model.h5 and decoder_model.h5")
