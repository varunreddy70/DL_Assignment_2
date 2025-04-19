import numpy as np
from model import build_model


X_train = np.load("X_train.npy")
y_train = np.load("y_train.npy")
char_to_id = np.load("char_to_id.npy", allow_pickle=True).item()

# Vocabulary sizes
num_encoder_tokens = len(char_to_id)
num_decoder_tokens = len(char_to_id)  

# Model configuration
config = {
    'rnn_type': 'LSTM',
    'embedding_dim': 512,
    'latent_dim': 1024,
    'encoder_layers': 2,
    'decoder_layers': 3,
    'dropout': 0.1,
    'epochs': 5,
    'batch_size': 64
}

# Build and compile the model
model = build_model(
    rnn_type=config['rnn_type'],
    embedding_dim=config['embedding_dim'],
    latent_dim=config['latent_dim'],
    encoder_layers=config['encoder_layers'],
    decoder_layers=config['decoder_layers'],
    dropout=config['dropout'],
    num_encoder_tokens=num_encoder_tokens,
    num_decoder_tokens=num_decoder_tokens
)

model.compile(optimizer='adam', loss='sparse_categorical_crossentropy')

# PARAMETER COUNT
trainable_params = model.count_params()
print(f"✅ Total Trainable Parameters: {trainable_params}")

# COMPUTATION COUNT ESTIMATION
# Based on the assumption: LSTM computation per time step = 4 * (embedding_dim * hidden + hidden^2 + hidden)
T = X_train.shape[1]  # Sequence length
m = config['embedding_dim']
k = config['latent_dim']
num_layers = 1  # since only 1 layer is actually implemented in encoder/decoder

lstm_computations = 4 * (m * k + k * k + k)  # per time step
total_computations_per_seq = T * lstm_computations * 2  # encoder + decoder
print(f"✅ Approximate Computations per Input-Output Pair: {total_computations_per_seq}")

history = model.fit(
    [X_train, y_train[:, :-1]],  # Teacher forcing
    y_train[:, 1:],
    batch_size=config['batch_size'],
    epochs=config['epochs'],
    validation_split=0.2
)

model.save("transliteration_model.h5")
print("Training completed. Model saved.")
