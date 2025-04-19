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


history = model.fit(
    [X_train, y_train[:, :-1]],  # Teacher forcing
    y_train[:, 1:],
    batch_size=config['batch_size'],
    epochs=config['epochs'],
    validation_split=0.2
)

model.save("transliteration_model.h5")
print("Training completed. Model saved.")
