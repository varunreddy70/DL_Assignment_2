import numpy as np
from model import build_model
from sklearn.model_selection import train_test_split

# Load data
X_train = np.load("X_train.npy")
y_train = np.load("y_train.npy")
char_to_id = np.load("char_to_id.npy", allow_pickle=True).item()

# Split into train and validation sets
X_train, X_val, y_train, y_val = train_test_split(
    X_train, y_train, test_size=0.2, random_state=42
)

# Vocabulary sizes
num_encoder_tokens = len(char_to_id)
num_decoder_tokens = len(char_to_id)  

# Model configuration
config = {
    'rnn_type': 'LSTM',
    'embedding_dim': 512,
    'latent_dim': 1024,
    'encoder_layers': 1,  # Kept as 1 to match computation assumptions
    'decoder_layers': 1,  # Kept as 1 to match computation assumptions
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

model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Parameter count
trainable_params = model.count_params()
print(f"✅ Total Trainable Parameters: {trainable_params}")

# Computation count estimation
T = X_train.shape[1]  # Sequence length
m = config['embedding_dim']
k = config['latent_dim']
num_layers = 1  # since only 1 layer is actually implemented

lstm_computations = 4 * (m * k + k * k + k)  # per time step
total_computations_per_seq = T * lstm_computations * 2  # encoder + decoder
print(f"✅ Approximate Computations per Input-Output Pair: {total_computations_per_seq}")

# Training with validation
history = model.fit(
    [X_train, y_train[:, :-1]],  # Teacher forcing
    y_train[:, 1:],
    batch_size=config['batch_size'],
    epochs=config['epochs'],
    validation_data=(
        [X_val, y_val[:, :-1]],
        y_val[:, 1:]
    )
)

# Save model
model.save("transliteration_model.h5")

# Print final metrics
final_train_loss = history.history['loss'][-1]
final_val_loss = history.history['val_loss'][-1]
final_train_acc = history.history['accuracy'][-1]
final_val_acc = history.history['val_accuracy'][-1]

print("\n=== Final Metrics ===")
print(f"Final Training Loss: {final_train_loss:.4f}")
print(f"Final Validation Loss: {final_val_loss:.4f}")
print(f"Final Training Accuracy: {final_train_acc:.4f}")
print(f"Final Validation Accuracy: {final_val_acc:.4f}")
print("Training completed. Model saved.")