import tensorflow as tf
from tensorflow.keras.layers import Input, LSTM, Dense, Embedding
from tensorflow.keras.models import Model
import numpy as np

def build_model(vocab_size, embedding_dim=256, hidden_units=512, cell_type='lstm'):
    # Encoder
    encoder_inputs = Input(shape=(None,))
    encoder_embedding = Embedding(vocab_size, embedding_dim)(encoder_inputs)
    
    if cell_type == 'lstm':
        encoder_rnn = LSTM(hidden_units, return_state=True)
    elif cell_type == 'gru':
        encoder_rnn = GRU(hidden_units, return_state=True)
        
    _, state_h, state_c = encoder_rnn(encoder_embedding)
    encoder_states = [state_h, state_c] if cell_type == 'lstm' else [state_h]

    # Decoder
    decoder_inputs = Input(shape=(None,))
    decoder_embedding = Embedding(vocab_size, embedding_dim)(decoder_inputs)
    
    decoder_rnn = LSTM(hidden_units, return_sequences=True, return_state=True) \
                   if cell_type == 'lstm' else GRU(hidden_units, return_sequences=True, return_state=True)
                   
    #decoder_outputs, _ = decoder_rnn(decoder_embedding, initial_state=encoder_states)
    if cell_type == 'lstm':
        decoder_outputs, _, _ = decoder_rnn(decoder_embedding, initial_state=encoder_states)
    else:
        decoder_outputs, _ = decoder_rnn(decoder_embedding, initial_state=encoder_states)

    decoder_dense = Dense(vocab_size, activation='softmax')(decoder_outputs)

    model = Model([encoder_inputs, decoder_inputs], decoder_dense)
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy')
    return model

if __name__ == "__main__":
    char_to_id = np.load("char_to_id.npy", allow_pickle=True).item()
    X_train = np.load("X_train.npy")
    y_train = np.load("y_train.npy")
    
    model = build_model(len(char_to_id), cell_type='lstm')
    model.fit(
        [X_train[:, :-1], y_train[:, :-1]],  # Teacher forcing
        y_train[:, 1:], 
        batch_size=64,
        epochs=10,
        validation_split=0.2
    )
    model.save("transliteration_model.h5")