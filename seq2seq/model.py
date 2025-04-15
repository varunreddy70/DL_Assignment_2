from tensorflow.keras.layers import Input, LSTM, GRU, Dense, Embedding
from tensorflow.keras.models import Model

def build_model(rnn_type, embedding_dim, latent_dim, encoder_layers, decoder_layers, dropout=0.1, num_encoder_tokens=0, num_decoder_tokens=0):
    # Remove the global variable assignment for latent_dim
    # latent_dim is now passed as an argument, no need to set it globally
    
    # Define input layers
    encoder_inputs = Input(shape=(None,), name="encoder_input")
    decoder_inputs = Input(shape=(None,), name="decoder_input")

    # Embeddings
    encoder_embedding = Embedding(num_encoder_tokens, embedding_dim, name="encoder_embedding")(encoder_inputs)
    decoder_embedding = Embedding(num_decoder_tokens, embedding_dim, name="decoder_embedding")(decoder_inputs)

    # Encoder
    if rnn_type == 'LSTM':
        encoder = LSTM(latent_dim, return_state=True, name="encoder_lstm", dropout=dropout)
        encoder_outputs, state_h, state_c = encoder(encoder_embedding)
        encoder_states = [state_h, state_c]
    else:
        encoder = GRU(latent_dim, return_state=True, name="encoder_gru", dropout=dropout)
        encoder_outputs, state_h = encoder(encoder_embedding)
        encoder_states = [state_h]

    # Decoder
    if rnn_type == 'LSTM':
        decoder = LSTM(latent_dim, return_sequences=True, return_state=True, name="decoder_lstm", dropout=dropout)
        decoder_outputs, _, _ = decoder(decoder_embedding, initial_state=encoder_states)
    else:
        decoder = GRU(latent_dim, return_sequences=True, return_state=True, name="decoder_gru", dropout=dropout)
        decoder_outputs, _ = decoder(decoder_embedding, initial_state=encoder_states)

    decoder_dense = Dense(num_decoder_tokens, activation='softmax', name="decoder_dense")
    decoder_outputs = decoder_dense(decoder_outputs)

    model = Model([encoder_inputs, decoder_inputs], decoder_outputs)
    return model
