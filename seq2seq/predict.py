import numpy as np
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.layers import Input, LSTM

class Transliterator:
    def __init__(self, model_path):
        self.model = load_model(model_path)
        self.char_to_id = np.load("char_to_id.npy", allow_pickle=True).item()
        self.id_to_char = {v: k for k, v in self.char_to_id.items()}
        self._build_inference_model()

    def _build_inference_model(self):
        latent_dim = 1024  # This should match your training configuration

        # -------------------
        # Build the Encoder
        # -------------------
        # Get the encoder input (auto-named "input_layer")
        encoder_inputs = self.model.input[0]
        # Retrieve the encoder embedding layer ("embedding")
        encoder_embedding_layer = self.model.get_layer("encoder_embedding")
        encoder_embedding = encoder_embedding_layer(encoder_inputs)
        # Get the encoder LSTM layer ("lstm")
        encoder_lstm = self.model.get_layer("encoder_lstm")
        # Call the encoder LSTM; it returns (output_sequence, state_h, state_c)
        encoder_outputs = encoder_lstm(encoder_embedding)
        state_h_enc = encoder_outputs[1]
        state_c_enc = encoder_outputs[2]
        # Define the inference encoder model
        self.encoder_model = Model(encoder_inputs, [state_h_enc, state_c_enc])

        # -------------------
        # Build the Decoder
        # -------------------
        # Define a new input for the decoder that will receive one token at a time
        decoder_inputs = Input(shape=(1,), name="decoder_input_test")
        # Also define inputs for the decoder's internal states
        decoder_state_input_h = Input(shape=(latent_dim,), name="input_h")
        decoder_state_input_c = Input(shape=(latent_dim,), name="input_c")
        decoder_states_inputs = [decoder_state_input_h, decoder_state_input_c]

        # Retrieve the decoder embedding layer ("embedding_1")
        decoder_embedding_layer = self.model.get_layer("decoder_embedding") 
        decoder_embedding = decoder_embedding_layer(decoder_inputs)

        # Instead of reusing the existing decoder LSTM (which may not return states),
        # we create a new LSTM layer with the same configuration and copy its weights.
        original_decoder_lstm = self.model.get_layer("decoder_lstm")
        decoder_lstm_infer = LSTM(
            original_decoder_lstm.units,
            return_sequences=True,
            return_state=True,
            dropout=original_decoder_lstm.dropout,
            name="decoder_lstm_infer"
        )
        # Build the new decoder LSTM for the given input shape and assign weights.
        decoder_lstm_infer.build(decoder_embedding.shape)
        decoder_lstm_infer.set_weights(original_decoder_lstm.get_weights())

        # Get the outputs and new states from the new decoder LSTM layer.
        decoder_lstm_outputs = decoder_lstm_infer(decoder_embedding, initial_state=decoder_states_inputs)
        decoder_outputs = decoder_lstm_outputs[0]
        state_h_dec = decoder_lstm_outputs[1]
        state_c_dec = decoder_lstm_outputs[2]

        # Retrieve the final dense layer ("dense") to generate token probabilities.
        decoder_dense = self.model.get_layer("decoder_dense")
        decoder_outputs = decoder_dense(decoder_outputs)

        # Define the inference decoder model.
        self.decoder_model = Model(
            [decoder_inputs] + decoder_states_inputs,
            [decoder_outputs, state_h_dec, state_c_dec]
        )

    def predict(self, input_text, max_length=20):
        # Convert input text into sequence of ids. Use <unk> if a character is not found.
        input_seq = np.array([[self.char_to_id.get(c, self.char_to_id.get('<unk>', 0)) for c in input_text.lower()]])
        # Get encoder states
        states_value = self.encoder_model.predict(input_seq)

        # Create a target sequence of length 1 with the start token.
        target_seq = np.zeros((1, 1))
        target_seq[0, 0] = self.char_to_id['<start>']

        output_text = ''
        for _ in range(max_length):
            # Predict the next token and states given the previous token and current states.
            output_tokens, h, c = self.decoder_model.predict([target_seq] + states_value)
            # Get the index with maximum probability from the output token vector.
            sampled_token_index = np.argmax(output_tokens[0, -1, :])
            sampled_char = self.id_to_char[sampled_token_index]

            if sampled_char == '<end>':
                break

            # Skip special tokens like <pad> or <unk> if any accidentally predicted
            if sampled_char not in ['<pad>', '<unk>']:
                output_text += sampled_char

            # Update target sequence: the next input token is the one we just predicted.
            target_seq[0, 0] = sampled_token_index
            # Update states.
            states_value = [h, c]

        return output_text

if __name__ == "__main__":
    transliterator = Transliterator("transliteration_model.h5")
    test_words = ["namaste", "pyaar", "dil", "shanti", "surya"]
    for word in test_words:
        print(f"{word.ljust(10)} → {transliterator.predict(word)}")
