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
        latent_dim = 1024  
        # Building the Encoder
        encoder_inputs = self.model.input[0]
        encoder_embedding_layer = self.model.get_layer("encoder_embedding")
        encoder_embedding = encoder_embedding_layer(encoder_inputs)
        # Get the encoder LSTM layer 
        encoder_lstm = self.model.get_layer("encoder_lstm")
        # Call the encoder LSTM
        encoder_outputs = encoder_lstm(encoder_embedding)
        state_h_enc = encoder_outputs[1]
        state_c_enc = encoder_outputs[2]
        self.encoder_model = Model(encoder_inputs, [state_h_enc, state_c_enc])
        # Building the Decoder
        decoder_inputs = Input(shape=(1,), name="decoder_input_test")
        decoder_state_input_h = Input(shape=(latent_dim,), name="input_h")
        decoder_state_input_c = Input(shape=(latent_dim,), name="input_c")
        decoder_states_inputs = [decoder_state_input_h, decoder_state_input_c]


        decoder_embedding_layer = self.model.get_layer("decoder_embedding") 
        decoder_embedding = decoder_embedding_layer(decoder_inputs)


        original_decoder_lstm = self.model.get_layer("decoder_lstm")
        decoder_lstm_infer = LSTM(
            original_decoder_lstm.units,
            return_sequences=True,
            return_state=True,
            dropout=original_decoder_lstm.dropout,
            name="decoder_lstm_infer"
        )

        decoder_lstm_infer.build(decoder_embedding.shape)
        decoder_lstm_infer.set_weights(original_decoder_lstm.get_weights())
        decoder_lstm_outputs = decoder_lstm_infer(decoder_embedding, initial_state=decoder_states_inputs)
        decoder_outputs = decoder_lstm_outputs[0]
        state_h_dec = decoder_lstm_outputs[1]
        state_c_dec = decoder_lstm_outputs[2]

        decoder_dense = self.model.get_layer("decoder_dense")
        decoder_outputs = decoder_dense(decoder_outputs)

        self.decoder_model = Model(
            [decoder_inputs] + decoder_states_inputs,
            [decoder_outputs, state_h_dec, state_c_dec]
        )

    def predict(self, input_text, max_length=20):

        input_seq = np.array([[self.char_to_id.get(c, self.char_to_id.get('<unk>', 0)) for c in input_text.lower()]])

        states_value = self.encoder_model.predict(input_seq)

        target_seq = np.zeros((1, 1))
        target_seq[0, 0] = self.char_to_id['<start>']

        output_text = ''
        for _ in range(max_length):
            
            output_tokens, h, c = self.decoder_model.predict([target_seq] + states_value)
            sampled_token_index = np.argmax(output_tokens[0, -1, :])
            sampled_char = self.id_to_char[sampled_token_index]

            if sampled_char == '<end>':
                break

            if sampled_char not in ['<pad>', '<unk>']:
                output_text += sampled_char


            target_seq[0, 0] = sampled_token_index
            # Update states.
            states_value = [h, c]

        return output_text

if __name__ == "__main__":
    transliterator = Transliterator("transliteration_model.h5")
    test_words = ["namaste", "pyaar", "dil", "shanti", "surya"]
    for word in test_words:
        print(f"{word.ljust(10)} → {transliterator.predict(word)}")
