import numpy as np
from tensorflow.keras.models import load_model, Model
from tensorflow.keras.layers import Input

def load_assets():
    model = load_model("seq2seq_model.h5")
    char_to_id = np.load("char_to_id.npy", allow_pickle=True).item()
    id_to_char = np.load("id_to_char.npy", allow_pickle=True).item()
    return model, char_to_id, id_to_char

def create_inference_model(model):
    # Encoder
    encoder_inputs = Input(shape=(None,))
    encoder_embedding_layer = model.layers[1]
    encoder_embedding = encoder_embedding_layer(encoder_inputs)

    encoder_lstm_layer = model.layers[3]
    encoder_outputs, state_h_enc, state_c_enc = encoder_lstm_layer(encoder_embedding)
    encoder_states = [state_h_enc, state_c_enc]
    encoder_model = Model(encoder_inputs, encoder_states)

    # Decoder
    decoder_inputs = Input(shape=(None,))
    decoder_embedding_layer = model.layers[2]
    decoder_embedding = decoder_embedding_layer(decoder_inputs)

    decoder_state_input_h = Input(shape=(128,))
    decoder_state_input_c = Input(shape=(128,))
    decoder_states_inputs = [decoder_state_input_h, decoder_state_input_c]

    decoder_lstm_layer = model.layers[4]
    decoder_outputs, state_h_dec, state_c_dec = decoder_lstm_layer(
        decoder_embedding, initial_state=decoder_states_inputs
    )
    decoder_states = [state_h_dec, state_c_dec]

    decoder_dense = model.layers[5]
    decoder_outputs = decoder_dense(decoder_outputs)

    decoder_model = Model(
        [decoder_inputs] + decoder_states_inputs,
        [decoder_outputs] + decoder_states
    )

    return encoder_model, decoder_model


def predict_word(input_text, encoder, decoder, char_to_id, id_to_char, max_length=20):
    try:
        # Encode input
        input_seq = np.array([[char_to_id[c] for c in input_text.lower() if c in char_to_id]])
        states_value = encoder.predict(input_seq, verbose=0)
        
        # Generate empty target sequence
        target_seq = np.zeros((1, 1))
        target_seq[0, 0] = char_to_id['<start>']
        
        # Generate output
        output_text = ''
        for _ in range(max_length):
            output_tokens, h, c = decoder.predict(
                [target_seq] + states_value, verbose=0
            )
            sampled_token_index = np.argmax(output_tokens[0, -1, :])
            sampled_char = id_to_char[sampled_token_index]
            
            if sampled_char == '<end>' or sampled_token_index == 0:
                break
                
            output_text += sampled_char
            target_seq = np.zeros((1, 1))
            target_seq[0, 0] = sampled_token_index
            states_value = [h, c]
            
        return output_text
    except Exception as e:
        return f"Prediction error: {str(e)}"

def main():
    # Load all required assets
    model, char_to_id, id_to_char = load_assets()
    
    # Create inference models
    encoder, decoder = create_inference_model(model)
    
    # Test cases
    test_words = ["namaste", "ghar", "pyaar", "dhanyavad", "hello"]
    for word in test_words:
        result = predict_word(word, encoder, decoder, char_to_id, id_to_char)
        print(f"Input: '{word}' → Output: '{result}'")
    
    # Debug info
    print("\nDebug Info:")
    print(f"Vocabulary size: {len(char_to_id)}")
    print("Sample Latin mappings:")
    latin_chars = {k: char_to_id[k] for k in sorted(char_to_id) 
                   if k.isascii() and len(k) == 1 and not k.startswith('<')}
    print(dict(list(latin_chars.items())[:5]))
    print("Sample Devanagari mappings:")
    devanagari_chars = {k: char_to_id[k] for k in sorted(char_to_id) 
                        if not k.isascii() and len(k) == 1 and not k.startswith('<')}
    print(dict(list(devanagari_chars.items())[:5]))

if __name__ == "__main__":
    main()
