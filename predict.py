import numpy as np
from tensorflow.keras.models import load_model

class Transliterator:
    def __init__(self, model_path):
        self.model = load_model(model_path)
        self.char_to_id = np.load("char_to_id.npy", allow_pickle=True).item()
        self.id_to_char = {v: k for k, v in self.char_to_id.items()}  # Reverse char_to_id

    def encode_input(self, text):
        # Convert text to a list of character indices
        return [self.char_to_id.get(c, self.char_to_id['<unk>']) for c in text.lower()]
    
def decode_sequence(self, input_seq):
    # Get initial states from the encoder model
    states_value = self.model.layers[3](np.array([input_seq]))  # Assuming LSTM at index 3

    # Generate empty target sequence of shape (1, 1) - Start token
    target_seq = np.zeros((1, 1))  
    target_seq[0, 0] = self.char_to_id['<start>']  # Start token

    decoded_sentence = []
    
    # Loop to generate sequence
    for _ in range(20):  # Maximum output length
        # Reshape target_seq to (batch_size=1, timesteps=1, embedding_dim)
        target_seq = np.reshape(target_seq, (1, 1, 1))  # Ensure the shape is (1, 1, 1) if you're working with token IDs
        
        # Feed the target sequence to the LSTM layer, along with initial states
        output_tokens, h, c = self.model.layers[5](target_seq, initial_state=states_value)  # Assuming LSTM at index 5
        
        # Sample the token with the highest probability
        sampled_token_id = np.argmax(output_tokens[0, -1, :])
        decoded_sentence.append(self.id_to_char[sampled_token_id])
        
        # Exit condition if we hit the <end> token
        if sampled_token_id == self.char_to_id['<end>']:
            break
        
        # Update the target sequence to the sampled token (for the next step in decoding)
        target_seq = np.array([[sampled_token_id]])
        
        # Update the states for the next time step
        states_value = [h, c]

    return ''.join(decoded_sentence[:-1])  # Remove the <end> token at the end of the sentence


if __name__ == "__main__":
    transliterator = Transliterator("transliteration_model.h5")
    test_words = ["namaste", "ghar", "dhanyavad"]
    
    for word in test_words:
        input_seq = transliterator.encode_input(word)
        output = transliterator.decode_sequence(input_seq)
        print(f"Input: {word} → Output: {output}")
