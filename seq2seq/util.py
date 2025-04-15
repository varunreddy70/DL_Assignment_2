import numpy as np

def batch_validate(encoder_model, decoder_model, val_inputs, val_targets, batch_size=64):
    correct = 0
    for i in range(0, len(val_inputs), batch_size):
        batch_inputs = val_inputs[i:i+batch_size]
        batch_targets = val_targets[i:i+batch_size]
        
        states_value = encoder_model.predict(batch_inputs)
        target_seq = np.zeros((len(batch_inputs), 1))
        target_seq[:, 0] = char_to_id['<start>']
        
        for t in range(max_decoder_seq_length):
            output_tokens, h, c = decoder_model.predict(
                [target_seq] + states_value
            )
            sampled_indices = np.argmax(output_tokens, axis=-1)
            
            for j in range(len(batch_inputs)):
                if t < len(batch_targets[j]) and sampled_indices[j, 0] == batch_targets[j][t]:
                    if t == len(batch_targets[j]) - 1:
                        correct += 1
                
            target_seq = sampled_indices
            states_value = [h, c]
    
    return correct / len(val_inputs)