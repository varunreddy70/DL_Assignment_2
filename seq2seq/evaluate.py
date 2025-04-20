import numpy as np
from tensorflow.keras.models import load_model

def evaluate_model(model_path, test_data_path):
    # Load model and vocab
    model = load_model(model_path)
    char_to_id = np.load("char_to_id.npy", allow_pickle=True).item()
    
    # Load and preprocess test data
    import pandas as pd
    from preprocess import preprocess
    
    test_data = pd.read_csv(test_data_path, sep="\t", header=None, 
                          names=["devanagari", "latin", "count"])
    test_data = test_data.dropna()
    
    X_test, y_test = preprocess(test_data, char_to_id)
    
    # Evaluate
    results = model.evaluate(
        [X_test, y_test[:, :-1]],
        y_test[:, 1:],
        batch_size=64,
        verbose=1
    )
    
    print("\n=== Test Set Evaluation ===")
    print(f"Test Loss: {results[0]:.4f}")
    print(f"Test Accuracy: {results[1]:.4f}")
    
    return results

if __name__ == "__main__":
    evaluate_model("transliteration_model.h5", "hi.translit.sampled.test.tsv")