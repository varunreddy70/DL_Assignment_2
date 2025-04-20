# GPT‑2 Lyrics Generation

Fine‑tunes the GPT‑2 language model to generate English song lyrics, using the HuggingFace Transformers library.

---

## Dependencies

- Python 3.8+  
- transformers  
- torch  
- datasets  

Install with:
bash
```
pip install transformers torch datasets
```

---

### Model Specification

- **Base model**: GPT-2 (124M parameters)

#### 🏋️ Training Configuration:
- **Batch size**: 2  
- **Epochs**: 5
- **Learning rate**: 5e-5  
- **Mixed Precision (FP16)**: Enabled  

#### 🎶 Generation Parameters:
- **Temperature**: 0.7  
- **Top-k**: 50  
- **Top-p**: 0.95  
- **Repetition penalty**: 1.2  
- **Max length**: 100 tokens  
- **Number of sequences**: 3  


##  Notebook Workflow

The notebook implements the following cells:

### 1. Setup 
   
bash
   !pip install transformers torch datasets

Installs required libraries like Transformers, Torch, and Datasets.

### 2. Load & Tokenize Data
Loads lyrics.txt, formats each lyric line with special tokens (<|startoftext|>, <|endoftext|>), and tokenizes them using a GPT-2 tokenizer.

### 3. Dataset Class
Creates a custom PyTorch dataset to handle the tokenized lyrics and prepare them for training.

### 4. Model & Tokenizer Initialization
Loads a pre-trained GPT-2 model and resizes its embeddings to include the new special tokens.

### 5. Training Setup
Defines training arguments (epochs, batch size, learning rate, logging) and uses HuggingFace's Trainer API to fine-tune the model.

### 6. Save Fine‑Tuned Model
Saves the fine-tuned model and tokenizer to the lyrics_gpt2/ directory for reuse.

### 7. Generate Lyrics
Uses a text-generation pipeline to generate new lyrics from the fine-tuned model using sampling techniques (temperature, top-k, top-p).


## Example Output

--- Generated Lyrics 1 ---
<|startoftext|>love flows like a river in the night  
echoes of our hearts calling out for light  
wandering souls find their home in song  
we dance forever, together strong<|endoftext|>

--- Generated Lyrics 2 ---
<|startoftext|>when the stars align and the moonlight gleams  
we chase our hopes and we dare to dream  
melodies rise with every breath we take  
hearts beat as one for love’s sweet sake<|endoftext|>

--- Generated Lyrics 3 ---
<|startoftext|>in the silence of the dawn we sing  
broken wings mend on hopeful wings  
rhythms guide us through the stormy seas  
lyrics bind us close in harmonies<|endoftext|>


---

### Key Points

- **Model**: GPT‑2 fine‑tuning for natural language generation  
- **Dataset**: Custom lyrics file (lyrics.txt) from kaggle 
- **Steps**: Data loading, tokenization, training, saving, generation  
- **Code Quality**: Modular notebook, clear comments, uses Trainer API  
- **Outputs**: Sample generated lyrics, saved model for reproducibility  

---

## References

- [GPT‑2 Fine‑Tuning Guide (HuggingFace)](https://towardsdatascience.com/natural-language-generation-part-2-gpt-2-and-huggingface-f3acb35bc86a)  
- [HuggingFace Transformers Documentation](https://huggingface.co/docs/transformers)
- [Lyrics dataset from kaggle](https://www.kaggle.com/datasets/paultimothymooney/poetry) 
