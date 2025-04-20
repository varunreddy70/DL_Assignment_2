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
pip install transformers torch datasets


---

## Repository Structure

.
├── GPT2_Lyrics_Generation.ipynb   # Jupyter notebook with all steps
├── lyrics.txt                     # Raw lyrics, one line per song lyric
├── results/                       # Checkpoints & logs saved by Trainer
└── lyrics_gpt2/                   # Fine‑tuned model & tokenizer
    ├── config.json
    ├── pytorch_model.bin
    ├── merges.txt
    ├── special_tokens_map.json
    ├── tokenizer_config.json
    └── vocab.json


---

##  Notebook Workflow

The notebook implements the following cells:

1. **Setup**  
   
bash
   !pip install transformers torch datasets

2. **Load & Tokenize Data**  
   
python
   from transformers import GPT2Tokenizer
   with open('lyrics.txt', 'r', encoding='utf-8') as f:
       lyrics = [line.strip() for line in f if line.strip()]

   tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
   tokenizer.add_special_tokens({
       'bos_token': '<|startoftext|>',
       'eos_token': '<|endoftext|>',
       'pad_token': '<|pad|>'
   })

   formatted = [f"<|startoftext|>{l}<|endoftext|>" for l in lyrics]
   tokenized_lyrics = tokenizer(
       formatted,
       truncation=True,
       max_length=512,
       padding="max_length"
   )

3. **Dataset Class**  
   
python
   import torch

   class LyricsDataset(torch.utils.data.Dataset):
       def __init__(self, encodings):
           self.encodings = encodings

       def __getitem__(self, idx):
           return {
               'input_ids':    torch.tensor(self.encodings['input_ids'][idx]),
               'attention_mask': torch.tensor(self.encodings['attention_mask'][idx]),
               'labels':       torch.tensor(self.encodings['input_ids'][idx])
           }

       def __len__(self):
           return len(self.encodings['input_ids'])

   dataset = LyricsDataset(tokenized_lyrics)

4. **Model & Tokenizer Initialization**  
   
python
   from transformers import GPT2LMHeadModel

   model = GPT2LMHeadModel.from_pretrained("gpt2")
   model.resize_token_embeddings(len(tokenizer))
   model.config.pad_token_id = tokenizer.pad_token_id

5. **Training Setup**  
   
python
   from transformers import TrainingArguments, Trainer

   training_args = TrainingArguments(
       output_dir="./results",
       num_train_epochs=3,
       per_device_train_batch_size=2,
       save_steps=10000,
       save_total_limit=2,
       learning_rate=5e-5,
       weight_decay=0.01,
       fp16=True,
       logging_steps=100,
       report_to="none"
   )

   trainer = Trainer(
       model=model,
       args=training_args,
       train_dataset=dataset,
       data_collator=lambda data: {
           'input_ids':     torch.stack([item['input_ids']    for item in data]),
           'attention_mask':torch.stack([item['attention_mask']for item in data]),
           'labels':        torch.stack([item['labels']       for item in data])
       }
   )

   trainer.train()

6. **Save Fine‑Tuned Model**  
   
python
   model.save_pretrained("./lyrics_gpt2")
   tokenizer.save_pretrained("./lyrics_gpt2")

7. **Generate Lyrics**  
   
python
   from transformers import pipeline

   lyrics_generator = pipeline(
       "text-generation",
       model="./lyrics_gpt2",
       tokenizer="./lyrics_gpt2",
       device=0  # or device=-1 for CPU
   )

   generated = lyrics_generator(
       "<|startoftext|>",
       max_length=100,
       num_return_sequences=3,
       temperature=0.7,
       top_k=50,
       top_p=0.95,
       repetition_penalty=1.2
   )

   for i, out in enumerate(generated, 1):
       print(f"--- Generated Lyrics {i} ---\n{out['generated_text']}\n")


---

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
