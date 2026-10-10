import torch
from torch.utils.data import Dataset
from datasets import load_dataset
from transformers import GPT2TokenizerFast

print("Cargando dataset de Wikipedia...")
all_wiki = load_dataset("wikimedia/wikipedia", "20231101.es", split="train")

n_articles = 150
text = ' '.join((all_wiki['text'][:n_articles]))

tokenizer = GPT2TokenizerFast.from_pretrained('gpt2')
tokenizer.pad_token = tokenizer.eos_token
tokens = tokenizer.encode(text)
tokens_tensor = torch.tensor(tokens, dtype=torch.long)

class Wikipedia(Dataset):
    def __init__(self, text, max_length=128):
        self.text = text
        self.max_length = max_length

    def __getitem__(self, idx):
        x = self.text[idx:idx + self.max_length]
        y = self.text[idx + 1:idx + self.max_length + 1]

        if not isinstance(x, torch.Tensor):
            x = torch.tensor(x, dtype=torch.long)
        if not isinstance(y, torch.Tensor):
            y = torch.tensor(y, dtype=torch.long)

        return x, y

    def __len__(self):
        return len(self.text) - self.max_length
