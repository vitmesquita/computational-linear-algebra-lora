from torch.utils.data import Dataset


class WikiTextDataset(Dataset):
    def __init__(self, texts, tokenizer, max_length=128):
        self.examples = []
        for text in texts:
            if len(text.strip()) > 0:
                tokens = tokenizer(
                    text,
                    truncation=True,
                    max_length=max_length,
                    padding="max_length",
                    return_tensors="pt",
                )
                self.examples.append(tokens["input_ids"].squeeze(0))

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        return self.examples[idx]
