from torch.utils.data import Dataset


class WikiTextDataset(Dataset):
    def __init__(self, texts, tokenizer, max_length=128, return_dict=False):
        self.examples = []
        self.return_dict = return_dict
        for text in texts:
            if len(text.strip()) > 0:
                tokens = tokenizer(
                    text,
                    truncation=True,
                    max_length=max_length,
                    padding="max_length",
                    return_attention_mask=True,
                    return_tensors="pt",
                )
                input_ids = tokens["input_ids"].squeeze(0)
                attention_mask = tokens["attention_mask"].squeeze(0)

                if self.return_dict:
                    labels = input_ids.clone()
                    labels[attention_mask == 0] = -100
                    self.examples.append(
                        {
                            "input_ids": input_ids,
                            "attention_mask": attention_mask,
                            "labels": labels,
                        }
                    )
                else:
                    self.examples.append(input_ids)

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        return self.examples[idx]
