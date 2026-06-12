import torch
import torch.nn as nn
import numpy as np
from transformers import GPT2LMHeadModel, GPT2Tokenizer
from datasets import load_dataset
from torch.utils.data import DataLoader, Dataset
from load_dataset import WikiTextDataset


class LoRALinear(nn.Module):
    def __init__(self,linear: nn.Linear, rank, alpha):
        super().__init__()
        self.linear = linear   
        self.old_weights_size = linear.weight.shape
        self.r = rank
        self.alpha = alpha

        self.A = nn.Parameter(torch.randn(self.r, self.old_weights_size[1]))
        self.B = nn.Parameter(torch.zeros(self.old_weights_size[0], self.r))

        self.linear.weight.requires_grad = False
        if self.linear.bias is not None:
            self.linear.bias.requires_grad = False
    
    def forward(self,x):
        deltaW = self.B @ self.A
        x = self.linear(x) + (self.alpha/self.r)* (x @ deltaW)
        return x
    
    def merge(self):
        with torch.no_grad():
            self.linear.weight += (self.alpha / self.r) * self.B @ self.A


if __name__ == "__main__":

    # --- device: Colab ---
    #device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # --- device: Mac MPS ---
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Using device: {device}")

    model = GPT2LMHeadModel.from_pretrained("gpt2")

    rank=4
    alpha = rank
    print('Freezing weights')
    for p in model.parameters():
        p.requires_grad = False

    print('Replacing layers')
    for block in model.transformer.h:
        block.attn.c_attn = LoRALinear(block.attn.c_attn, rank=rank, alpha=alpha)

    # Adicionar após o loop de substituição:
    print('Verificando sanidade...')

    # ΔW deve ser zero no início (B=0)
    for block in model.transformer.h:
        lora = block.attn.c_attn
        assert (lora.B @ lora.A).abs().max().item() == 0.0, "ΔW não é zero no início"

    # Apenas A e B devem ter gradiente
    for name, p in model.named_parameters():
        if p.requires_grad:
            assert 'lora' not in name.lower() or ('A' in name or 'B' in name), \
                f"Parâmetro inesperado com grad: {name}"

    n_treinaveis = sum(p.numel() for p in model.parameters() if p.requires_grad)
    n_total = sum(p.numel() for p in model.parameters())
    print(f"Parâmetros treináveis: {n_treinaveis:,} / {n_total:,} ({100*n_treinaveis/n_total:.3f}%)")


    model = model.to(device)


    # Mac
    max_lenght = 128
    batch_size = 1
    num_workers = 0
    # Colab 
    #max_lenght = 256
    #batch_size = 4
    #num_workers = 2
    
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    tokenizer.pad_token = tokenizer.eos_token

    raw = load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1")

    # primeira rodada: só 200 exemplos para verificar
    train_texts = raw["train"]["text"][:200]

    train_dataset = WikiTextDataset(train_texts, tokenizer, max_length=max_lenght)
    train_loader  = DataLoader(train_dataset, batch_size=batch_size,
                           shuffle=True, num_workers=num_workers)

    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=2e-4,
        weight_decay=0.01
    )

    for epoch in range(10):
        total_loss = 0.0
        for batch in train_loader:
            input_ids = batch.to(device)
            optimizer.zero_grad()
            outputs = model(input_ids=input_ids, labels=input_ids)
            loss = outputs.loss
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        avg_loss = total_loss / len(train_loader)
        print(f"Epoch {epoch} | Loss médio: {avg_loss:.4f}")
            # Mac: checkpoint comentado (path /content é Colab)
            # Colab: descomentar
            # torch.save(
            #     {k: v for k, v in model.state_dict().items() if 'lora' in k.lower()},
            #     f"/content/lora_epoch_{epoch}.pt"
            # )
    print(f"Loss final: {loss.item():.4f}")