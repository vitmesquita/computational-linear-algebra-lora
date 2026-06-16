import csv
import json
import random
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from transformers import GPT2LMHeadModel, GPT2Tokenizer
from datasets import load_dataset
from torch.utils.data import DataLoader
from load_dataset import WikiTextDataset


if __name__ == "__main__":

    seed = 42
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    print(f"Using device: {device}")

    model = GPT2LMHeadModel.from_pretrained("gpt2")

    target_module_names = []
    initial_weights = {}
    for idx, _ in enumerate(model.transformer.h):
        target_module_names.append(f"transformer.h.{idx}.attn.c_attn")
        target_module_names.append(f"transformer.h.{idx}.mlp.c_fc")
        target_module_names.append(f"transformer.h.{idx}.mlp.c_proj")
    for module_name in target_module_names:
        initial_weights[module_name] = model.get_submodule(module_name).weight.detach().cpu().float().clone()

    n_treinaveis = sum(p.numel() for p in model.parameters() if p.requires_grad)
    n_total = sum(p.numel() for p in model.parameters())
    print(f"Parâmetros treináveis: {n_treinaveis:,} / {n_total:,} ({100*n_treinaveis/n_total:.3f}%)")

    model = model.to(device)

    max_length = 128
    batch_size = 1
    num_workers = 2
    num_epochs = 3
    max_train_examples = None
    max_test_examples = None

    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    tokenizer.pad_token = tokenizer.eos_token

    raw = load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1")

    train_texts = raw["train"]["text"]
    test_texts = raw["test"]["text"]
    if max_train_examples is not None:
        train_texts = train_texts[:max_train_examples]
    if max_test_examples is not None:
        test_texts = test_texts[:max_test_examples]

    train_dataset = WikiTextDataset(train_texts, tokenizer, max_length=max_length)
    test_dataset = WikiTextDataset(test_texts, tokenizer, max_length=max_length)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    if Path("/content").exists():
        output_root = Path("/content/drive/MyDrive/cla_lora_runs")
    else:
        output_root = Path("outputs/cla_lora_runs")
    run_name = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    run_dir = output_root / "full_finetune" / run_name
    run_dir.mkdir(parents=True, exist_ok=True)

    config = {
        "method": "full_finetune",
        "model_name": "gpt2",
        "dataset_name": "Salesforce/wikitext",
        "dataset_config": "wikitext-2-raw-v1",
        "max_length": max_length,
        "batch_size": batch_size,
        "num_workers": num_workers,
        "num_epochs": num_epochs,
        "seed": seed,
        "device": str(device),
        "output_path": str(run_dir),
        "started_at": datetime.now(timezone.utc).isoformat(),
    }
    with (run_dir / "config.json").open("w", encoding="utf-8") as f:
        json.dump(config, f, indent=2, ensure_ascii=False)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=2e-4,
        weight_decay=0.01
    )

    history = []
    total_start = time.time()

    for epoch in range(num_epochs):
        epoch_start = time.time()
        total_loss = 0.0
        model.train()

        for batch in train_loader:
            input_ids = batch.to(device)
            optimizer.zero_grad()
            outputs = model(input_ids=input_ids, labels=input_ids)
            loss = outputs.loss
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        avg_loss = total_loss / len(train_loader)

        model.eval()
        total_test_loss = 0.0
        with torch.no_grad():
            for batch in test_loader:
                input_ids = batch.to(device)
                outputs = model(input_ids=input_ids, labels=input_ids)
                total_test_loss += outputs.loss.item()
        avg_test_loss = total_test_loss / len(test_loader)

        epoch_time = time.time() - epoch_start
        cumulative_time = time.time() - total_start
        history.append(
            {
                "epoch": epoch,
                "train_loss": avg_loss,
                "test_loss": avg_test_loss,
                "epoch_time_sec": epoch_time,
                "cumulative_time_sec": cumulative_time,
            }
        )

        with (run_dir / "train_history.csv").open("w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(
                f,
                fieldnames=[
                    "epoch",
                    "train_loss",
                    "test_loss",
                    "epoch_time_sec",
                    "cumulative_time_sec",
                ],
            )
            writer.writeheader()
            writer.writerows(history)

        torch.save(
            {
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "history": history,
            },
            run_dir / "latest_checkpoint.pt",
        )

        print(f"Epoch {epoch} | Loss médio treino: {avg_loss:.4f} | Loss médio teste: {avg_test_loss:.4f}")

    torch.save(
        {
            "epoch": num_epochs - 1,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "history": history,
        },
        run_dir / "final_checkpoint.pt",
    )

    delta_map = {}
    svd_map = {}
    layer_stats = {}
    for module_name in target_module_names:
        final_weight = model.get_submodule(module_name).weight.detach().cpu().float().clone()
        delta_weight = final_weight - initial_weights[module_name]
        delta_map[module_name] = delta_weight

        svdvals = torch.linalg.svdvals(delta_weight)
        svd_map[module_name] = svdvals

        energy = svdvals.pow(2)
        total_energy = energy.sum().item()
        energy_90_rank = 0
        energy_95_rank = 0
        if total_energy > 0:
            cumulative = torch.cumsum(energy, dim=0) / total_energy
            energy_90_rank = int((cumulative >= 0.90).nonzero(as_tuple=False)[0].item() + 1)
            energy_95_rank = int((cumulative >= 0.95).nonzero(as_tuple=False)[0].item() + 1)

        fro_norm = torch.linalg.matrix_norm(delta_weight, ord="fro").item()
        spectral_norm = svdvals[0].item() if svdvals.numel() > 0 else 0.0
        stable_rank = (fro_norm ** 2) / (spectral_norm ** 2) if spectral_norm > 0 else 0.0

        layer_stats[module_name] = {
            "shape": list(delta_weight.shape),
            "fro_norm": fro_norm,
            "spectral_norm": spectral_norm,
            "stable_rank": stable_rank,
            "energy_90_rank": energy_90_rank,
            "energy_95_rank": energy_95_rank,
        }

    torch.save(delta_map, run_dir / "target_deltas.pt")
    torch.save(svd_map, run_dir / "target_svdvals.pt")
    with (run_dir / "layer_stats.json").open("w", encoding="utf-8") as f:
        json.dump(layer_stats, f, indent=2, ensure_ascii=False)

    total_time = time.time() - total_start
    summary = {
        "method": "full_finetune",
        "total_params": n_total,
        "trainable_params": n_treinaveis,
        "trainable_pct": 100 * n_treinaveis / n_total,
        "final_train_loss": history[-1]["train_loss"],
        "final_test_loss": history[-1]["test_loss"],
        "total_time_sec": total_time,
        "run_path": str(run_dir),
        "completed_at": datetime.now(timezone.utc).isoformat(),
    }
    with (run_dir / "summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print(f"Loss final de treino: {history[-1]['train_loss']:.4f}")
    print(f"Loss final de teste: {history[-1]['test_loss']:.4f}")
