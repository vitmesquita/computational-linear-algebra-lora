import csv
import json
import random
import time
from datetime import datetime, timezone
from pathlib import Path

import torch
import torch.nn as nn
import numpy as np
from transformers import GPT2LMHeadModel, GPT2Tokenizer
from datasets import load_dataset
from torch.utils.data import DataLoader, Dataset
from load_dataset import WikiTextDataset


def budget_scheduler(step, b_initial, b_final, t_i, t_f, T):
    """Eq. (12) do paper AdaLoRA."""
    if step < t_i:
        return b_initial
    elif step >= T - t_f:
        return b_final
    else:
        progress = (step - t_i) / (T - t_i - t_f)
        return b_final + (b_initial - b_final) * (1 - progress) ** 3


def global_update_lambda(model, budget):
    modules = [m for m in model.modules() if isinstance(m, AdaLoRALinear)]

    all_scores = [m.calculate_ipt() for m in modules]
    global_scores = torch.cat(all_scores)
    budget = int(round(budget))
    budget = max(1, min(budget, global_scores.numel()))
    threshold = global_scores.topk(budget).values.min()

    for module, scores in zip(modules, all_scores):
        mask = scores >= threshold
        module.Lambda.data[~mask] = 0.0


class ScoreCalculator():
    def __init__(self, weight_old):
        self.old_weights = weight_old
        self.score_matrix = torch.zeros_like(weight_old)
        self.I_bar = torch.zeros_like(weight_old)
        self.U_bar = torch.zeros_like(weight_old)

    def calculate_sensibility(self, param):
        with torch.no_grad():
            grad = param.grad
            if grad is None:
                return torch.zeros_like(param)
            I = (param * grad).abs()
            return I

    def scoring(self, param, beta_1, beta_2):
        with torch.no_grad():
            I = self.calculate_sensibility(param)
            self.I_bar = beta_1*self.I_bar + (1-beta_1)*I
            self.U_bar = beta_2*self.U_bar + (1-beta_2)*(I-self.I_bar).abs()
            score_matrix = self.I_bar * self.U_bar
        return score_matrix


class AdaLoRALinear(nn.Module, ScoreCalculator):
    def __init__(self, linear: nn.Linear, rank, alpha, beta_1, beta_2):
        super().__init__()
        ScoreCalculator.__init__(self, linear.weight.data)
        self.old_weights_size = linear.weight.shape
        self.linear = linear
        self.r = rank
        self.alpha = alpha

        self.P = nn.Parameter(torch.randn(self.old_weights_size[0], self.r) / (self.old_weights_size[0] ** 0.5))
        self.Q = nn.Parameter(torch.randn(self.r, self.old_weights_size[1]) / (self.old_weights_size[1] ** 0.5))
        self.Lambda = nn.Parameter(torch.zeros(self.r))

        self.P_score_calculator = ScoreCalculator(self.P.data)
        self.Lambda_score_calculator = ScoreCalculator(self.Lambda.data)
        self.Q_score_calculator = ScoreCalculator(self.Q.data)

        self.linear.weight.requires_grad = False
        if self.linear.bias is not None:
            self.linear.bias.requires_grad = False

        self.d = self.old_weights_size[0]
        self.k = self.old_weights_size[1]
        self.beta_1 = beta_1
        self.beta_2 = beta_2
        self.ipt = 1

    def forward(self, x):
        deltaW = self.P @ (self.Lambda.unsqueeze(1) * self.Q)
        x = self.linear(x) + (self.alpha/self.r) * (x @ deltaW)
        return x

    def merge(self):
        with torch.no_grad():
            self.linear.weight.data += (self.alpha/self.r) * (self.P @ (self.Lambda.unsqueeze(1) * self.Q))

    def calculate_ipt(self):
        with torch.no_grad():
            self.P_score = self.P_score_calculator.scoring(self.P, self.beta_1, self.beta_2)
            self.Lambda_score = self.Lambda_score_calculator.scoring(self.Lambda, self.beta_1, self.beta_2)
            self.Q_score = self.Q_score_calculator.scoring(self.Q, self.beta_1, self.beta_2)

            ipt = self.Lambda_score.clone()
            ipt += (1/self.d) * (self.P_score.sum(dim=0))
            ipt += (1/self.k) * (self.Q_score.sum(dim=1))
            self.ipt = ipt
        return ipt


if __name__ == "__main__":

    seed = 42
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    # --- device: Colab ---
    if torch.cuda.is_available():
        device = torch.device("cuda")
    # --- device: Mac MPS ---
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    print(f"Using device: {device}")

    model = GPT2LMHeadModel.from_pretrained("gpt2")

    reg_weight = 0.1
    rank = 4
    alpha = rank
    beta_1 = 0.85
    beta_2 = 0.85
    step = 0
    delta_T = 10

    print('Freezing weights')
    for p in model.parameters():
        p.requires_grad = False

    print('Replacing layers')
    for block in model.transformer.h:
        block.attn.c_attn = AdaLoRALinear(block.attn.c_attn, rank=rank, alpha=alpha, beta_1=beta_1, beta_2=beta_2)
        block.mlp.c_fc = AdaLoRALinear(block.mlp.c_fc, rank=rank, alpha=alpha, beta_1=beta_1, beta_2=beta_2)
        block.mlp.c_proj = AdaLoRALinear(block.mlp.c_proj, rank=rank, alpha=alpha, beta_1=beta_1, beta_2=beta_2)

    print('Verificando sanidade...')
    n_treinaveis = sum(p.numel() for p in model.parameters() if p.requires_grad)
    n_total = sum(p.numel() for p in model.parameters())
    print(f"Parâmetros treináveis: {n_treinaveis:,} / {n_total:,} ({100*n_treinaveis/n_total:.3f}%)")

    for block in model.transformer.h:
        ada = block.attn.c_attn
        assert (ada.P @ (ada.Lambda.unsqueeze(1) * ada.Q)).abs().max().item() == 0.0, "ΔW não é zero no início"

    model = model.to(device)
    for module in model.modules():
        if isinstance(module, AdaLoRALinear):
            for calc in [module.P_score_calculator, module.Lambda_score_calculator, module.Q_score_calculator]:
                calc.I_bar = calc.I_bar.to(device)
                calc.U_bar = calc.U_bar.to(device)

    max_lenght = 128
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

    train_dataset = WikiTextDataset(train_texts, tokenizer, max_length=max_lenght)
    test_dataset = WikiTextDataset(test_texts, tokenizer, max_length=max_lenght)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    T = len(train_loader) * num_epochs
    t_i = max(1, T // 5)
    t_f = max(1, T // 5)
    n_matrices = sum(1 for m in model.modules() if isinstance(m, AdaLoRALinear))
    b_final = rank * n_matrices // 2
    b_initial = int(1.5 * b_final)

    print(f"T={T} | t_i={t_i} | t_f={t_f} | b_initial={b_initial} | b_final={b_final}")

    if Path("/content").exists():
        output_root = Path("/content/drive/MyDrive/cla_lora_runs")
    else:
        output_root = Path("outputs/cla_lora_runs")
    run_name = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    run_dir = output_root / "adalora" / run_name
    run_dir.mkdir(parents=True, exist_ok=True)

    config = {
        "method": "adalora",
        "model_name": "gpt2",
        "dataset_name": "Salesforce/wikitext",
        "dataset_config": "wikitext-2-raw-v1",
        "max_length": max_lenght,
        "batch_size": batch_size,
        "num_workers": num_workers,
        "num_epochs": num_epochs,
        "rank": rank,
        "alpha": alpha,
        "reg_weight": reg_weight,
        "beta_1": beta_1,
        "beta_2": beta_2,
        "delta_T": delta_T,
        "seed": seed,
        "device": str(device),
        "output_path": str(run_dir),
        "started_at": datetime.now(timezone.utc).isoformat(),
    }
    with (run_dir / "config.json").open("w", encoding="utf-8") as f:
        json.dump(config, f, indent=2, ensure_ascii=False)

    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
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
            output = model(input_ids=input_ids, labels=input_ids)
            loss = output.loss

            reg = 0
            for module in model.modules():
                if isinstance(module, AdaLoRALinear):
                    reg += torch.norm(module.P.T @ module.P - torch.eye(module.r, device=module.P.device), p='fro')**2
                    reg += torch.norm(module.Q @ module.Q.T - torch.eye(module.r, device=module.P.device), p='fro')**2

            loss = loss + reg_weight * reg
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

            if t_i <= step < T - t_f and step % delta_T == 0:
                current_budget = budget_scheduler(step, b_initial, b_final, t_i, t_f, T)
                global_update_lambda(model, current_budget)

            step += 1

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
    for idx, block in enumerate(model.transformer.h):
        target_layers = {
            f"transformer.h.{idx}.attn.c_attn": block.attn.c_attn,
            f"transformer.h.{idx}.mlp.c_fc": block.mlp.c_fc,
            f"transformer.h.{idx}.mlp.c_proj": block.mlp.c_proj,
        }
        for layer_name, module in target_layers.items():
            delta_weight = ((alpha / rank) * (module.P @ (module.Lambda.unsqueeze(1) * module.Q))).detach().cpu().float().clone()
            delta_map[layer_name] = delta_weight

            svdvals = torch.linalg.svdvals(delta_weight)
            svd_map[layer_name] = svdvals

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
            effective_rank = int((module.Lambda.detach().abs() > 1e-8).sum().item())

            layer_stats[layer_name] = {
                "shape": list(delta_weight.shape),
                "fro_norm": fro_norm,
                "spectral_norm": spectral_norm,
                "stable_rank": stable_rank,
                "energy_90_rank": energy_90_rank,
                "energy_95_rank": energy_95_rank,
                "effective_rank": effective_rank,
            }

    torch.save(delta_map, run_dir / "target_deltas.pt")
    torch.save(svd_map, run_dir / "target_svdvals.pt")
    with (run_dir / "layer_stats.json").open("w", encoding="utf-8") as f:
        json.dump(layer_stats, f, indent=2, ensure_ascii=False)

    total_time = time.time() - total_start
    summary = {
        "method": "adalora",
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
