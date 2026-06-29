import csv
import json
import random
import shutil
import subprocess
import threading
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

try:
    import psutil
except ImportError:
    psutil = None


def safe_mean(values):
    return sum(values) / len(values) if values else None


def safe_max(values):
    return max(values) if values else None


def read_nvidia_smi_stats(device_index):
    if device_index is None or shutil.which("nvidia-smi") is None:
        return {}
    try:
        result = subprocess.run(
            [
                "nvidia-smi",
                "-i",
                str(device_index),
                "--query-gpu=utilization.gpu,memory.used,memory.total",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True,
            text=True,
            check=True,
        )
        line = result.stdout.strip().splitlines()[0]
        gpu_util_percent, gpu_mem_used_mb, gpu_mem_total_mb = [value.strip() for value in line.split(",")]
        return {
            "gpu_util_percent": float(gpu_util_percent),
            "gpu_mem_used_mb": float(gpu_mem_used_mb),
            "gpu_mem_total_mb": float(gpu_mem_total_mb),
        }
    except (IndexError, OSError, subprocess.SubprocessError, ValueError):
        return {}


def build_hardware_info(device, sample_interval_sec):
    hardware_info = {
        "device": str(device),
        "torch_version": torch.__version__,
        "cuda_version": torch.version.cuda,
        "hardware_sample_interval_sec": sample_interval_sec,
        "psutil_available": psutil is not None,
        "nvidia_smi_available": shutil.which("nvidia-smi") is not None,
        "cpu_logical_count": None,
        "cpu_physical_count": None,
        "system_ram_total_gb": None,
        "gpu_name": None,
        "gpu_total_memory_mb": None,
        "gpu_current_util_percent": None,
        "gpu_current_memory_used_mb": None,
        "cuda_device_index": None,
    }
    if psutil is not None:
        hardware_info["cpu_logical_count"] = psutil.cpu_count(logical=True)
        hardware_info["cpu_physical_count"] = psutil.cpu_count(logical=False)
        hardware_info["system_ram_total_gb"] = psutil.virtual_memory().total / (1024 ** 3)
    if device.type == "cuda":
        device_index = torch.cuda.current_device()
        props = torch.cuda.get_device_properties(device_index)
        hardware_info["cuda_device_index"] = device_index
        hardware_info["gpu_name"] = props.name
        hardware_info["gpu_total_memory_mb"] = props.total_memory / (1024 ** 2)
        smi_stats = read_nvidia_smi_stats(device_index)
        hardware_info["gpu_current_util_percent"] = smi_stats.get("gpu_util_percent")
        hardware_info["gpu_current_memory_used_mb"] = smi_stats.get("gpu_mem_used_mb")
    return hardware_info


def capture_hardware_sample(process, device, device_index):
    sample = {
        "system_cpu_percent": None,
        "process_ram_rss_gb": None,
        "system_ram_used_gb": None,
        "gpu_util_percent": None,
        "gpu_mem_used_mb": None,
    }
    if psutil is not None:
        try:
            sample["system_cpu_percent"] = psutil.cpu_percent(interval=None)
        except Exception:
            pass
        try:
            sample["system_ram_used_gb"] = psutil.virtual_memory().used / (1024 ** 3)
        except Exception:
            pass
    if process is not None:
        try:
            sample["process_ram_rss_gb"] = process.memory_info().rss / (1024 ** 3)
        except Exception:
            pass
    if device.type == "cuda":
        smi_stats = read_nvidia_smi_stats(device_index)
        sample["gpu_util_percent"] = smi_stats.get("gpu_util_percent")
        sample["gpu_mem_used_mb"] = smi_stats.get("gpu_mem_used_mb")
    return sample


def start_hardware_monitor(device, sample_interval_sec):
    stop_event = threading.Event()
    samples = []
    process = psutil.Process() if psutil is not None else None
    device_index = torch.cuda.current_device() if device.type == "cuda" else None

    if psutil is not None:
        try:
            psutil.cpu_percent(interval=None)
        except Exception:
            pass

    def worker():
        while not stop_event.is_set():
            samples.append(capture_hardware_sample(process, device, device_index))
            stop_event.wait(sample_interval_sec)
        samples.append(capture_hardware_sample(process, device, device_index))

    thread = threading.Thread(target=worker, daemon=True)
    thread.start()
    return stop_event, thread, samples


def summarize_hardware_samples(samples, device, process_cpu_time_epoch_sec):
    cpu_values = [sample["system_cpu_percent"] for sample in samples if sample["system_cpu_percent"] is not None]
    process_ram_values = [sample["process_ram_rss_gb"] for sample in samples if sample["process_ram_rss_gb"] is not None]
    system_ram_values = [sample["system_ram_used_gb"] for sample in samples if sample["system_ram_used_gb"] is not None]
    gpu_util_values = [sample["gpu_util_percent"] for sample in samples if sample["gpu_util_percent"] is not None]
    gpu_mem_values = [sample["gpu_mem_used_mb"] for sample in samples if sample["gpu_mem_used_mb"] is not None]

    hardware_metrics = {
        "hardware_sample_count": len(samples),
        "process_cpu_time_epoch_sec": process_cpu_time_epoch_sec,
        "system_cpu_percent_mean": safe_mean(cpu_values),
        "system_cpu_percent_max": safe_max(cpu_values),
        "process_ram_rss_gb_mean": safe_mean(process_ram_values),
        "process_ram_rss_gb_max": safe_max(process_ram_values),
        "system_ram_used_gb_mean": safe_mean(system_ram_values),
        "system_ram_used_gb_max": safe_max(system_ram_values),
        "gpu_util_percent_mean": safe_mean(gpu_util_values),
        "gpu_util_percent_max": safe_max(gpu_util_values),
        "gpu_mem_used_mb_mean": safe_mean(gpu_mem_values),
        "gpu_mem_used_mb_max": safe_max(gpu_mem_values),
        "gpu_peak_allocated_mb": None,
        "gpu_peak_reserved_mb": None,
        "gpu_allocated_end_mb": None,
        "gpu_reserved_end_mb": None,
    }
    if device.type == "cuda":
        device_index = torch.cuda.current_device()
        hardware_metrics["gpu_peak_allocated_mb"] = torch.cuda.max_memory_allocated(device_index) / (1024 ** 2)
        hardware_metrics["gpu_peak_reserved_mb"] = torch.cuda.max_memory_reserved(device_index) / (1024 ** 2)
        hardware_metrics["gpu_allocated_end_mb"] = torch.cuda.memory_allocated(device_index) / (1024 ** 2)
        hardware_metrics["gpu_reserved_end_mb"] = torch.cuda.memory_reserved(device_index) / (1024 ** 2)
    return hardware_metrics


def build_hardware_summary(hardware_history):
    return {
        "hardware_sample_count_total": sum(row["hardware_sample_count"] for row in hardware_history),
        "peak_process_ram_rss_gb": safe_max([row["process_ram_rss_gb_max"] for row in hardware_history if row["process_ram_rss_gb_max"] is not None]),
        "peak_system_ram_used_gb": safe_max([row["system_ram_used_gb_max"] for row in hardware_history if row["system_ram_used_gb_max"] is not None]),
        "peak_gpu_util_percent": safe_max([row["gpu_util_percent_max"] for row in hardware_history if row["gpu_util_percent_max"] is not None]),
        "peak_gpu_mem_used_mb": safe_max([row["gpu_mem_used_mb_max"] for row in hardware_history if row["gpu_mem_used_mb_max"] is not None]),
        "peak_gpu_allocated_mb": safe_max([row["gpu_peak_allocated_mb"] for row in hardware_history if row["gpu_peak_allocated_mb"] is not None]),
        "peak_gpu_reserved_mb": safe_max([row["gpu_peak_reserved_mb"] for row in hardware_history if row["gpu_peak_reserved_mb"] is not None]),
        "mean_system_cpu_percent": safe_mean([row["system_cpu_percent_mean"] for row in hardware_history if row["system_cpu_percent_mean"] is not None]),
        "mean_gpu_util_percent": safe_mean([row["gpu_util_percent_mean"] for row in hardware_history if row["gpu_util_percent_mean"] is not None]),
    }


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
        base = self.linear(x)
        lora = (x @ self.B) @ self.A
        return base + (self.alpha / self.r) * lora

    def merge(self):
        with torch.no_grad():
            self.linear.weight += (self.alpha / self.r) * self.B @ self.A


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

    if device.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        if hasattr(torch, "set_float32_matmul_precision"):
            torch.set_float32_matmul_precision("high")

    model = GPT2LMHeadModel.from_pretrained("gpt2")

    rank = 4
    alpha = rank
    print('Freezing weights')
    for p in model.parameters():
        p.requires_grad = False

    print('Replacing layers')
    for block in model.transformer.h:
        block.attn.c_attn = LoRALinear(block.attn.c_attn, rank=rank, alpha=alpha)
        block.attn.c_proj = LoRALinear(block.attn.c_proj, rank=rank, alpha=alpha)
        block.mlp.c_fc = LoRALinear(block.mlp.c_fc, rank=rank, alpha=alpha)
        block.mlp.c_proj = LoRALinear(block.mlp.c_proj, rank=rank, alpha=alpha)

    print('Verificando sanidade...')

    for block in model.transformer.h:
        lora = block.attn.c_attn
        assert (lora.B @ lora.A).abs().max().item() == 0.0, "ΔW não é zero no início"

    for name, p in model.named_parameters():
        if p.requires_grad:
            assert 'lora' not in name.lower() or ('A' in name or 'B' in name), \
                f"Parâmetro inesperado com grad: {name}"

    n_treinaveis = sum(p.numel() for p in model.parameters() if p.requires_grad)
    n_total = sum(p.numel() for p in model.parameters())
    print(f"Parâmetros treináveis: {n_treinaveis:,} / {n_total:,} ({100*n_treinaveis/n_total:.3f}%)")

    model = model.to(device)

    experiment_profile = "legacy_controlled_v1"
    comparison_suite_id = "legacy_controlled_v1"
    comparison_target = "controlled_method_comparison"
    loss_protocol = "legacy"
    display_name = "lora"

    max_lenght = 128
    batch_size = 32
    effective_batch_size = batch_size
    learning_rate = 5e-5
    weight_decay = 0.01
    max_grad_norm = 1.0
    num_workers = 2
    pin_memory = device.type == "cuda"
    persistent_workers = num_workers > 0
    num_epochs = 5
    hardware_sample_interval_sec = 1.0
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

    train_dataset = WikiTextDataset(train_texts, tokenizer, max_length=max_lenght, return_dict=False)
    test_dataset = WikiTextDataset(test_texts, tokenizer, max_length=max_lenght, return_dict=False)
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=persistent_workers,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=persistent_workers,
    )
    optimizer_steps_per_epoch_planned = len(train_loader)
    optimizer_steps_total_planned = optimizer_steps_per_epoch_planned * num_epochs

    if Path("/content").exists():
        output_root = Path("/content/drive/MyDrive/cla_lora_runs")
    else:
        output_root = Path("outputs/cla_lora_runs")
    run_name = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    run_dir = output_root / "lora" / run_name
    run_dir.mkdir(parents=True, exist_ok=True)

    config = {
        "method": "lora",
        "display_name": display_name,
        "experiment_profile": experiment_profile,
        "comparison_suite_id": comparison_suite_id,
        "comparison_target": comparison_target,
        "loss_protocol": loss_protocol,
        "model_name": "gpt2",
        "dataset_name": "Salesforce/wikitext",
        "dataset_config": "wikitext-2-raw-v1",
        "max_length": max_lenght,
        "batch_size": batch_size,
        "effective_batch_size": effective_batch_size,
        "optimizer_name": "AdamW",
        "learning_rate": learning_rate,
        "weight_decay": weight_decay,
        "max_grad_norm": max_grad_norm,
        "num_workers": num_workers,
        "pin_memory": pin_memory,
        "persistent_workers": persistent_workers,
        "num_epochs": num_epochs,
        "hardware_sample_interval_sec": hardware_sample_interval_sec,
        "rank": rank,
        "alpha": alpha,
        "seed": seed,
        "optimizer_steps_per_epoch_planned": optimizer_steps_per_epoch_planned,
        "optimizer_steps_total_planned": optimizer_steps_total_planned,
        "device": str(device),
        "output_path": str(run_dir),
        "started_at": datetime.now(timezone.utc).isoformat(),
    }
    with (run_dir / "config.json").open("w", encoding="utf-8") as f:
        json.dump(config, f, indent=2, ensure_ascii=False)

    # --- hardware metrics: static runtime information ---
    hardware_info = build_hardware_info(device, hardware_sample_interval_sec)
    with (run_dir / "hardware_info.json").open("w", encoding="utf-8") as f:
        json.dump(hardware_info, f, indent=2, ensure_ascii=False)

    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=learning_rate,
        weight_decay=weight_decay
    )

    # --- model metrics: training and evaluation history ---
    history = []
    total_start = time.time()

    # --- hardware metrics: per-phase and per-epoch history ---
    hardware_history = []
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(torch.cuda.current_device())
    pretrain_cpu_start = time.process_time()
    pretrain_stop_event, pretrain_thread, pretrain_samples = start_hardware_monitor(
        device,
        hardware_sample_interval_sec,
    )
    total_test_loss = 0.0
    model.eval()
    with torch.no_grad():
        for batch in test_loader:
            input_ids = batch.to(device, non_blocking=pin_memory)
            outputs = model(input_ids=input_ids, labels=input_ids)
            total_test_loss += outputs.loss.item()
    pretrain_legacy_test_loss = total_test_loss / max(len(test_loader), 1)
    pretrain_stop_event.set()
    pretrain_thread.join()
    hardware_history.append(
        {
            "epoch": -1,
            "phase": "pretrain_eval",
            **summarize_hardware_samples(
                pretrain_samples,
                device,
                time.process_time() - pretrain_cpu_start,
            ),
        }
    )
    history.append(
        {
            "epoch": -1,
            "phase": "pretrain_eval",
            "train_loss": None,
            "train_model_loss": None,
            "train_objective_loss": None,
            "legacy_test_loss": pretrain_legacy_test_loss,
            "test_loss": pretrain_legacy_test_loss,
            "optimizer_steps_epoch": 0,
            "optimizer_steps_total": 0,
            "epoch_time_sec": 0.0,
            "cumulative_time_sec": 0.0,
        }
    )
    optimizer_steps_total = 0

    for epoch in range(num_epochs):
        # --- hardware metrics: epoch monitoring ---
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(torch.cuda.current_device())
        epoch_cpu_start = time.process_time()
        epoch_stop_event, epoch_thread, epoch_samples = start_hardware_monitor(
            device,
            hardware_sample_interval_sec,
        )

        # --- model metrics: epoch training and evaluation ---
        epoch_start = time.time()
        total_loss = 0.0
        model.train()
        optimizer_steps_epoch = 0

        for step, batch in enumerate(train_loader, start=1):
            input_ids = batch.to(device, non_blocking=pin_memory)
            optimizer.zero_grad(set_to_none=True)
            outputs = model(input_ids=input_ids, labels=input_ids)
            loss = outputs.loss
            loss.backward()
            total_loss += loss.item()
            torch.nn.utils.clip_grad_norm_(
                filter(lambda p: p.requires_grad, model.parameters()),
                max_grad_norm,
            )
            optimizer.step()
            optimizer_steps_epoch += 1
            optimizer_steps_total += 1

        avg_loss = total_loss / max(len(train_loader), 1)

        model.eval()
        total_test_loss = 0.0
        with torch.no_grad():
            for batch in test_loader:
                input_ids = batch.to(device, non_blocking=pin_memory)
                outputs = model(input_ids=input_ids, labels=input_ids)
                total_test_loss += outputs.loss.item()
        avg_test_loss = total_test_loss / len(test_loader)

        epoch_time = time.time() - epoch_start
        cumulative_time = time.time() - total_start
        history.append(
            {
                "epoch": epoch,
                "phase": "epoch_end",
                "train_loss": avg_loss,
                "train_model_loss": avg_loss,
                "train_objective_loss": avg_loss,
                "legacy_test_loss": avg_test_loss,
                "test_loss": avg_test_loss,
                "optimizer_steps_epoch": optimizer_steps_epoch,
                "optimizer_steps_total": optimizer_steps_total,
                "epoch_time_sec": epoch_time,
                "cumulative_time_sec": cumulative_time,
            }
        )
        epoch_stop_event.set()
        epoch_thread.join()
        hardware_history.append(
            {
                "epoch": epoch,
                "phase": "epoch_end",
                **summarize_hardware_samples(
                    epoch_samples,
                    device,
                    time.process_time() - epoch_cpu_start,
                ),
            }
        )

        # --- model metrics: persisted history ---
        with (run_dir / "train_history.csv").open("w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(
                f,
                fieldnames=[
                    "epoch",
                    "phase",
                    "train_loss",
                    "train_model_loss",
                    "train_objective_loss",
                    "legacy_test_loss",
                    "test_loss",
                    "optimizer_steps_epoch",
                    "optimizer_steps_total",
                    "epoch_time_sec",
                    "cumulative_time_sec",
                ],
            )
            writer.writeheader()
            writer.writerows(history)

        # --- hardware metrics: persisted history ---
        with (run_dir / "hardware_history.csv").open("w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(
                f,
                fieldnames=[
                    "epoch",
                    "phase",
                    "hardware_sample_count",
                    "process_cpu_time_epoch_sec",
                    "system_cpu_percent_mean",
                    "system_cpu_percent_max",
                    "process_ram_rss_gb_mean",
                    "process_ram_rss_gb_max",
                    "system_ram_used_gb_mean",
                    "system_ram_used_gb_max",
                    "gpu_util_percent_mean",
                    "gpu_util_percent_max",
                    "gpu_mem_used_mb_mean",
                    "gpu_mem_used_mb_max",
                    "gpu_peak_allocated_mb",
                    "gpu_peak_reserved_mb",
                    "gpu_allocated_end_mb",
                    "gpu_reserved_end_mb",
                ],
            )
            writer.writeheader()
            writer.writerows(hardware_history)

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
            f"transformer.h.{idx}.attn.c_proj": block.attn.c_proj,
            f"transformer.h.{idx}.mlp.c_fc": block.mlp.c_fc,
            f"transformer.h.{idx}.mlp.c_proj": block.mlp.c_proj,
        }
        for layer_name, module in target_layers.items():
            delta_weight = ((alpha / rank) * (module.B @ module.A)).detach().cpu().float().clone()
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

            layer_stats[layer_name] = {
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
        "method": "lora",
        "display_name": display_name,
        "experiment_profile": experiment_profile,
        "comparison_suite_id": comparison_suite_id,
        "comparison_target": comparison_target,
        "loss_protocol": loss_protocol,
        "total_params": n_total,
        "trainable_params": n_treinaveis,
        "trainable_pct": 100 * n_treinaveis / n_total,
        "final_train_loss": history[-1]["train_loss"],
        "final_train_model_loss": history[-1]["train_model_loss"],
        "final_train_objective_loss": history[-1]["train_objective_loss"],
        "final_test_loss": history[-1]["test_loss"],
        "pretrain_legacy_test_loss": pretrain_legacy_test_loss,
        "final_legacy_test_loss": history[-1]["legacy_test_loss"],
        "optimizer_name": "AdamW",
        "learning_rate": learning_rate,
        "weight_decay": weight_decay,
        "effective_batch_size": effective_batch_size,
        "max_grad_norm": max_grad_norm,
        "num_epochs": num_epochs,
        "seed": seed,
        "max_length": max_lenght,
        "optimizer_steps_per_epoch_planned": optimizer_steps_per_epoch_planned,
        "optimizer_steps_total_planned": optimizer_steps_total_planned,
        "total_time_sec": total_time,
        "run_path": str(run_dir),
        "completed_at": datetime.now(timezone.utc).isoformat(),
        **build_hardware_summary(hardware_history),
    }
    with (run_dir / "summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print(f"Loss final de treino: {history[-1]['train_loss']:.4f}")
    print(f"Loss final de teste: {history[-1]['test_loss']:.4f}")
