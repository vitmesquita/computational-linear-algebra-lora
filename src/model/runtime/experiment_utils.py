import csv
import json
import random
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
from datasets import load_dataset
from torch.utils.data import DataLoader
from transformers import GPT2Tokenizer

from data.load_dataset import WikiTextDataset
from metrics.hardware_metrics import HARDWARE_HISTORY_FIELDS, build_hardware_summary
from metrics.training_metrics import TRAIN_HISTORY_FIELDS


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def resolve_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def configure_runtime(device):
    if device.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        if hasattr(torch, "set_float32_matmul_precision"):
            torch.set_float32_matmul_precision("high")


def build_wikitext_dataloaders(
    max_length,
    batch_size,
    num_workers,
    pin_memory,
    persistent_workers,
    max_train_examples,
    max_test_examples,
):
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    tokenizer.pad_token = tokenizer.eos_token

    raw = load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1")
    train_texts = raw["train"]["text"]
    test_texts = raw["test"]["text"]

    if max_train_examples is not None:
        train_texts = train_texts[:max_train_examples]
    if max_test_examples is not None:
        test_texts = test_texts[:max_test_examples]

    train_dataset = WikiTextDataset(train_texts, tokenizer, max_length=max_length, return_dict=False)
    test_dataset = WikiTextDataset(test_texts, tokenizer, max_length=max_length, return_dict=False)

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
    return train_loader, test_loader


def build_run_dir(method):
    if Path("/content").exists():
        output_root = Path("/content/drive/MyDrive/cla_lora_runs")
    else:
        output_root = Path("outputs/cla_lora_runs")
    run_name = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    run_dir = output_root / method / run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir


def build_run_config(
    method,
    display_name,
    experiment_profile,
    comparison_suite_id,
    comparison_target,
    loss_protocol,
    max_length,
    batch_size,
    effective_batch_size,
    learning_rate,
    weight_decay,
    max_grad_norm,
    num_workers,
    pin_memory,
    persistent_workers,
    num_epochs,
    hardware_sample_interval_sec,
    seed,
    optimizer_steps_per_epoch_planned,
    optimizer_steps_total_planned,
    device,
    run_dir,
    extra_fields=None,
):
    config = {
        "method": method,
        "display_name": display_name,
        "experiment_profile": experiment_profile,
        "comparison_suite_id": comparison_suite_id,
        "comparison_target": comparison_target,
        "loss_protocol": loss_protocol,
        "model_name": "gpt2",
        "dataset_name": "Salesforce/wikitext",
        "dataset_config": "wikitext-2-raw-v1",
        "max_length": max_length,
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
    }
    if extra_fields is not None:
        config.update(extra_fields)
    config.update(
        {
            "seed": seed,
            "optimizer_steps_per_epoch_planned": optimizer_steps_per_epoch_planned,
            "optimizer_steps_total_planned": optimizer_steps_total_planned,
            "device": str(device),
            "output_path": str(run_dir),
            "started_at": datetime.now(timezone.utc).isoformat(),
        }
    )
    return config


def write_json_artifact(run_dir, filename, payload):
    with (run_dir / filename).open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)


def write_epoch_artifacts(run_dir, history, hardware_history):
    with (run_dir / "train_history.csv").open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=TRAIN_HISTORY_FIELDS)
        writer.writeheader()
        writer.writerows(history)

    with (run_dir / "hardware_history.csv").open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=HARDWARE_HISTORY_FIELDS)
        writer.writeheader()
        writer.writerows(hardware_history)


def save_checkpoint(run_dir, filename, epoch, model, optimizer, history):
    torch.save(
        {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "history": history,
        },
        run_dir / filename,
    )


def build_run_summary(
    method,
    display_name,
    experiment_profile,
    comparison_suite_id,
    comparison_target,
    loss_protocol,
    total_params,
    trainable_params,
    history,
    pretrain_legacy_test_loss,
    learning_rate,
    weight_decay,
    effective_batch_size,
    max_grad_norm,
    num_epochs,
    seed,
    max_length,
    optimizer_steps_per_epoch_planned,
    optimizer_steps_total_planned,
    total_time_sec,
    run_dir,
    hardware_history,
    extra_fields=None,
):
    summary = {
        "method": method,
        "display_name": display_name,
        "experiment_profile": experiment_profile,
        "comparison_suite_id": comparison_suite_id,
        "comparison_target": comparison_target,
        "loss_protocol": loss_protocol,
        "total_params": total_params,
        "trainable_params": trainable_params,
        "trainable_pct": 100 * trainable_params / total_params,
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
        "max_length": max_length,
        "optimizer_steps_per_epoch_planned": optimizer_steps_per_epoch_planned,
        "optimizer_steps_total_planned": optimizer_steps_total_planned,
        "total_time_sec": total_time_sec,
        "run_path": str(run_dir),
    }
    if extra_fields is not None:
        summary.update(extra_fields)
    summary.update(
        {
            "completed_at": datetime.now(timezone.utc).isoformat(),
            **build_hardware_summary(hardware_history),
        }
    )
    return summary
