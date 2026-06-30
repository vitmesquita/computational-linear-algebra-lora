import time

import torch
import torch.nn as nn
from transformers import GPT2LMHeadModel

from model.analysis.algebraic_analysis import build_adalora_delta_map, build_svd_artifacts, save_spectral_artifacts
from model.metrics.hardware_metrics import build_hardware_history_row, build_hardware_info, start_hardware_monitor
from model.metrics.training_metrics import build_epoch_history_row, build_pretrain_history_row
from model.runtime.experiment_utils import (
    build_run_config,
    build_run_dir,
    build_run_summary,
    build_wikitext_dataloaders,
    configure_runtime,
    resolve_device,
    save_checkpoint,
    set_seed,
    write_epoch_artifacts,
    write_json_artifact,
)


def budget_scheduler(step, b_initial, b_final, t_i, t_f, T):
    """Eq. (12) do paper AdaLoRA."""
    if step < t_i:
        return b_initial
    if step >= T - t_f:
        return b_final
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


class ScoreCalculator:
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
            return (param * grad).abs()

    def scoring(self, param, beta_1, beta_2):
        with torch.no_grad():
            importance = self.calculate_sensibility(param)
            self.I_bar = beta_1 * self.I_bar + (1 - beta_1) * importance
            self.U_bar = beta_2 * self.U_bar + (1 - beta_2) * (importance - self.I_bar).abs()
            score_matrix = self.I_bar * self.U_bar
        return score_matrix


class AdaLoRALinear(nn.Module):
    def __init__(self, linear: nn.Linear, rank, alpha, beta_1, beta_2):
        super().__init__()
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
        base = self.linear(x)
        lora = ((x @ self.P) * self.Lambda) @ self.Q
        return base + (self.alpha / self.r) * lora

    def merge(self):
        with torch.no_grad():
            self.linear.weight.data += (self.alpha / self.r) * (self.P @ (self.Lambda.unsqueeze(1) * self.Q))

    def calculate_ipt(self):
        with torch.no_grad():
            self.P_score = self.P_score_calculator.scoring(self.P, self.beta_1, self.beta_2)
            self.Lambda_score = self.Lambda_score_calculator.scoring(self.Lambda, self.beta_1, self.beta_2)
            self.Q_score = self.Q_score_calculator.scoring(self.Q, self.beta_1, self.beta_2)

            ipt = self.Lambda_score.clone()
            ipt += (1 / self.d) * self.P_score.sum(dim=0)
            ipt += (1 / self.k) * self.Q_score.sum(dim=1)
            self.ipt = ipt
        return ipt


def main():
    seed = 42
    set_seed(seed)

    device = resolve_device()
    print(f"Using device: {device}")
    configure_runtime(device)

    model = GPT2LMHeadModel.from_pretrained("gpt2")

    reg_weight = 0.1
    rank = 4
    alpha = rank
    beta_1 = 0.85
    beta_2 = 0.85
    step = 0
    delta_T = 10

    print("Freezing weights")
    for p in model.parameters():
        p.requires_grad = False

    print("Replacing layers")
    for block in model.transformer.h:
        block.attn.c_attn = AdaLoRALinear(block.attn.c_attn, rank=rank, alpha=alpha, beta_1=beta_1, beta_2=beta_2)
        block.attn.c_proj = AdaLoRALinear(block.attn.c_proj, rank=rank, alpha=alpha, beta_1=beta_1, beta_2=beta_2)
        block.mlp.c_fc = AdaLoRALinear(block.mlp.c_fc, rank=rank, alpha=alpha, beta_1=beta_1, beta_2=beta_2)
        block.mlp.c_proj = AdaLoRALinear(block.mlp.c_proj, rank=rank, alpha=alpha, beta_1=beta_1, beta_2=beta_2)

    print("Sanity check...")
    for block in model.transformer.h:
        ada = block.attn.c_attn
        assert (ada.P @ (ada.Lambda.unsqueeze(1) * ada.Q)).abs().max().item() == 0.0, "ΔW não é zero no início"

    n_treinaveis = sum(p.numel() for p in model.parameters() if p.requires_grad)
    n_total = sum(p.numel() for p in model.parameters())
    print(f"Parâmetros treináveis: {n_treinaveis:,} / {n_total:,} ({100*n_treinaveis/n_total:.3f}%)")

    model = model.to(device)
    for module in model.modules():
        if isinstance(module, AdaLoRALinear):
            for calc in [module.P_score_calculator, module.Lambda_score_calculator, module.Q_score_calculator]:
                calc.I_bar = calc.I_bar.to(device)
                calc.U_bar = calc.U_bar.to(device)

    experiment_profile = "legacy_controlled_v1"
    comparison_suite_id = "legacy_controlled_v1"
    comparison_target = "controlled_method_comparison"
    loss_protocol = "legacy"
    display_name = "adalora"

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

    train_loader, test_loader = build_wikitext_dataloaders(
        max_length=max_lenght,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=persistent_workers,
        max_train_examples=max_train_examples,
        max_test_examples=max_test_examples,
    )
    optimizer_steps_per_epoch_planned = len(train_loader)
    optimizer_steps_total_planned = optimizer_steps_per_epoch_planned * num_epochs

    T = len(train_loader) * num_epochs
    t_i = max(1, T // 5)
    t_f = max(1, T // 5)
    n_matrices = sum(1 for m in model.modules() if isinstance(m, AdaLoRALinear))
    b_final = rank * n_matrices // 2
    b_initial = int(1.5 * b_final)
    print(f"T={T} | t_i={t_i} | t_f={t_f} | b_initial={b_initial} | b_final={b_final}")

    run_dir = build_run_dir("adalora")
    config = build_run_config(
        method="adalora",
        display_name=display_name,
        experiment_profile=experiment_profile,
        comparison_suite_id=comparison_suite_id,
        comparison_target=comparison_target,
        loss_protocol=loss_protocol,
        max_length=max_lenght,
        batch_size=batch_size,
        effective_batch_size=effective_batch_size,
        learning_rate=learning_rate,
        weight_decay=weight_decay,
        max_grad_norm=max_grad_norm,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=persistent_workers,
        num_epochs=num_epochs,
        hardware_sample_interval_sec=hardware_sample_interval_sec,
        seed=seed,
        optimizer_steps_per_epoch_planned=optimizer_steps_per_epoch_planned,
        optimizer_steps_total_planned=optimizer_steps_total_planned,
        device=device,
        run_dir=run_dir,
        extra_fields={
            "rank": rank,
            "alpha": alpha,
            "reg_weight": reg_weight,
            "beta_1": beta_1,
            "beta_2": beta_2,
            "delta_T": delta_T,
        },
    )
    write_json_artifact(run_dir, "config.json", config)
    write_json_artifact(run_dir, "hardware_info.json", build_hardware_info(device, hardware_sample_interval_sec))

    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=learning_rate,
        weight_decay=weight_decay,
    )

    history = []
    total_start = time.time()
    hardware_history = []

    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(torch.cuda.current_device())
    pretrain_cpu_start = time.process_time()
    pretrain_stop_event, pretrain_thread, pretrain_samples = start_hardware_monitor(device, hardware_sample_interval_sec)

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
        build_hardware_history_row(
            epoch=-1,
            phase="pretrain_eval",
            samples=pretrain_samples,
            device=device,
            process_cpu_time_epoch_sec=time.process_time() - pretrain_cpu_start,
        )
    )
    history.append(build_pretrain_history_row(pretrain_legacy_test_loss))

    for epoch in range(num_epochs):
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(torch.cuda.current_device())
        epoch_cpu_start = time.process_time()
        epoch_stop_event, epoch_thread, epoch_samples = start_hardware_monitor(device, hardware_sample_interval_sec)

        epoch_start = time.time()
        total_model_loss = 0.0
        total_loss = 0.0
        model.train()
        optimizer_steps_epoch = 0

        for batch_idx, batch in enumerate(train_loader, start=1):
            input_ids = batch.to(device, non_blocking=pin_memory)
            optimizer.zero_grad(set_to_none=True)
            output = model(input_ids=input_ids, labels=input_ids)
            model_loss = output.loss

            reg = 0
            for module in model.modules():
                if isinstance(module, AdaLoRALinear):
                    reg += torch.norm(module.P.T @ module.P - torch.eye(module.r, device=module.P.device), p="fro") ** 2
                    reg += torch.norm(module.Q @ module.Q.T - torch.eye(module.r, device=module.P.device), p="fro") ** 2

            loss = model_loss + reg_weight * reg
            loss.backward()
            total_model_loss += model_loss.item()
            total_loss += loss.item()
            torch.nn.utils.clip_grad_norm_(filter(lambda p: p.requires_grad, model.parameters()), max_grad_norm)
            optimizer.step()
            optimizer_steps_epoch += 1

            if t_i <= step < T - t_f and step % delta_T == 0:
                current_budget = budget_scheduler(step, b_initial, b_final, t_i, t_f, T)
                global_update_lambda(model, current_budget)

            step += 1

        avg_loss = total_loss / max(len(train_loader), 1)
        avg_model_loss = total_model_loss / max(len(train_loader), 1)

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
            build_epoch_history_row(
                epoch=epoch,
                train_loss=avg_loss,
                train_model_loss=avg_model_loss,
                train_objective_loss=avg_loss,
                test_loss=avg_test_loss,
                optimizer_steps_epoch=optimizer_steps_epoch,
                optimizer_steps_total=step,
                epoch_time_sec=epoch_time,
                cumulative_time_sec=cumulative_time,
            )
        )

        epoch_stop_event.set()
        epoch_thread.join()
        hardware_history.append(
            build_hardware_history_row(
                epoch=epoch,
                phase="epoch_end",
                samples=epoch_samples,
                device=device,
                process_cpu_time_epoch_sec=time.process_time() - epoch_cpu_start,
            )
        )

        write_epoch_artifacts(run_dir, history, hardware_history)
        save_checkpoint(run_dir, "latest_checkpoint.pt", epoch, model, optimizer, history)
        print(f"Epoch {epoch} | Loss médio treino: {avg_loss:.4f} | Loss médio teste: {avg_test_loss:.4f}")

    save_checkpoint(run_dir, "final_checkpoint.pt", num_epochs - 1, model, optimizer, history)

    delta_map, extra_layer_fields = build_adalora_delta_map(model, rank, alpha)
    svd_map, layer_stats = build_svd_artifacts(delta_map, extra_layer_fields)
    save_spectral_artifacts(run_dir, delta_map, svd_map, layer_stats)

    summary = build_run_summary(
        method="adalora",
        display_name=display_name,
        experiment_profile=experiment_profile,
        comparison_suite_id=comparison_suite_id,
        comparison_target=comparison_target,
        loss_protocol=loss_protocol,
        total_params=n_total,
        trainable_params=n_treinaveis,
        history=history,
        pretrain_legacy_test_loss=pretrain_legacy_test_loss,
        learning_rate=learning_rate,
        weight_decay=weight_decay,
        effective_batch_size=effective_batch_size,
        max_grad_norm=max_grad_norm,
        num_epochs=num_epochs,
        seed=seed,
        max_length=max_lenght,
        optimizer_steps_per_epoch_planned=optimizer_steps_per_epoch_planned,
        optimizer_steps_total_planned=optimizer_steps_total_planned,
        total_time_sec=time.time() - total_start,
        run_dir=run_dir,
        hardware_history=hardware_history,
    )
    write_json_artifact(run_dir, "summary.json", summary)

    print(f"Loss final de treino: {history[-1]['train_loss']:.4f}")
    print(f"Loss final de teste: {history[-1]['test_loss']:.4f}")


if __name__ == "__main__":
    main()
