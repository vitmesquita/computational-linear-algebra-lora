import time

import torch
from transformers import GPT2LMHeadModel

from model.analysis.algebraic_analysis import (
    build_full_finetune_delta_map,
    build_svd_artifacts,
    capture_reference_weights,
    get_target_module_names,
    save_spectral_artifacts,
)
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


def main():
    seed = 42
    set_seed(seed)

    device = resolve_device()
    print(f"Using device: {device}")
    configure_runtime(device)

    model = GPT2LMHeadModel.from_pretrained("gpt2")
    target_module_names = get_target_module_names(model)
    initial_weights = capture_reference_weights(model, target_module_names)

    n_treinaveis = sum(p.numel() for p in model.parameters() if p.requires_grad)
    n_total = sum(p.numel() for p in model.parameters())
    print(f"Parâmetros treináveis: {n_treinaveis:,} / {n_total:,} ({100*n_treinaveis/n_total:.3f}%)")

    model = model.to(device)

    experiment_profile = "legacy_controlled_v1"
    comparison_suite_id = "legacy_controlled_v1"
    loss_protocol = "legacy"
    comparison_target = "controlled_method_comparison"
    display_name = "full_finetune"

    max_length = 128
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
        max_length=max_length,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=persistent_workers,
        max_train_examples=max_train_examples,
        max_test_examples=max_test_examples,
    )
    optimizer_steps_per_epoch_planned = len(train_loader)
    optimizer_steps_total_planned = optimizer_steps_per_epoch_planned * num_epochs

    run_dir = build_run_dir("full_finetune")
    config = build_run_config(
        method="full_finetune",
        display_name=display_name,
        experiment_profile=experiment_profile,
        comparison_suite_id=comparison_suite_id,
        comparison_target=comparison_target,
        loss_protocol=loss_protocol,
        max_length=max_length,
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
    )
    write_json_artifact(run_dir, "config.json", config)
    write_json_artifact(run_dir, "hardware_info.json", build_hardware_info(device, hardware_sample_interval_sec))

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=learning_rate,
        weight_decay=weight_decay,
    )

    history = []
    total_start = time.time()

    def evaluate_legacy(dataloader):
        model.eval()
        total_loss = 0.0
        total_batches = 0
        with torch.no_grad():
            for batch in dataloader:
                input_ids = batch.to(device, non_blocking=pin_memory)
                outputs = model(input_ids=input_ids, labels=input_ids)
                total_loss += outputs.loss.item()
                total_batches += 1
        return total_loss / max(total_batches, 1)

    hardware_history = []
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(torch.cuda.current_device())
    pretrain_cpu_start = time.process_time()
    pretrain_stop_event, pretrain_thread, pretrain_samples = start_hardware_monitor(device, hardware_sample_interval_sec)
    pretrain_legacy_test_loss = evaluate_legacy(test_loader)
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
    optimizer_steps_total = 0

    for epoch in range(num_epochs):
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(torch.cuda.current_device())
        epoch_cpu_start = time.process_time()
        epoch_stop_event, epoch_thread, epoch_samples = start_hardware_monitor(device, hardware_sample_interval_sec)

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
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
            optimizer.step()
            optimizer_steps_epoch += 1
            optimizer_steps_total += 1

        avg_loss = total_loss / max(len(train_loader), 1)
        avg_test_loss = evaluate_legacy(test_loader)

        epoch_time = time.time() - epoch_start
        cumulative_time = time.time() - total_start
        history.append(
            build_epoch_history_row(
                epoch=epoch,
                train_loss=avg_loss,
                train_model_loss=avg_loss,
                train_objective_loss=avg_loss,
                test_loss=avg_test_loss,
                optimizer_steps_epoch=optimizer_steps_epoch,
                optimizer_steps_total=optimizer_steps_total,
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

    delta_map = build_full_finetune_delta_map(model, initial_weights, target_module_names)
    svd_map, layer_stats = build_svd_artifacts(delta_map)
    save_spectral_artifacts(run_dir, delta_map, svd_map, layer_stats)

    summary = build_run_summary(
        method="full_finetune",
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
        max_length=max_length,
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
