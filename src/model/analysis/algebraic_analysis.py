import json

import torch


def get_target_module_names(model):
    target_module_names = []
    for idx, _ in enumerate(model.transformer.h):
        target_module_names.append(f"transformer.h.{idx}.attn.c_attn")
        target_module_names.append(f"transformer.h.{idx}.attn.c_proj")
        target_module_names.append(f"transformer.h.{idx}.mlp.c_fc")
        target_module_names.append(f"transformer.h.{idx}.mlp.c_proj")
    return target_module_names


def capture_reference_weights(model, module_names):
    return {
        module_name: model.get_submodule(module_name).weight.detach().cpu().float().clone()
        for module_name in module_names
    }


def build_lora_delta_map(model, rank, alpha):
    delta_map = {}
    for idx, block in enumerate(model.transformer.h):
        target_layers = {
            f"transformer.h.{idx}.attn.c_attn": block.attn.c_attn,
            f"transformer.h.{idx}.attn.c_proj": block.attn.c_proj,
            f"transformer.h.{idx}.mlp.c_fc": block.mlp.c_fc,
            f"transformer.h.{idx}.mlp.c_proj": block.mlp.c_proj,
        }
        for layer_name, module in target_layers.items():
            delta_map[layer_name] = ((alpha / rank) * (module.B @ module.A)).detach().cpu().float().clone()
    return delta_map


def build_adalora_delta_map(model, rank, alpha):
    delta_map = {}
    extra_layer_fields = {}
    for idx, block in enumerate(model.transformer.h):
        target_layers = {
            f"transformer.h.{idx}.attn.c_attn": block.attn.c_attn,
            f"transformer.h.{idx}.attn.c_proj": block.attn.c_proj,
            f"transformer.h.{idx}.mlp.c_fc": block.mlp.c_fc,
            f"transformer.h.{idx}.mlp.c_proj": block.mlp.c_proj,
        }
        for layer_name, module in target_layers.items():
            delta_map[layer_name] = ((alpha / rank) * (module.P @ (module.Lambda.unsqueeze(1) * module.Q))).detach().cpu().float().clone()
            extra_layer_fields[layer_name] = {
                "effective_rank": int((module.Lambda.detach().abs() > 1e-8).sum().item())
            }
    return delta_map, extra_layer_fields


def build_full_finetune_delta_map(model, reference_weights, module_names):
    delta_map = {}
    for module_name in module_names:
        final_weight = model.get_submodule(module_name).weight.detach().cpu().float().clone()
        delta_map[module_name] = final_weight - reference_weights[module_name]
    return delta_map


def build_svd_artifacts(delta_map, extra_layer_fields=None):
    svd_map = {}
    layer_stats = {}

    for layer_name, delta_weight in delta_map.items():
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
        if extra_layer_fields is not None and layer_name in extra_layer_fields:
            layer_stats[layer_name].update(extra_layer_fields[layer_name])

    return svd_map, layer_stats


def save_spectral_artifacts(run_dir, delta_map, svd_map, layer_stats):
    torch.save(delta_map, run_dir / "target_deltas.pt")
    torch.save(svd_map, run_dir / "target_svdvals.pt")
    with (run_dir / "layer_stats.json").open("w", encoding="utf-8") as f:
        json.dump(layer_stats, f, indent=2, ensure_ascii=False)
