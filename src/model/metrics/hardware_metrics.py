import shutil
import subprocess
import threading

import torch

try:
    import psutil
except ImportError:
    psutil = None


HARDWARE_HISTORY_FIELDS = [
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
]


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


def build_hardware_history_row(epoch, phase, samples, device, process_cpu_time_epoch_sec):
    return {
        "epoch": epoch,
        "phase": phase,
        **summarize_hardware_samples(samples, device, process_cpu_time_epoch_sec),
    }


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
