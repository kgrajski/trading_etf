# -*- coding: utf-8 -*-
"""
Device detection utilities for PyTorch.

Provides consistent device selection across the project, supporting:
- CUDA (NVIDIA GPUs on Linux/Windows, including AWS GPU instances)
- MPS (Apple Silicon M1/M2/M3)
- CPU (fallback)

Usage:
    from src.utils.device import get_device
    device = get_device()  # Automatically selects best available
    model = model.to(device)
    tensor = tensor.to(device)
"""

import torch


def get_device(prefer_mps: bool = True, verbose: bool = True) -> torch.device:
    """
    Get the best available PyTorch device.

    Priority order:
    1. CUDA (if available) - for AWS GPU instances and NVIDIA GPUs
    2. MPS (if available and prefer_mps=True) - for Apple Silicon
    3. CPU (fallback)

    Args:
        prefer_mps: Whether to prefer MPS over CPU on Apple Silicon.
                   Set to False if MPS causes issues with specific operations.
        verbose: Whether to print device selection info.

    Returns:
        torch.device: The selected device.
    """
    if torch.cuda.is_available():
        device = torch.device("cuda")
        if verbose:
            gpu_name = torch.cuda.get_device_name(0)
            gpu_memory = torch.cuda.get_device_properties(0).total_memory / (1024**3)
            print(f"🚀 Device: CUDA ({gpu_name}, {gpu_memory:.1f} GB)")
    elif prefer_mps and torch.backends.mps.is_available():
        device = torch.device("mps")
        if verbose:
            print("🍎 Device: MPS (Apple Silicon)")
    else:
        device = torch.device("cpu")
        if verbose:
            print("💻 Device: CPU")

    return device


def get_device_info() -> dict:
    """
    Get detailed information about available devices.

    Returns:
        dict: Device availability and properties.
    """
    info = {
        "cuda_available": torch.cuda.is_available(),
        "mps_available": torch.backends.mps.is_available(),
        "device_count": torch.cuda.device_count() if torch.cuda.is_available() else 0,
        "current_device": None,
        "cuda_devices": [],
    }

    if torch.cuda.is_available():
        info["current_device"] = torch.cuda.current_device()
        for i in range(torch.cuda.device_count()):
            props = torch.cuda.get_device_properties(i)
            info["cuda_devices"].append(
                {
                    "index": i,
                    "name": props.name,
                    "total_memory_gb": props.total_memory / (1024**3),
                    "major": props.major,
                    "minor": props.minor,
                    "multi_processor_count": props.multi_processor_count,
                }
            )

    return info


def move_to_device(data, device: torch.device):
    """
    Recursively move data (tensors, dicts, lists) to the specified device.

    Args:
        data: Data to move (tensor, dict, list, or other).
        device: Target device.

    Returns:
        Data with tensors moved to device.
    """
    if isinstance(data, torch.Tensor):
        return data.to(device)
    elif isinstance(data, dict):
        return {k: move_to_device(v, device) for k, v in data.items()}
    elif isinstance(data, (list, tuple)):
        return type(data)(move_to_device(v, device) for v in data)
    else:
        return data


def ensure_reproducibility(seed: int = 42) -> None:
    """
    Set random seeds for reproducible results.

    Note: For full reproducibility with CUDA, you may also need:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    Args:
        seed: Random seed value.
    """
    import random

    import numpy as np

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
