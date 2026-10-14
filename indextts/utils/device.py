"""Device helpers shared by the CUDA, MUSA and CPU inference paths."""

import torch


def musa_available() -> bool:
    return hasattr(torch, "musa") and torch.musa.is_available()


def default_device() -> str:
    if musa_available():
        return "musa:0"
    if torch.cuda.is_available():
        return "cuda:0"
    if hasattr(torch, "xpu") and torch.xpu.is_available():
        return "xpu"
    if hasattr(torch, "mps") and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def device_type(device) -> str:
    return torch.device(device).type


def device_count(device=None) -> int:
    backend = device_type(device) if device is not None else None
    if backend == "cuda" or backend is None and torch.cuda.is_available():
        return torch.cuda.device_count()
    if backend == "musa" or backend is None and musa_available():
        return torch.musa.device_count()
    return 0


def backend_device(device, index: int) -> str:
    backend = device_type(device)
    return f"{backend}:{index}"


def set_device(device) -> None:
    backend = device_type(device)
    if backend == "cuda":
        torch.cuda.set_device(device)
    elif backend == "musa" and musa_available():
        torch.musa.set_device(device)


def empty_cache(device=None) -> None:
    backend = device_type(device) if device is not None else None
    if backend == "cuda" or backend is None and torch.cuda.is_available():
        torch.cuda.empty_cache()
    elif backend == "musa" and musa_available():
        torch.musa.empty_cache()
    elif backend == "mps" and hasattr(torch, "mps"):
        torch.mps.empty_cache()


def synchronize(device=None) -> None:
    backend = device_type(device) if device is not None else None
    if backend == "cuda" or backend is None and torch.cuda.is_available():
        torch.cuda.synchronize(device)
    elif backend == "musa" and musa_available():
        torch.musa.synchronize(device)


def total_memory_gb(device) -> float:
    backend = device_type(device)
    index = torch.device(device).index or 0
    if backend == "cuda":
        return torch.cuda.get_device_properties(index).total_memory / (1024 ** 3)
    if backend == "musa" and musa_available():
        return torch.musa.get_device_properties(index).total_memory / (1024 ** 3)
    return 0.0


def bf16_supported(device) -> bool:
    backend = device_type(device)
    if backend == "cuda":
        return torch.cuda.is_bf16_supported()
    if backend == "musa" and musa_available():
        return True
    return False
