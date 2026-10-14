import os
from pathlib import Path

import torch


_LOADED = False
_TWIDDLES = {}


def load_real_stft(build_directory=None):
    global _LOADED
    if _LOADED:
        return
    from torch_musa.utils.musa_extension import load

    root = Path(__file__).parent
    build_directory = build_directory or os.environ.get(
        "INDEXTTS_MUSA_BUILD_DIR",
        str(Path(os.environ.get("XDG_CACHE_HOME", Path.home() / ".cache")) / "indextts" / "musa_ops"),
    )
    Path(build_directory).mkdir(parents=True, exist_ok=True)
    load(
        name="indextts_real_stft_v4",
        sources=[str(root / "real_stft.cpp"), str(root / "real_stft.mu")],
        build_directory=build_directory,
        extra_cflags=["-O3", "-std=c++17"],
        extra_musa_cflags=["-O3", "-std=c++17"],
        with_musa=True,
        is_python_module=False,
        verbose=False,
    )
    _LOADED = True


def real_stft(input_tensor, window, hop):
    load_real_stft()
    key = (input_tensor.device, input_tensor.dtype)
    if key not in _TWIDDLES:
        indices = torch.arange(512, device=input_tensor.device, dtype=torch.float32)
        angles = -2.0 * torch.pi * indices / 1024.0
        _TWIDDLES[key] = (torch.cos(angles), torch.sin(angles))
    twiddle_cos, twiddle_sin = _TWIDDLES[key]
    return torch.ops.musa_audio.real_stft(
        input_tensor, window, twiddle_cos, twiddle_sin, hop
    )


def real_stft_512(input_tensor, window, hop):
    load_real_stft()
    key = (input_tensor.device, input_tensor.dtype, 512)
    if key not in _TWIDDLES:
        indices = torch.arange(256, device=input_tensor.device, dtype=torch.float32)
        angles = -2.0 * torch.pi * indices / 512.0
        _TWIDDLES[key] = (torch.cos(angles), torch.sin(angles))
    twiddle_cos, twiddle_sin = _TWIDDLES[key]
    return torch.ops.musa_audio.real_stft_512(
        input_tensor, window, twiddle_cos, twiddle_sin, hop
    )
