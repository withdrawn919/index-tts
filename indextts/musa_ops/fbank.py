from __future__ import annotations

import torch

from .real_stft import real_stft_512


_MEL_BANKS = {}
_ONES = {}
_WINDOWS = {}


def _musa_fbank(
    waveform: torch.Tensor,
    *,
    num_mel_bins: int = 80,
    sample_frequency: float = 16000.0,
    frame_length: float = 25.0,
    frame_shift: float = 10.0,
    low_freq: float = 20.0,
    high_freq: float = 0.0,
    dither: float = 0.0,
    energy_floor: float = 1.0,
    preemphasis_coefficient: float = 0.97,
    remove_dc_offset: bool = True,
    use_log_fbank: bool = True,
    use_power: bool = True,
) -> torch.Tensor:
    """MUSA implementation of the fixed Kaldi fbank configuration used by IndexTTS."""
    if waveform.device.type != "musa":
        raise ValueError("musa_fbank expects a MUSA waveform")
    if waveform.dim() != 2 or waveform.size(0) != 1:
        raise ValueError("musa_fbank expects a mono waveform shaped [1, samples]")
    if dither != 0.0:
        raise ValueError("musa_fbank currently supports dither=0 only")
    if not 0.0 <= preemphasis_coefficient <= 1.0:
        raise ValueError("preemphasis_coefficient must be between zero and one")
    if sample_frequency != 16000.0 or frame_length != 25.0 or frame_shift != 10.0:
        raise ValueError("musa_fbank supports 16 kHz / 25 ms / 10 ms only")
    if waveform.dtype != torch.float32:
        waveform = waveform.float()

    window_size = int(sample_frequency * frame_length / 1000.0)
    window_shift = int(sample_frequency * frame_shift / 1000.0)
    padded_window_size = 1 << (window_size - 1).bit_length()
    if (window_size, window_shift, padded_window_size) != (400, 160, 512):
        raise ValueError("the current MUSA fbank kernel supports 16 kHz / 25 ms / 10 ms only")
    if waveform.size(1) < window_size:
        return waveform.new_empty((0, num_mel_bins))

    frame_count = 1 + (waveform.size(1) - window_size) // window_shift
    strides = (window_shift * waveform.stride(1), waveform.stride(1))
    frames = waveform.as_strided((frame_count, window_size), strides)
    if remove_dc_offset:
        frames = frames - frames.mean(dim=1, keepdim=True)
    if preemphasis_coefficient != 0.0:
        previous = torch.nn.functional.pad(frames.unsqueeze(0), (1, 0), mode="replicate")
        frames = frames - preemphasis_coefficient * previous.squeeze(0)[:, :-1]

    key = (str(waveform.device), waveform.dtype, num_mel_bins, low_freq, high_freq)
    if key not in _MEL_BANKS:
        from torchaudio.compliance.kaldi import get_mel_banks

        banks, _ = get_mel_banks(
            num_mel_bins,
            padded_window_size,
            sample_frequency,
            low_freq,
            high_freq,
            100.0,
            -500.0,
            1.0,
        )
        _MEL_BANKS[key] = torch.nn.functional.pad(
            banks.to(device=waveform.device, dtype=waveform.dtype), (0, 1)
        )
        _ONES[key] = torch.ones(padded_window_size, device=waveform.device, dtype=waveform.dtype)
        _WINDOWS[key] = torch.hann_window(window_size, periodic=False).pow(0.85).to(waveform.device)

    frames = frames * _WINDOWS[key].unsqueeze(0)
    frames = torch.nn.functional.pad(frames.unsqueeze(0), (0, padded_window_size - window_size)).squeeze(0)

    spectrum = real_stft_512(frames, _ONES[key], padded_window_size)
    spectrum = spectrum[:, :, 0, :]
    spectrum = spectrum[..., 0].square() + spectrum[..., 1].square()
    if not use_power:
        spectrum = spectrum.sqrt()
    mel = spectrum @ _MEL_BANKS[key].transpose(0, 1)
    if use_log_fbank:
        mel = mel.clamp_min(torch.finfo(waveform.dtype).eps).log()
    return mel


def musa_fbank(waveform: torch.Tensor, **kwargs) -> torch.Tensor:
    return _musa_fbank(waveform, **kwargs)
