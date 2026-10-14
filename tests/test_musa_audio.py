import pytest
import torch
import torchaudio

pytest.importorskip("torch_musa")

from indextts.musa_ops.fbank import musa_fbank
from indextts.musa_ops.real_stft import real_stft, real_stft_512


pytestmark = pytest.mark.skipif(not torch.musa.is_available(), reason="MUSA required")


@pytest.fixture(autouse=True)
def strict_musa_fp32():
    with torch.backends.mudnn.flags(allow_tf32=False):
        yield


@pytest.mark.parametrize("fft_size", [512, 1024])
@pytest.mark.parametrize("batch", [1, 3])
def test_fft_matches_cpu(fft_size, batch):
    torch.manual_seed(2026)
    waveform = torch.randn(batch, 4097)
    window = torch.hann_window(fft_size)
    expected = torch.view_as_real(torch.stft(
        waveform, fft_size, hop_length=160, window=window, center=False,
        return_complex=True,
    ))
    kernel = real_stft_512 if fft_size == 512 else real_stft
    actual = kernel(waveform.musa(), window.musa(), 160).cpu()
    torch.testing.assert_close(actual, expected, atol=4e-5, rtol=3e-5)


@pytest.mark.parametrize("samples", [400, 401, 16000, 240000])
@pytest.mark.parametrize("silent", [False, True])
def test_fbank_matches_cpu(samples, silent):
    torch.manual_seed(2026)
    waveform = torch.zeros(1, samples) if silent else torch.randn(1, samples)
    expected = torchaudio.compliance.kaldi.fbank(
        waveform, num_mel_bins=80, dither=0, sample_frequency=16000,
    )
    actual = musa_fbank(waveform.musa()).cpu()
    torch.testing.assert_close(actual, expected, atol=2e-3, rtol=1e-4)


def test_fbank_cache_respects_configuration():
    waveform = torch.randn(1, 16000)
    device_waveform = waveform.musa()
    for bins, low_frequency in [(80, 20), (40, 40), (80, 20)]:
        expected = torchaudio.compliance.kaldi.fbank(
            waveform, num_mel_bins=bins, low_freq=low_frequency,
            sample_frequency=16000, dither=0,
        )
        actual = musa_fbank(
            device_waveform, num_mel_bins=bins, low_freq=low_frequency,
        ).cpu()
        torch.testing.assert_close(actual, expected, atol=2e-3, rtol=1e-4)


def test_fbank_rejects_unsupported_dither():
    with pytest.raises(ValueError, match="dither"):
        musa_fbank(torch.zeros(1, 400, device="musa"), dither=1.0)
