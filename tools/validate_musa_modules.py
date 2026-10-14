import argparse
import copy
import json
from pathlib import Path

import torch
import torch_musa
import torchaudio

from indextts.musa_ops.fbank import musa_fbank
from indextts.s2mel.modules.audio import mel_spectrogram
from indextts.s2mel.modules.bigvgan.bigvgan import BigVGAN
from indextts.s2mel.modules.campplus.DTDNN import CAMPPlus
from tools.validate_musa_audio import benchmark_pair, save_gpu_trace


def compare(actual, expected):
    actual = actual.detach().float().cpu().flatten()
    expected = expected.detach().float().cpu().flatten()
    error = actual - expected
    relative_l2 = (error.norm() / expected.norm().clamp_min(1e-12)).item()
    cosine = torch.nn.functional.cosine_similarity(actual, expected, dim=0).item()
    return {
        "mean_abs": error.abs().mean().item(), "max_abs": error.abs().max().item(),
        "relative_l2": relative_l2, "cosine": cosine,
        "finite": bool(actual.isfinite().all()),
        "diagnostic_pass": bool(actual.isfinite().all()) and relative_l2 <= 0.01 and cosine >= 0.9999,
    }


@torch.inference_mode()
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-dir", default="checkpoints")
    parser.add_argument("--prompt", default="examples/voice_01.wav")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--cpu-threads", type=int, default=4)
    parser.add_argument("--rounds", type=int, default=3)
    parser.add_argument("--allow-tf32", action="store_true")
    args = parser.parse_args()
    torch.set_num_threads(args.cpu_threads)
    torch.backends.mudnn.allow_tf32 = args.allow_tf32
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    root = Path(args.model_dir)
    audio, sample_rate = torchaudio.load(args.prompt)
    audio16 = torchaudio.functional.resample(audio[:1], sample_rate, 16000).float()
    audio22 = torchaudio.functional.resample(audio[:1], sample_rate, 22050).float()
    baseline_features = torchaudio.compliance.kaldi.fbank(
        audio16, num_mel_bins=80, dither=0, sample_frequency=16000,
    )
    baseline_features = (baseline_features - baseline_features.mean(0, keepdim=True)).unsqueeze(0)
    device_features = musa_fbank(audio16.musa())
    device_features = (device_features - device_features.mean(0, keepdim=True)).unsqueeze(0)
    baseline_features_device = baseline_features.musa()
    cpu_campplus = CAMPPlus(feat_dim=80, embedding_size=192).eval()
    cpu_campplus.load_state_dict(torch.load(root / "hf_cache/campplus_cn_common.bin", map_location="cpu"))
    musa_campplus = copy.deepcopy(cpu_campplus).musa().eval()
    expected_embedding = cpu_campplus(baseline_features)
    baseline_embedding_device = musa_campplus(baseline_features_device)
    candidate_embedding = musa_campplus(device_features)
    report = {
        "cpu_threads": torch.get_num_threads(),
        "allow_tf32": torch.backends.mudnn.allow_tf32,
        "diagnostic_gate": "finite, relative L2 <= 1%, cosine >= 0.9999; not production audio-quality acceptance",
        "campplus_same_input": compare(baseline_embedding_device, expected_embedding),
        "campplus_feature_change_only": compare(candidate_embedding, baseline_embedding_device),
        "campplus_combined": compare(candidate_embedding, expected_embedding),
    }
    print(json.dumps(report, indent=2), flush=True)
    cpu_vocoder = BigVGAN.from_pretrained(str(root / "hf_cache/bigvgan"), use_cuda_kernel=False)
    cpu_vocoder.remove_weight_norm()
    cpu_vocoder.eval()
    musa_vocoder = copy.deepcopy(cpu_vocoder).musa().eval()
    mel = mel_spectrogram(audio22, 1024, 80, 22050, 256, 1024, 0, None)
    device_mel = mel.musa()
    expected_waveform = cpu_vocoder(mel)
    candidate_waveform = musa_vocoder(device_mel)
    report["bigvgan_same_input"] = compare(candidate_waveform, expected_waveform)
    with torch.backends.mudnn.flags(allow_tf32=False):
        strict_waveform = musa_vocoder(device_mel)
    report["bigvgan_same_input_tf32_off"] = compare(strict_waveform, expected_waveform)
    report["input_shapes"] = {"campplus": list(baseline_features.shape), "bigvgan": list(mel.shape)}
    report["parameter_devices"] = {
        "campplus": sorted({str(value.device) for value in musa_campplus.parameters()}),
        "bigvgan": sorted({str(value.device) for value in musa_vocoder.parameters()}),
    }
    report["buffer_devices"] = {
        "campplus": sorted({str(value.device) for value in musa_campplus.buffers()}),
        "bigvgan": sorted({str(value.device) for value in musa_vocoder.buffers()}),
    }
    (output / "modules.json").write_text(json.dumps(report, indent=2))
    print("bigvgan", report["bigvgan_same_input"], flush=True)
    print("bigvgan_tf32_off", report["bigvgan_same_input_tf32_off"], flush=True)
    cpu_flush = torch.zeros(128 * 1024 * 1024 // 4)
    device_flush = torch.zeros(256 * 1024 * 1024 // 4, device="musa")
    report["benchmark"] = {}
    for label, cpu_call, musa_call in [
        ("campplus", lambda: cpu_campplus(baseline_features), lambda: musa_campplus(baseline_features_device)),
        ("bigvgan", lambda: cpu_vocoder(mel), lambda: musa_vocoder(device_mel)),
    ]:
        result = benchmark_pair(cpu_call, musa_call, args.rounds, cpu_flush, device_flush, warmup=1)
        report["benchmark"][label] = result
        print(label, {key: value for key, value in result.items() if not isinstance(value, list)}, flush=True)
    with torch.profiler.profile(
        activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.MUSA],
        record_shapes=True,
    ) as profiler:
        musa_campplus(device_features)
        musa_vocoder(device_mel)
        torch.musa.synchronize()
    save_gpu_trace(profiler, output / "modules.gpu.json")
    torchaudio.save(str(output / "bigvgan_cpu.wav"), expected_waveform[0], 22050)
    torchaudio.save(str(output / "bigvgan_musa.wav"), candidate_waveform[0].cpu(), 22050)
    (output / "modules.json").write_text(json.dumps(report, indent=2))
    if not all(report[label]["diagnostic_pass"] for label in
               ["campplus_same_input", "campplus_feature_change_only", "campplus_combined", "bigvgan_same_input"]):
        raise AssertionError("module diagnostic precision gate failed; see modules.json")


if __name__ == "__main__":
    main()
