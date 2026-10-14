import argparse
import json
import statistics
import time
from collections import Counter
from pathlib import Path

import torch
import torch_musa
import torchaudio

from indextts.musa_ops.fbank import musa_fbank
from indextts.musa_ops.real_stft import real_stft
from indextts.s2mel.modules.audio import mel_spectrogram


def error_metrics(actual, expected):
    actual = actual.cpu()
    difference = (actual - expected).abs()
    return {
        "shape": list(actual.shape),
        "mean_abs": difference.mean().item(),
        "max_abs": difference.max().item(),
        "finite": bool(actual.isfinite().all()),
        "pass_atol_2e_3_rtol_1e_4": torch.allclose(actual, expected, atol=2e-3, rtol=1e-4),
    }


def save_gpu_trace(profiler, output):
    raw_path = output.with_name(output.stem + ".raw.json")
    profiler.export_chrome_trace(str(raw_path))
    trace = json.loads(raw_path.read_text())
    events = [event for event in trace["traceEvents"]
              if event.get("ph") == "M" or event.get("cat") in ("kernel", "gpu_memcpy", "gpu_memset")]
    trace["traceEvents"] = events
    output.write_text(json.dumps(trace))
    return dict(Counter(event["name"] for event in events
                        if event.get("cat") == "kernel" and "real_stft" in event["name"]))


def benchmark_pair(cpu_call, musa_call, rounds, cpu_flush, device_flush, warmup=5):
    for _ in range(warmup):
        cpu_call()
        musa_call()
    torch.musa.synchronize()
    durations = {"cpu_us": [], "musa_us": []}
    for iteration in range(rounds):
        order = [("cpu_us", cpu_call), ("musa_us", musa_call)]
        if iteration % 2:
            order.reverse()
        for name, operation in order:
            cpu_flush.add_(1)
            device_flush.add_(1)
            torch.musa.synchronize()
            start = time.perf_counter_ns()
            operation()
            if name == "musa_us":
                torch.musa.synchronize()
            durations[name].append((time.perf_counter_ns() - start) / 1000.0)
    durations["cpu_median_us"] = statistics.median(durations["cpu_us"])
    durations["musa_median_us"] = statistics.median(durations["musa_us"])
    durations["speedup"] = durations["cpu_median_us"] / durations["musa_median_us"]
    return durations


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--prompt", default="examples/voice_01.wav")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--rounds", type=int, default=20)
    parser.add_argument("--cpu-threads", type=int, default=4)
    parser.add_argument("--demo", action="store_true")
    parser.add_argument("--allow-tf32", action="store_true")
    parser.add_argument("--model-dir", default="checkpoints")
    args = parser.parse_args()
    if args.demo and args.allow_tf32:
        parser.error("the model demo disables TF32; omit --allow-tf32 for consistent measurements")
    torch.set_num_threads(args.cpu_threads)
    torch.backends.mudnn.allow_tf32 = args.allow_tf32
    torch.manual_seed(2026)
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    waveform, sample_rate = torchaudio.load(args.prompt)
    audio_16k = torchaudio.functional.resample(waveform[:1], sample_rate, 16000).float()
    audio_22k = torchaudio.functional.resample(waveform[:1], sample_rate, 22050).float()

    def mel(audio):
        return mel_spectrogram(audio, 1024, 80, 22050, 256, 1024, 0, None)

    def cpu_fbank(audio):
        return torchaudio.compliance.kaldi.fbank(
            audio, num_mel_bins=80, dither=0, sample_frequency=16000,
        )

    report = {
        "cpu_threads": torch.get_num_threads(), "torch": torch.__version__,
        "allow_tf32": torch.backends.mudnn.allow_tf32,
        "backend": "musa",
        "flush_cpu_mib": 128, "flush_gpu_mib": 256,
        "timing": "interleaved wall clock; warmed code, cold data; flush excluded; MUSA synchronized",
        "precision": {}, "benchmark": {},
    }
    for label, audio, cpu_operation, musa_operation in [
        ("fbank", audio_16k, cpu_fbank, musa_fbank),
        ("mel", audio_22k, mel, mel),
    ]:
        for case, signal in [("real", audio), ("silence", torch.zeros_like(audio)),
                             ("random", torch.randn_like(audio) * 0.1)]:
            metrics = error_metrics(musa_operation(signal.musa()), cpu_operation(signal))
            report["precision"][f"{label}_{case}"] = metrics
            print(label, case, metrics, flush=True)
            if not metrics["pass_atol_2e_3_rtol_1e_4"]:
                raise AssertionError(f"{label} {case} precision failed")

    cpu_flush = torch.zeros(128 * 1024 * 1024 // 4)
    device_flush = torch.zeros(256 * 1024 * 1024 // 4, device="musa")
    for seconds in (3, 15):
        audio16 = audio_16k.repeat(1, 1 + seconds * 16000 // audio_16k.size(1))[:, :seconds * 16000].contiguous()
        audio22 = audio_22k.repeat(1, 1 + seconds * 22050 // audio_22k.size(1))[:, :seconds * 22050].contiguous()
        device16, device22 = audio16.musa(), audio22.musa()
        window = torch.hann_window(1024)
        device_window = window.musa()
        operations = {
            "fbank_resident": (lambda: cpu_fbank(audio16), lambda: musa_fbank(device16)),
            "fbank_h2d": (lambda: cpu_fbank(audio16), lambda: musa_fbank(audio16.musa())),
            "mel_resident": (lambda: mel(audio22), lambda: mel(device22)),
            "mel_h2d": (lambda: mel(audio22), lambda: mel(audio22.musa())),
            "stft_resident": (
                lambda: torch.stft(audio22, 1024, 256, window=window, center=False, return_complex=True),
                lambda: real_stft(device22, device_window, 256),
            ),
        }
        for label, (cpu_call, musa_call) in operations.items():
            result = benchmark_pair(cpu_call, musa_call, args.rounds, cpu_flush, device_flush)
            report["benchmark"][f"{label}_{seconds}s"] = result
            print(label, seconds, {key: value for key, value in result.items() if not isinstance(value, list)}, flush=True)

    with torch.profiler.profile(
        activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.MUSA],
        record_shapes=True,
    ) as profiler:
        musa_fbank(device16)
        mel(device22)
        torch.musa.synchronize()
    report["preprocess_kernel_counts"] = save_gpu_trace(profiler, output / "preprocess.gpu.json")
    (output / "results.json").write_text(json.dumps(report, indent=2))
    if args.demo:
        from indextts.infer_v2_5 import IndexTTS2

        model = IndexTTS2(
            cfg_path=str(Path(args.model_dir) / "config.yaml"), model_dir=args.model_dir,
            device="musa:0", use_bf16=True, use_cuda_kernel=False, use_qwen_emo=False,
        )
        with torch.profiler.profile(
            activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.MUSA],
            record_shapes=True,
        ) as profiler:
            model.infer(
                spk_audio_prompt=args.prompt, text="欢迎体验 MUSA 算子适配。", lang="ZH",
                output_path=str(output / "demo.wav"), verbose=True,
            )
            torch.musa.synchronize()
        report["demo_kernel_counts"] = save_gpu_trace(profiler, output / "demo.gpu.json")
        if not any("1024" in name for name in report["demo_kernel_counts"]):
            raise AssertionError("demo did not execute the native 1024-point STFT")
        if not any("512_kernel" in name for name in report["demo_kernel_counts"]):
            raise AssertionError("demo did not execute the native 512-point FFT")
        (output / "results.json").write_text(json.dumps(report, indent=2))
        print("demo_kernel_counts", report["demo_kernel_counts"], flush=True)


if __name__ == "__main__":
    main()
