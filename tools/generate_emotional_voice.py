import argparse
import json
import os
import random
import subprocess
import sys
from pathlib import Path

import librosa
import numpy as np
import soundfile as sf
import torch
import torch_musa

from indextts.infer_v2_5 import IndexTTS2
from indextts.utils.profiling import inference_trace


def main():
    parser = argparse.ArgumentParser(description="Generate emotional speech, not melody-conditioned singing")
    parser.add_argument("--model-dir", default="checkpoints")
    parser.add_argument("--prompt-wav", default="examples/voice_01.wav")
    parser.add_argument("--text-file", required=True)
    parser.add_argument("--emotion-text", required=True)
    parser.add_argument("--emotion-runtime", required=True,
                        help="Directory containing the isolated Transformers runtime for QwenEmotion")
    parser.add_argument("--emotion-weight", type=float, default=0.8)
    parser.add_argument("--duration-factor", type=float, default=1.15)
    parser.add_argument("--seconds", type=float, default=30.0)
    parser.add_argument("--output", required=True)
    parser.add_argument("--trace")
    parser.add_argument("--seed", type=int, default=2026)
    args = parser.parse_args()
    if not 0.5 <= args.duration_factor <= 2.0 or not 0.0 < args.seconds:
        parser.error("duration-factor must be in [0.5, 2.0]; seconds must be positive")
    if not 0.0 <= args.emotion_weight <= 1.0:
        parser.error("emotion-weight must be in [0, 1]")
    output = Path(args.output)
    raw_path = output.with_name(output.stem + ".raw.wav")
    report_path = output.with_suffix(".json")
    if any(path.exists() for path in (output, raw_path, report_path)):
        parser.error("output already exists; choose a new filename")
    text = Path(args.text_file).read_text().strip()
    if not text:
        parser.error("text-file is empty")
    output.parent.mkdir(parents=True, exist_ok=True)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.musa.manual_seed_all(args.seed)
    emotion_path = output.with_suffix(".emotion.json")
    environment = os.environ.copy()
    environment["PYTHONPATH"] = os.pathsep.join([
        args.emotion_runtime, str(Path(__file__).resolve().parents[1]),
        environment.get("PYTHONPATH", ""),
    ])
    subprocess.run([
        sys.executable, str(Path(__file__).with_name("classify_musa_emotion.py")),
        "--model-dir", str(Path(args.model_dir) / "qwen0.6bemo4-merge"),
        "--emotion-text", args.emotion_text, "--output", str(emotion_path),
    ], env=environment, check=True)
    emotion_result = json.loads(emotion_path.read_text())
    model = IndexTTS2(
        cfg_path=str(Path(args.model_dir) / "config.yaml"), model_dir=args.model_dir,
        device="musa:0", use_bf16=True, use_cuda_kernel=False, use_qwen_emo=False,
    )
    print("QwenEmotion device:", emotion_result["devices"], flush=True)
    with inference_trace("musa:0", args.trace):
        model.infer(
            spk_audio_prompt=args.prompt_wav, text=text, lang="ZH", output_path=str(raw_path),
            emo_vector=emotion_result["vector"], emo_alpha=args.emotion_weight,
            use_random=False, duration_factor=args.duration_factor,
            do_sample=False, num_beams=1, verbose=True,
        )
    waveform, sample_rate = sf.read(raw_path, dtype="float32")
    original_seconds = len(waveform) / sample_rate
    rate = original_seconds / args.seconds
    if not 0.5 <= rate <= 2.0:
        raise RuntimeError(f"Raw duration {original_seconds:.3f}s requires excessive stretching; raw audio retained")
    adjusted = librosa.effects.time_stretch(waveform, rate=rate)
    target_samples = round(args.seconds * sample_rate)
    adjusted = librosa.util.fix_length(adjusted, size=target_samples)
    if not np.isfinite(adjusted).all():
        raise RuntimeError("Non-finite output after duration adjustment")
    sf.write(output, adjusted, sample_rate, subtype="PCM_16")
    report = {
        "type": "emotional spoken lyrics; not a singing model; no accompaniment",
        "model_dir": args.model_dir, "reference_audio": str(Path(args.prompt_wav).resolve()),
        "text": text, "emotion_text": args.emotion_text, "emotion_weight": args.emotion_weight,
        "qwen_devices": emotion_result["devices"], "emotion_result": emotion_result,
        "duration_factor": args.duration_factor,
        "raw_seconds": original_seconds, "final_seconds": target_samples / sample_rate,
        "sample_rate": sample_rate, "postprocess": "CPU librosa pitch-preserving time_stretch",
        "time_stretch_rate": rate, "raw_audio": str(raw_path), "output": str(output),
    }
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2))
    print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
