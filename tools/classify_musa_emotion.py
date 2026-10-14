import argparse
import json
from pathlib import Path

import torch
import torch_musa
import transformers

from indextts.infer_v2_5 import QwenEmotion


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-dir", required=True)
    parser.add_argument("--emotion-text", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    torch.backends.mudnn.allow_tf32 = False
    model = QwenEmotion(args.model_dir, device="musa:0")
    devices = sorted({str(parameter.device) for parameter in model.model.parameters()})
    if devices != ["musa:0"]:
        raise RuntimeError(f"Unexpected emotion model devices: {devices}")
    emotions = model.inference(args.emotion_text, max_new_tokens=512)
    result = {
        "transformers": transformers.__version__, "devices": devices,
        "prompt": args.emotion_text, "emotions": emotions, "vector": list(emotions.values()),
    }
    Path(args.output).write_text(json.dumps(result, ensure_ascii=False, indent=2))
    print(json.dumps(result, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
