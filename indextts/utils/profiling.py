from contextlib import contextmanager
from pathlib import Path

import torch


@contextmanager
def inference_trace(device, output_path=None):
    if output_path is None:
        yield
        return

    backend = torch.device(device).type
    activities = [torch.profiler.ProfilerActivity.CPU]
    if backend == "musa":
        activities.append(torch.profiler.ProfilerActivity.MUSA)
    elif backend == "cuda":
        activities.append(torch.profiler.ProfilerActivity.CUDA)
    elif backend != "cpu":
        raise ValueError(f"Trace collection is not configured for {backend}")
    destination = Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with torch.profiler.profile(activities=activities, record_shapes=True) as profiler:
        with torch.profiler.record_function("IndexTTS2.infer"):
            yield
        if backend in ("musa", "cuda"):
            getattr(torch, backend).synchronize(device)
    profiler.export_chrome_trace(str(destination))
