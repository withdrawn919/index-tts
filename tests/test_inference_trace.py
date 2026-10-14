import json

import torch

from indextts.utils.profiling import inference_trace


def test_trace_retains_cpu_operations(tmp_path):
    destination = tmp_path / "trace" / "cpu.json"
    with inference_trace("cpu", destination):
        torch.ones(4, 4) @ torch.ones(4, 4)
    events = json.loads(destination.read_text())["traceEvents"]
    assert any(event.get("cat") == "cpu_op" and event["name"] == "aten::mm" for event in events)
    assert any(event["name"] == "IndexTTS2.infer" for event in events)


def test_trace_disabled():
    with inference_trace("cpu"):
        output = torch.ones(2) + 1
    torch.testing.assert_close(output, torch.full((2,), 2.0))
