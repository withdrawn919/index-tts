import pytest
import torch

pytest.importorskip("torch_musa")
pytest.importorskip("triton")

from indextts.musa_ops.cosine import cosine_argmax


pytestmark = pytest.mark.skipif(not torch.musa.is_available(), reason="MUSA required")


@pytest.mark.parametrize("rows", [3, 17, 2, 8, 4, 5, 10, 24])
@pytest.mark.parametrize("kind", ["random", "zero", "tie"])
def test_cosine_argmax_cpu_baseline(rows, kind):
    torch.manual_seed(rows)
    query = torch.randn(1, 192)
    matrix = torch.randn(rows, 192)
    if kind == "zero":
        query.zero_()
    elif kind == "tie":
        matrix[:] = matrix[:1].clone()
    expected = torch.nn.functional.cosine_similarity(query, matrix, dim=1).argmax()
    actual = cosine_argmax(query.musa(), matrix.musa()).cpu()
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
