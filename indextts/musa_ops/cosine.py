import torch

import triton
import triton.language as tl


@triton.jit
def _cosine_argmax_kernel(query, matrix, output, ROWS: tl.constexpr, WIDTH: tl.constexpr,
                          BLOCK_ROWS: tl.constexpr, BLOCK_WIDTH: tl.constexpr):
    rows = tl.arange(0, BLOCK_ROWS)
    columns = tl.arange(0, BLOCK_WIDTH)
    query_values = tl.load(query + columns, columns < WIDTH, 0)
    matrix_values = tl.load(
        matrix + rows[:, None] * WIDTH + columns[None, :],
        (rows[:, None] < ROWS) & (columns[None, :] < WIDTH), 0,
    )
    query_norm = tl.maximum(tl.sqrt(tl.sum(query_values * query_values, 0)), 1.0e-8)
    matrix_norm = tl.maximum(tl.sqrt(tl.sum(matrix_values * matrix_values, 1)), 1.0e-8)
    scores = tl.sum((query_values / query_norm)[None, :] * (matrix_values / matrix_norm[:, None]), 1)
    scores = tl.where(rows < ROWS, scores, -float("inf"))
    index = tl.argmax(scores, axis=0, tie_break_left=True)
    tl.store(output, index.to(tl.int64))


def cosine_argmax(query, matrix):
    if query.device.type != "musa" or matrix.device != query.device:
        raise ValueError("cosine_argmax requires MUSA tensors on the same device")
    if matrix.ndim != 2 or not 1 <= matrix.shape[0] <= 32 or not 1 <= matrix.shape[1] <= 256:
        raise ValueError("cosine_argmax supports 1..32 rows and 1..256 columns")
    if query.shape != (1, matrix.shape[1]):
        raise ValueError("cosine_argmax expects a single query row")
    query = query.float().contiguous()
    matrix = matrix.float().contiguous()
    output = torch.empty((), dtype=torch.int64, device=query.device)
    with torch.musa.device(query.device):
        _cosine_argmax_kernel[(1,)](
            query, matrix, output, matrix.shape[0], matrix.shape[1],
            triton.next_power_of_2(matrix.shape[0]), triton.next_power_of_2(matrix.shape[1]),
            num_warps=4, enable_fp_fusion=False,
        )
    return output
