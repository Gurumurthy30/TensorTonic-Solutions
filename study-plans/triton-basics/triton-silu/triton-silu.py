import torch
import triton
import triton.language as tl


@triton.jit
def silu_kernel(x_ptr, out_ptr, n, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    pointer = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    
    # load X and Y from global memory to registers
    X = tl.load(x_ptr + pointer, mask=pointer < n, other=0.0)

    # SiLU formula
    result = X * tl.sigmoid(X)
    
    # store to global memory
    tl.store(out_ptr + pointer, result, mask=pointer < n)


def solve(x: torch.Tensor, out: torch.Tensor) -> None:
    """Launch silu_kernel: out = x / (1 + exp(-x))."""
    n = x.numel()
    BLOCK_SIZE = 1024
    grid = ((n + BLOCK_SIZE - 1) // BLOCK_SIZE,)
    silu_kernel[grid](x, out, n, BLOCK_SIZE=BLOCK_SIZE)