import torch
import triton
import triton.language as tl


@triton.jit
def relu_kernel(x_ptr, out_ptr, n, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    pointer = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    
    # load X and Y from global memory to registers
    X = tl.load(x_ptr + pointer, mask=pointer < n, other=0.0)

    # store to global memory
    result = tl.where(X < 0, 0.0, X)
    tl.store(out_ptr + pointer, result, mask=pointer < n)

def solve(x: torch.Tensor, out: torch.Tensor) -> None:
    """Launch relu_kernel: out = max(x, 0)."""
    n = x.numel()
    BLOCK_SIZE = 1024
    grid = ((n + BLOCK_SIZE - 1) // BLOCK_SIZE,)
    relu_kernel[grid](x, out, n, BLOCK_SIZE=BLOCK_SIZE)