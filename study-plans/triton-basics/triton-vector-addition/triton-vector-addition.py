import torch
import triton
import triton.language as tl


@triton.jit
def vector_add_kernel(x_ptr, y_ptr, out_ptr, n, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    pointer = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    
    # load X and Y from global memory to registers
    X = tl.load(x_ptr + pointer, mask=pointer < n, other=0.0)
    Y = tl.load(y_ptr + pointer, mask=pointer < n, other=0.0)
    
    # save the result to global memory
    tl.store(out_ptr + pointer, X + Y, mask=pointer < n)


def solve(x: torch.Tensor, y: torch.Tensor, out: torch.Tensor) -> None:
    """Launch vector_add_kernel on the provided tensors."""
    n = x.numel()
    BLOCK_SIZE = 1024
    grid = ((n + BLOCK_SIZE - 1) // BLOCK_SIZE,)
    vector_add_kernel[grid](x, y, out, n, BLOCK_SIZE=BLOCK_SIZE)