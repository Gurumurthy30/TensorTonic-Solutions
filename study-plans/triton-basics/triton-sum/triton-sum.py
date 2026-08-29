import torch
import triton
import triton.language as tl


@triton.jit
def sum_kernel(x_ptr, out_ptr, n, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0) 
    pointer = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE) 

    # load X and Y from global memory to registers
    X = tl.load(x_ptr + pointer, mask=pointer < n, other=0.0)  
    
    # store to global memory
    tl.atomic_add(out_ptr, tl.sum(X))


def solve(x: torch.Tensor, out: torch.Tensor) -> None:
    """Launch sum_kernel on the provided tensors."""
    n = x.numel()
    out.zero_()
    BLOCK_SIZE = 1024
    grid = ((n + BLOCK_SIZE - 1) // BLOCK_SIZE,)
    sum_kernel[grid](x, out, n, BLOCK_SIZE=BLOCK_SIZE)