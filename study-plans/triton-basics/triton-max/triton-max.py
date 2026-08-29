import torch
import triton
import triton.language as tl


@triton.jit
def max_kernel(x_ptr, out_ptr, n, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    pointer = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)

    # load X and Y from global memory to registers
    X = tl.load(x_ptr + pointer, mask=pointer < n, other=float('-inf')) 
    
    # store to global memory
    tl.store(out_ptr, tl.max(X))

def solve(x: torch.Tensor, out: torch.Tensor) -> None:
    """Launch max_kernel on the provided tensor with a single-program reduction."""
    n = x.numel()
    BLOCK_SIZE = triton.next_power_of_2(n)
    grid = (1,)
    max_kernel[grid](x, out, n, BLOCK_SIZE=BLOCK_SIZE)