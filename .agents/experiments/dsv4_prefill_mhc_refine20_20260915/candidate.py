"""Experimental continuation of Sinkhorn8 output to20; not production-wired."""
import triton
import triton.language as tl


@triton.jit
def refine12(comb, eps:tl.constexpr):
    row=tl.program_id(0)
    offsets=tl.arange(0,16)
    matrix=tl.reshape(tl.load(comb+row*16+offsets),(4,4))
    # Initial normalization +7 repeats was already performed by the8 tail.
    # Do not repeat exp/initialization: add exactly12 row/column pairs.
    for _ in tl.static_range(12):
        matrix=matrix/(tl.sum(matrix,axis=1)[:,None]+eps)
        matrix=matrix/(tl.sum(matrix,axis=0)[None,:]+eps)
    tl.store(comb+row*16+offsets,tl.reshape(matrix,(16,)))
