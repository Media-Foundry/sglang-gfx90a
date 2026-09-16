"""Grouped H256 post tiles: preserve each RMS partial, no pre-mix fusion."""
import triton
import triton.language as tl

@triton.jit
def grouped_post(x,residual,post,comb,out,partials,G:tl.constexpr):
    token=tl.program_id(0)
    blocks=tl.program_id(1)*G+tl.arange(0,G)
    h=blocks[:,None]*256+tl.arange(0,256)[None,:]
    xv=tl.load(x+token*4096+h).to(tl.float32)
    base=residual+token*16384+h
    r0=tl.load(base).to(tl.float32)
    r1=tl.load(base+4096).to(tl.float32)
    r2=tl.load(base+8192).to(tl.float32)
    r3=tl.load(base+12288).to(tl.float32)
    for hc in tl.static_range(4):
        pv=tl.load(post+token*4+hc)
        cb=comb+token*16+hc
        c0=tl.load(cb);c1=tl.load(cb+4);c2=tl.load(cb+8);c3=tl.load(cb+12)
        acc=pv*xv
        acc+=c0*r0+c1*r1+c2*r2+c3*r3
        tl.store(out+token*16384+hc*4096+h,acc)
        rounded=acc.to(tl.bfloat16).to(tl.float32)
        sq=tl.sum(rounded*rounded,1)
        tl.store(partials+(token*4+hc)*16+blocks,sq)
