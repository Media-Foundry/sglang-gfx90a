"""Independent V4 post->K1024 projection partial prototype, not dispatched."""
import triton
import triton.language as tl


@triton.jit
def post_project(x,residual,post,comb,fn,out,squares,projections):
    row=tl.program_id(0);block=tl.program_id(1)
    h=block*1024+tl.arange(0,1024)
    xv=tl.load(x+row*4096+h).to(tl.float32)
    base=residual+row*16384+h
    r0=tl.load(base).to(tl.float32)
    r1=tl.load(base+4096).to(tl.float32)
    r2=tl.load(base+8192).to(tl.float32)
    r3=tl.load(base+12288).to(tl.float32)
    for hc in tl.static_range(4):
        pv=tl.load(post+row*4+hc)
        cb=comb+row*16+hc
        c0=tl.load(cb);c1=tl.load(cb+4);c2=tl.load(cb+8);c3=tl.load(cb+12)
        acc=pv*xv
        acc+=c0*r0+c1*r1+c2*r2+c3*r3
        rounded=acc.to(tl.bfloat16).to(tl.float32)
        tl.store(out+row*16384+hc*4096+h,rounded)
        square=tl.sum(tl.reshape(rounded*rounded,(4,256)),axis=1)
        tl.store(squares+row*64+hc*16+block*4+tl.arange(0,4),square)
        # Process columns serially, not24 simultaneously live accumulators.
        for column in range(24):
            weight=tl.load(fn+column*16384+hc*4096+h)
            dot=tl.sum(weight*rounded,axis=0)
            tl.store(projections+(row*16+hc*4+block)*24+column,dot)


@triton.jit
def finish(projections,squares,mixes):
    row=tl.program_id(0);column=tl.program_id(1)
    accum=tl.full((),0,tl.float32)
    for chunk in tl.static_range(16):
        accum+=tl.load(projections+(row*16+chunk)*24+column)
    sq=tl.sum(tl.load(squares+row*64+tl.arange(0,64)),axis=0)
    tl.store(mixes+row*24+column,accum*tl.rsqrt(sq/16384+1e-6))
