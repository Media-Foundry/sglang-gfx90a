"""Four token rows share FP32 Fn loads; preserve individual K1024 sums."""
import os
import torch
import triton
import triton.language as tl

@triton.jit
def _premix_reuse_kernel(x,fn,partials,out,M,BM:tl.constexpr,EPS:tl.constexpr=1e-6):
    row=tl.program_id(1)*BM
    n=tl.program_id(0)+tl.arange(0,1)
    acc0=tl.zeros((1,),tl.float32)
    acc1=tl.zeros((1,),tl.float32)
    if BM==4:
        acc2=tl.zeros((1,),tl.float32)
        acc3=tl.zeros((1,),tl.float32)
    for start in tl.static_range(0,16384,1024):
        k=start+tl.arange(0,1024)
        w=tl.load(fn+n[:,None]*16384+k[None,:],mask=n[:,None]<24,other=0.)
        a0=tl.load(x+row*16384+k).to(tl.float32)
        a1=tl.load(x+(row+1)*16384+k,mask=row+1<M,other=0.).to(tl.float32)
        acc0+=tl.sum(w*a0[None,:],axis=1)
        acc1+=tl.sum(w*a1[None,:],axis=1)
        if BM==4:
            a2=tl.load(x+(row+2)*16384+k,mask=row+2<M,other=0.).to(tl.float32)
            a3=tl.load(x+(row+3)*16384+k,mask=row+3<M,other=0.).to(tl.float32)
            acc2+=tl.sum(w*a2[None,:],axis=1)
            acc3+=tl.sum(w*a3[None,:],axis=1)
    p=tl.arange(0,64)
    sq0=tl.sum(tl.load(partials+row*64+p),axis=0)
    sq1=tl.sum(tl.load(partials+(row+1)*64+p,mask=row+1<M,other=0.),axis=0)
    tl.store(out+row*24+n,acc0*tl.rsqrt(sq0/16384+EPS),mask=n<24)
    tl.store(out+(row+1)*24+n,acc1*tl.rsqrt(sq1/16384+EPS),mask=(n<24)&(row+1<M))
    if BM==4:
        sq2=tl.sum(tl.load(partials+(row+2)*64+p,mask=row+2<M,other=0.),axis=0)
        sq3=tl.sum(tl.load(partials+(row+3)*64+p,mask=row+3<M,other=0.),axis=0)
        tl.store(out+(row+2)*24+n,acc2*tl.rsqrt(sq2/16384+EPS),mask=(n<24)&(row+2<M))
        tl.store(out+(row+3)*24+n,acc3*tl.rsqrt(sq3/16384+EPS),mask=(n<24)&(row+3<M))


def premix_reuse4(residual, fn, rms_partials, rms_eps, *, group_size=4, pair_columns=False):
    m = residual.shape[0]
    tensors = (residual, fn, rms_partials)
    if not (
        m > 0 and residual.shape == (m, 4, 4096)
        and residual.dtype == torch.bfloat16
        and fn.shape == (24, 16384) and fn.dtype == torch.float32
        and rms_partials.shape == (m, 64) and rms_partials.dtype == torch.float32
        and residual.device.type == "cuda" and bool(torch.version.hip)
        and all(t.device == residual.device and t.is_contiguous() for t in tensors)
        and "gfx90a" in torch.cuda.get_device_properties(residual.device).gcnArchName
    ):
        return None
    out = torch.empty((m, 1, 24), dtype=torch.float32, device=residual.device)
    if group_size == 8 and 8192 <= m <= 65536:
        if pair_columns:
            if os.getenv("SGLANG_DSV4_DEBUG_PREFILL_MIX_MFMA", "0") == "1":
                from sglang.kernels.ops.layernorm.gfx90a_mhc_premix_mfma import try_premix_mfma

                if try_premix_mfma(residual, fn, rms_partials, out, rms_eps):
                    return out
            from sglang.kernels.ops.layernorm.gfx90a_mhc_premix_pair import premix8_pair

            premix8_pair[(12, triton.cdiv(m, 8))](
                residual, fn, rms_partials, out, m, float(rms_eps), num_warps=1
            )
            return out
        from sglang.kernels.ops.layernorm.gfx90a_mhc_premix_reuse8 import premix8

        premix8[(24, triton.cdiv(m, 8))](
            residual, fn, rms_partials, out, m, float(rms_eps), num_warps=1
        )
        return out
    _premix_reuse_kernel[(24, triton.cdiv(m, 4))](
        residual, fn, rms_partials, out, m, 4, float(rms_eps), num_warps=1
    )
    return out
