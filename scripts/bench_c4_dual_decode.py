#!/usr/bin/env python3
import struct
import statistics

import torch

from sglang.kernels.ops.attention.dsv4.c4_dual_decode import c4_dual_decode
from sglang.kernels.ops.attention.dsv4.compress import (
    CompressorDecodePlan,
    compress_forward,
    compress_norm_rope_store,
)
from sglang.srt.layers.attention.dsa.utils import (
    aiter_can_use_preshuffle_paged_mqa,
)


def plan_tensor(seq_lens, device):
    raw = b"".join(
        struct.pack("<Iiii", s, i * 4 + (s & 3), i, i) for i, s in enumerate(seq_lens)
    )
    return torch.tensor(list(raw), dtype=torch.uint8, device=device).view(-1, 16)


def main():
    assert aiter_can_use_preshuffle_paged_mqa(), (
        "This production-layout oracle requires SGLANG_USE_AITER=1 and the "
        "AIter page-64 preshuffle path."
    )
    torch.manual_seed(7)
    dev = "cuda"
    m = 32
    seq = [8 + (i & 3) for i in range(m)]
    plan = plan_tensor(seq, dev)
    pd = CompressorDecodePlan(4, plan)
    # One private pair of state pages per row; arbitrary state is sufficient for
    # an operator equivalence test because both paths consume the same PlanD.
    cs = torch.randn((m + 1, 4, 2048), device=dev)
    ins = torch.randn((m + 1, 4, 512), device=dev)
    ci = torch.randn((m, 2048), device=dev)
    ii = torch.randn((m, 512), device=dev)
    ca = torch.randn((8, 512), device=dev)
    ia = torch.randn((8, 128), device=dev)
    cn = torch.randn((512,), device=dev)
    inn = torch.randn((128,), device=dev)
    freqs = torch.randn((32, 64), device=dev)
    index_loc = torch.arange(m, dtype=torch.int64, device=dev)
    # Production Unified-KV lives in a separate address domain and applies a
    # fixed page offset.  Keep the locations deliberately different here so
    # the operator oracle catches accidental aliasing of the two metadata
    # streams.
    core_loc = index_loc + 17
    cc = torch.full((m + 18, 1024), 0xA5, dtype=torch.uint8, device=dev)
    ic = torch.full((1, 8448), 0xA5, dtype=torch.uint8, device=dev)

    rcs, ris, rcc, ric = cs.clone(), ins.clone(), cc.clone(), ic.clone()
    co = compress_forward(rcs, ci, ca, pd, head_dim=512, compress_ratio=4)
    io = compress_forward(ris, ii, ia, pd, head_dim=128, compress_ratio=4)
    fc = torch.view_as_complex(freqs.view(32, 32, 2))
    compress_norm_rope_store(co, pd, norm_weight=cn, norm_eps=1e-6, freq_cis=fc, out_loc=core_loc, kvcache=rcc, page_size=1, bf16_store=True)
    compress_norm_rope_store(io, pd, norm_weight=inn, norm_eps=1e-6, freq_cis=fc, out_loc=index_loc, kvcache=ric, page_size=64)

    ctmp = torch.empty((m, 512), device=dev)
    itmp = torch.empty((m, 128), device=dev)
    c4_dual_decode(cs, ci, ca, ins, ii, ia, plan, cn, inn, freqs, freqs, core_loc, index_loc, cc, ic, ctmp, itmp, 1e-6, 1e-6)
    torch.cuda.synchronize()
    checks = (
        ("core_tmp", ctmp, co),
        ("index_tmp", itmp, io),
        ("core_state", cs, rcs),
        ("index_state", ins, ris),
        ("core_cache", cc, rcc),
        ("index_cache", ic, ric),
    )
    for name, a, b in checks:
        exact = torch.equal(a, b)
        max_byte = int(
            (a.view(torch.uint8).to(torch.int16) - b.view(torch.uint8).to(torch.int16))
            .abs()
            .max()
        )
        print(f"{name:12s} exact={exact} max_byte_delta={max_byte}")
        assert exact, name

    s0, s1 = torch.cuda.Stream(), torch.cuda.Stream()
    def ref_serial():
        x = compress_forward(rcs, ci, ca, pd, head_dim=512, compress_ratio=4, out=co)
        compress_norm_rope_store(x, pd, norm_weight=cn, norm_eps=1e-6, freq_cis=fc, out_loc=core_loc, kvcache=rcc, page_size=1, bf16_store=True)
        x = compress_forward(ris, ii, ia, pd, head_dim=128, compress_ratio=4, out=io)
        compress_norm_rope_store(x, pd, norm_weight=inn, norm_eps=1e-6, freq_cis=fc, out_loc=index_loc, kvcache=ric, page_size=64)

    def ref_parallel():
        s0.wait_stream(torch.cuda.current_stream()); s1.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(s0):
            x=compress_forward(rcs,ci,ca,pd,head_dim=512,compress_ratio=4,out=co)
            compress_norm_rope_store(x,pd,norm_weight=cn,norm_eps=1e-6,freq_cis=fc,out_loc=core_loc,kvcache=rcc,page_size=1,bf16_store=True)
        with torch.cuda.stream(s1):
            x=compress_forward(ris,ii,ia,pd,head_dim=128,compress_ratio=4,out=io)
            compress_norm_rope_store(x,pd,norm_weight=inn,norm_eps=1e-6,freq_cis=fc,out_loc=index_loc,kvcache=ric,page_size=64)
        torch.cuda.current_stream().wait_stream(s0); torch.cuda.current_stream().wait_stream(s1)
    def dual(): c4_dual_decode(cs,ci,ca,ins,ii,ia,plan,cn,inn,freqs,freqs,core_loc,index_loc,cc,ic,ctmp,itmp,1e-6,1e-6)
    for f in (ref_serial, ref_parallel, dual):
        for _ in range(20): f()
    def tm(f):
        a,b=torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True); a.record()
        for _ in range(200): f()
        b.record(); b.synchronize(); return a.elapsed_time(b)*1000/200
    samples = {f.__name__: [] for f in (ref_serial, ref_parallel, dual)}
    for _ in range(8):
        for f in (ref_serial, ref_parallel, dual, dual, ref_parallel, ref_serial):
            samples[f.__name__].append(tm(f))

    def trimmed(values):
        ordered = sorted(values)
        return statistics.mean(ordered[2:-2])

    for name, values in samples.items():
        print(
            f"{name:12s} median_us={statistics.median(values):.3f} "
            f"trimmed_us={trimmed(values):.3f} "
            f"range=[{min(values):.3f},{max(values):.3f}]"
        )
    dual_us = trimmed(samples["dual"])
    serial_saved = trimmed(samples["ref_serial"]) - dual_us
    parallel_saved = trimmed(samples["ref_parallel"]) - dual_us
    print(f"saved_vs_serial_us={serial_saved:.3f}")
    print(f"saved_vs_parallel_us={parallel_saved:.3f}")
    # The conservative gate is the ordinary single-stream path; the explicit
    # two-stream Python reference includes stream-event/join overhead.
    print(f"stop_gate_ge_20us={serial_saved >= 20.0}")


if __name__ == "__main__":
    main()
