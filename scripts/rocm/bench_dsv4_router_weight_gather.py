#!/usr/bin/env python3
"""Isolate deferred weight gather in the current sqrtsoftplus Top-6 router."""
import argparse
import json
import statistics
from pathlib import Path

import torch
import triton
import triton.language as tl
from sglang.kernels.ops.moe.moe_fused_gate import _router_triton_kernel


@triton.jit
def candidate(scores, bias, weights, ids, M, scale, APPLY: tl.constexpr,
              ARGMAX: tl.constexpr, PACKED: tl.constexpr):
    row = tl.program_id(0) + tl.arange(0, 1)
    n = tl.arange(0, 256)
    logits = tl.load(scores + row[:, None] * 256 + n[None, :],
                     mask=row[:, None] < M, other=0.).to(tl.float32)
    b = tl.load(bias + n).to(tl.float32)
    sp = tl.where(logits > 20., logits, tl.log(1. + tl.exp(logits)))
    activated = tl.sqrt(sp)
    cur = activated + b[None, :]
    cur = tl.where(cur == cur, cur, -1.e30)
    k = tl.arange(0, 8)
    selected = tl.zeros((1, 8), tl.int32)
    for slot in tl.static_range(6):
        if PACKED:
            # Canonicalize signed zero so float equality retains lowest-ID ties.
            ranked = tl.where(cur == 0., 0., cur)
            bits = ranked.to(tl.uint32, bitcast=True)
            key = bits ^ tl.where((bits & 0x80000000) != 0, 0xFFFFFFFF, 0x80000000).to(tl.uint32)
            pair = (key.to(tl.uint64) << 32) | (255 - n[None, :]).to(tl.uint64)
            best = tl.max(pair, axis=1)[:, None]
            winner = (255 - (best & 0xFFFFFFFF)).to(tl.int32)
        elif ARGMAX:
            winner = tl.argmax(cur, axis=1, tie_break_left=True)[:, None].to(tl.int32)
        else:
            maximum = tl.max(cur, axis=1)[:, None]
            winners = tl.where(cur == maximum, n[None, :], 257)
            winner = tl.min(winners, axis=1)[:, None].to(tl.int32)
        selected = tl.where(k[None, :] == slot, winner, selected)
        cur = tl.where(n[None, :] == winner, -float('inf'), cur)
    values = tl.gather(activated, selected, axis=1)
    values = tl.where(k[None, :] < 6, values, 0.)
    total = tl.sum(values, axis=1)[:, None]
    values = values / tl.where(total > 0., total, 1.)
    if APPLY:
        values = values * scale
    mask = (row[:, None] < M) & (k[None, :] < 6)
    tl.store(weights + row[:, None] * 6 + k[None, :], values, mask)
    tl.store(ids + row[:, None] * 6 + k[None, :], selected, mask)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--argmax', action='store_true')
    p.add_argument('--packed', action='store_true')
    a = p.parse_args()
    assert not (a.argmax and a.packed)
    assert 'gfx90a' in torch.cuda.get_device_properties(0).gcnArchName
    torch.manual_seed(2090804)
    results = []
    for m in (1, 32):
        for apply in (False, True):
            scores = torch.randn(m, 256, device='cuda', dtype=torch.bfloat16)
            bias = torch.randn(256, device='cuda', dtype=torch.bfloat16)
            aw = torch.empty(m, 6, device='cuda')
            ai = torch.empty(m, 6, device='cuda', dtype=torch.int32)
            bw, bi = torch.empty_like(aw), torch.empty_like(ai)
            def old():
                _router_triton_kernel[(m,)](
                    scores, bias, aw, ai, m, 1.5, 0., N=256, K=6, K_ROUTED=6,
                    BLOCK_M=1, BLOCK_N=256, BLOCK_K=8, N_GROUP=1, TOPK_GROUP=1,
                    EXPERTS_PER_GROUP=256, BLOCK_G=1, SCORING_FUNC=1,
                    HAS_SOFTCAP=False, RENORMALIZE=True, APPLY_SCALE=apply,
                    USE_PDL=False, stride_sm=256, stride_sn=1,
                    stride_wm=6, stride_wk=1, stride_im=6, stride_ik=1, num_warps=1)
            def new():
                candidate[(m,)](scores, bias, bw, bi, m, 1.5, APPLY=apply,
                               ARGMAX=a.argmax, PACKED=a.packed, num_warps=1)
            funcs = {'A': old, 'B': new}
            graphs = {}
            for name, fn in funcs.items():
                fn()
                torch.cuda.synchronize()
                g = torch.cuda.CUDAGraph()
                with torch.cuda.graph(g):
                    fn()
                graphs[name] = g
            exact_ids = exact_weights = 0
            max_abs = 0.
            for i in range(100):
                scores.normal_().mul_(1 + i % 20)
                bias.normal_()
                if i % 10 == 0:
                    scores.fill_(0 if i % 20 == 0 else -120)
                    bias.zero_()
                for g in graphs.values():
                    g.replay()
                torch.cuda.synchronize()
                exact_ids += int(torch.equal(ai, bi))
                exact_weights += int(torch.equal(aw.view(torch.int32), bw.view(torch.int32)))
                max_abs = max(max_abs, float((aw-bw).abs().max()))
            assert exact_ids == 100
            stable = bw.clone()
            allocated = torch.cuda.memory_allocated()
            for _ in range(1000):
                graphs['B'].replay()
            torch.cuda.synchronize()
            assert torch.equal(stable.view(torch.int32), bw.view(torch.int32))
            assert allocated == torch.cuda.memory_allocated()
            bursts = {}
            for name, fn in funcs.items():
                g = torch.cuda.CUDAGraph()
                with torch.cuda.graph(g):
                    for _ in range(100):
                        fn()
                bursts[name] = g
            samples = {name: [] for name in funcs}
            for name in ['A', 'B', 'B', 'A'] * 5:
                g = bursts[name]
                for _ in range(3):
                    g.replay()
                start, end = (torch.cuda.Event(enable_timing=True) for _ in range(2))
                start.record()
                g.replay()
                end.record()
                end.synchronize()
                samples[name].append(start.elapsed_time(end)*10)
            result = dict(m=m, apply_scale=apply, argmax=a.argmax, packed=a.packed, ids_exact=exact_ids,
                weights_bits_exact=exact_weights, max_abs=max_abs,
                replay1000_exact=True, allocation_stable=True, samples_us=samples,
                trimmed_us={k: statistics.mean(sorted(v)[1:-1]) for k, v in samples.items()})
            results.append(result)
            a.output.write_text(json.dumps(results, indent=2)+'\n')
            print(json.dumps(result), flush=True)


if __name__ == '__main__':
    main()
