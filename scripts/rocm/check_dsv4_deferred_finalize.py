#!/usr/bin/env python3
"""Check full grouped-down deferred ownership and existing default semantics."""
import argparse
import json
from pathlib import Path

import torch

from scripts.rocm.bench_dsv4_tp4_m32_noa2a_ep2_oracle import full_metadata
from sglang.kernels.ops.moe.gfx90a_fp4_expert_gemv import gfx90a_fp4_expert_down_grouped


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    assert 'gfx90a' in torch.cuda.get_device_properties(0).gcnArchName
    torch.manual_seed(2090802)
    ids = torch.stack([torch.randperm(256, device='cuda')[:6] for _ in range(32)]).int()
    meta = full_metadata(ids)
    w = torch.randint(0, 256, (256, 4096, 128), device='cuda', dtype=torch.uint8)
    # Nonconstant physical scales exercise the exact same production indexing.
    ws = torch.randint(120, 129, (256, 4096, 8), device='cuda', dtype=torch.uint8)
    x = torch.randint(-127, 128, (32, 6, 256), device='cuda', dtype=torch.int8)
    xs = torch.rand(32, 6, 8, device='cuda') * .01
    tw = torch.softmax(torch.randn(32, 6, device='cuda'), dim=-1)
    shared = torch.randn(32, 4096, device='cuda', dtype=torch.bfloat16)
    outputs = [torch.empty_like(shared) for _ in range(4)]

    def run(out, deferred=False):
        return gfx90a_fp4_expert_down_grouped(
            x, xs, w, ws, meta.sorted_ids, meta.sorted_experts, meta.valid, tw,
            out=out, assignments=4, rows=2, waves=8, blocks=832,
            use_lds_lut=True, defer_reduction=deferred,
        )

    assert run(outputs[0]) is outputs[0]
    run(outputs[1], True).finalize(shared)
    torch.cuda.synchronize()
    graphs = {}
    for name in ('default', 'deferred'):
        g = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g):
            if name == 'default':
                run(outputs[0]).add_(shared)
            else:
                handle = run(outputs[1], True)
                handle.finalize(shared)
        graphs[name] = g
    exact = 0
    for iteration in range(100):
        x.random_(-127, 128)
        xs.uniform_(.0001, .01)
        shared.normal_()
        tw.copy_(torch.softmax(torch.randn_like(tw), dim=-1))
        if iteration % 10 == 0:
            ws.random_(120, 129)
            # Permute expert owners without changing A4 padding/slot addresses.
            permutation = torch.randperm(256, device='cuda').int()
            meta.sorted_experts.copy_(permutation[meta.sorted_experts.long()])
        for g in graphs.values():
            g.replay()
        torch.cuda.synchronize()
        assert torch.isfinite(outputs[0]).all()
        exact += int(torch.equal(outputs[0].view(torch.int16), outputs[1].view(torch.int16)))
    assert exact == 100, exact
    # An outstanding first call must survive another call and an input mutation.
    first = run(outputs[2], True)
    ref_first = run(outputs[0]).add_(shared).clone()
    x.random_(-127, 128)
    second = run(outputs[3], True)
    assert first.partial.data_ptr() != second.partial.data_ptr()
    ref_second = run(outputs[1]).add_(shared).clone()
    second.finalize(shared)
    first.finalize(shared)
    torch.cuda.synchronize()
    assert torch.equal(outputs[2].view(torch.int16), ref_first.view(torch.int16))
    assert torch.equal(outputs[3].view(torch.int16), ref_second.view(torch.int16))
    graphs['deferred'].replay()
    stable = outputs[1].clone()
    allocated = torch.cuda.memory_allocated()
    for _ in range(1000):
        graphs['deferred'].replay()
    torch.cuda.synchronize()
    assert torch.equal(stable.view(torch.int16), outputs[1].view(torch.int16))
    assert torch.cuda.memory_allocated() == allocated
    # Mirror the model's existing fork/join, leaving down before the join.
    # The shared producer is only a stand-in, not the model's shared GEMM.
    side = torch.cuda.Stream()
    current = torch.cuda.current_stream()
    shared_outputs = [torch.empty_like(shared), torch.empty_like(shared)]
    def stream_call(which):
        side.wait_stream(current)
        value = run(outputs[which], which == 1)
        with torch.cuda.stream(side):
            torch.mul(shared, 2.0, out=shared_outputs[which])
        current.wait_stream(side)
        if which == 1:
            value.finalize(shared_outputs[which])
        else:
            value.add_(shared_outputs[which])

    stream_graphs = []
    for which in (0, 1):
        stream_call(which)
        torch.cuda.synchronize()
        g = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g):
            stream_call(which)
        stream_graphs.append(g)
    stream_exact = 0
    for _ in range(100):
        x.random_(-127, 128)
        shared.normal_()
        for g in stream_graphs:
            g.replay()
        torch.cuda.synchronize()
        stream_exact += int(torch.equal(outputs[0].view(torch.int16), outputs[1].view(torch.int16)))
    assert stream_exact == 100
    stable = outputs[1].clone()
    allocated = torch.cuda.memory_allocated()
    for _ in range(1000):
        stream_graphs[1].replay()
    torch.cuda.synchronize()
    assert torch.equal(stable.view(torch.int16), outputs[1].view(torch.int16))
    assert torch.cuda.memory_allocated() == allocated
    result = dict(mutations_raw_bits_exact=exact, interleaved_calls_exact=True,
        independent_partials=True, replay1000_exact=True, allocation_stable=True,
        cross_stream_mutations_exact=stream_exact, cross_stream_replay1000_exact=True,
        partial_bytes=first.partial.numel() * first.partial.element_size(),
        scope='full grouped down; synthetic shared producer; no service selector/E2E')
    args.output.write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result))


if __name__ == '__main__':
    main()
