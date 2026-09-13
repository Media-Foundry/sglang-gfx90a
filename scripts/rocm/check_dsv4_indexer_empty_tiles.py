"""Standalone empty-tile oracle; does not modify the serving implementation.

The candidate calls the original Triton kernel as a device function for every
nonempty tile, and writes the original zero result for an empty tile. All
patching is local to this test process, during eager launch/graph capture.
Do not run concurrently with a service benchmark.
"""
import argparse
import json
import math
import os
from pathlib import Path
import statistics
from unittest.mock import patch

os.environ.setdefault('SGLANG_OPT_USE_TRITON_INDEXER_FULL', '1')
os.environ.setdefault('SGLANG_DSV4_GFX90A_CANONICAL_INDEXER_ORDER', '3')

import torch
import triton
import triton.language as tl

from sglang.kernels.ops.attention.dsv4.topk import topk_transform_512
from sglang.kernels.ops.kvcache.triton_store_cache import triton_fused_store_indexer
from sglang.srt.layers.attention.dsv4 import indexer

_control_kernel = indexer._fp8_paged_mqa_logits_kernel


@triton.jit
def _empty_tile_candidate(
    q_u8, kvcache_u8, weights, seq_lens, page_table, out,
    max_num_pages: tl.constexpr, max_seq_len: tl.constexpr,
    num_heads: tl.constexpr,
    q_stride_b: tl.constexpr, q_stride_h: tl.constexpr, q_stride_d: tl.constexpr,
    kv_stride_page: tl.constexpr,
    weights_stride_b: tl.constexpr, weights_stride_h: tl.constexpr,
    page_table_stride_b: tl.constexpr, page_table_stride_p: tl.constexpr,
    out_stride_b: tl.constexpr, out_stride_s: tl.constexpr,
    FP8_TY: tl.constexpr, DOT_TY: tl.constexpr, PRESHUFFLE_TILE: tl.constexpr,
    BLOCK_S: tl.constexpr, BLOCK_H: tl.constexpr, BLOCK_D: tl.constexpr,
    TRIVIAL_TOPK: tl.constexpr,
    SKIP_EMPTY_TILES: tl.constexpr = False,
):
    row = tl.program_id(0)
    tile = tl.program_id(1)
    length = tl.load(seq_lens + row)
    if tile * BLOCK_S < length:
        _control_kernel(
            q_u8, kvcache_u8, weights, seq_lens, page_table, out,
            max_num_pages, max_seq_len, num_heads,
            q_stride_b, q_stride_h, q_stride_d, kv_stride_page,
            weights_stride_b, weights_stride_h,
            page_table_stride_b, page_table_stride_p, out_stride_b, out_stride_s,
            FP8_TY, DOT_TY, PRESHUFFLE_TILE, BLOCK_S, BLOCK_H, BLOCK_D, TRIVIAL_TOPK,
        )
    else:
        columns = tile * BLOCK_S + tl.arange(0, BLOCK_S)
        tl.store(out + row * out_stride_b + columns * out_stride_s,
                 0.0, mask=columns < max_seq_len)


def run_case(batch, width, live_width, skip, args):
    device = 'cuda'
    pages_n = math.ceil(live_width / 64)
    table_pages = math.ceil(width / 64)
    cache = torch.empty(pages_n, 64 * 132, device=device, dtype=torch.uint8)
    values = torch.randn(pages_n * 64, 128, device=device)
    locations = torch.arange(values.shape[0], device=device, dtype=torch.int32)
    q = torch.randn(batch, 1, 64, 128, device=device).to(indexer.FP8_DTYPE)
    weights = torch.randn(batch, 64, device=device)
    lengths = torch.full((batch,), live_width, device=device, dtype=torch.int32)
    pages = torch.zeros(batch, table_pages, device=device, dtype=torch.int32)
    pages[:, :pages_n] = torch.arange(pages_n, device=device, dtype=torch.int32)
    triton_fused_store_indexer(values, cache, locations, 64)

    def stage():
        scores = indexer._fp8_paged_mqa_logits_triton(
            q, cache.view(pages_n, 64, 1, 132), weights, lengths, pages, width, skip)
        assert scores is not None
        logical = torch.empty(batch, 512, device=device, dtype=torch.int32)
        physical = torch.empty_like(logical)
        topk_transform_512(scores, lengths, pages, physical, 64, logical)
        return scores, logical, physical

    outputs, graphs = {}, {}
    for name, kernel in (('A', _control_kernel), ('B', _empty_tile_candidate)):
        with patch.object(indexer, '_fp8_paged_mqa_logits_kernel', kernel):
            stage()
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                outputs[name] = stage()
            graphs[name] = graph

    def check(label):
        a, b = outputs['A'], outputs['B']
        assert torch.equal(a[0].view(torch.int32), b[0].view(torch.int32)), ('scores', label)
        assert torch.equal(a[1], b[1]), ('logical Top-K', label)
        assert torch.equal(a[2], b[2]), ('physical Top-K', label)

    for graph in graphs.values():
        for _ in range(args.replays):
            graph.replay()
    check('fixed graph replay')
    times = []
    for cycle in range(args.abba_cycles):
        for name in ('A', 'B', 'B', 'A'):
            begin, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            begin.record()
            for _ in range(args.timing_iterations):
                graphs[name].replay()
            end.record()
            end.synchronize()
            times.append(dict(cycle=cycle, arm=name,
                              stage_us=begin.elapsed_time(end)*1000/args.timing_iterations))

    boundaries = [0, 1, 15, 16, 17, 511, 512, 513, live_width]
    # Deterministic boundary/tie fixtures, even for the batch-one case.
    for n in boundaries:
        lengths.fill_(min(n, live_width))
        q.zero_()
        for graph in graphs.values():
            graph.replay()
        check(('boundary/tie', n))
    for mutation in range(args.mutations):
        q.copy_(torch.randn(q.shape, device=device).to(indexer.FP8_DTYPE))
        weights.normal_()
        values.normal_()
        triton_fused_store_indexer(values, cache, locations, 64)
        lengths.random_(0, live_width+1)
        pages[:, :pages_n] = torch.randperm(pages_n, device=device, dtype=torch.int32)
        for graph in graphs.values():
            graph.replay()
        check(('mutation', mutation))
    medians = {name: statistics.median(t['stage_us'] for t in times if t['arm'] == name)
               for name in ('A', 'B')}
    return dict(batch=batch, width=width, live_width=live_width, trivial_topk=skip,
                score_bits_exact=True, logical_and_physical_exact=True,
                mutations=args.mutations, fixed_graph_replays=args.replays,
                abba=times, medians_us=medians,
                stage_speedup=medians['A']/medians['B'])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--mutations', type=int, default=100)
    parser.add_argument('--replays', type=int, default=1000)
    parser.add_argument('--timing-iterations', type=int, default=20)
    parser.add_argument('--abba-cycles', type=int, default=2)
    parser.add_argument('--screen', action='store_true', help='One B64 case first')
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    assert min(args.mutations, args.replays, args.timing_iterations, args.abba_cycles) > 0
    assert not args.output.exists()
    assert torch.version.hip and 'gfx90a' in torch.cuda.get_device_properties(0).gcnArchName
    torch.manual_seed(20260914)
    cases = [(64, 262144, 640, 0)] if args.screen else [
        (1, 262144, 640, 0), (32, 262144, 640, 0), (64, 262144, 640, 0),
        (4, 577, 577, 0), (4, 8192, 8192, 0), (1, 262144, 262144, 0),
        (32, 576, 576, 512)]
    result = dict(status='running', production_modified=False, cases=[])
    for case in cases:
        row = run_case(*case, args)
        result['cases'].append(row)
        args.output.write_text(json.dumps(result, indent=2)+'\n')
        print(json.dumps(row), flush=True)
    result['status'] = 'complete'
    args.output.write_text(json.dumps(result, indent=2)+'\n')


if __name__ == '__main__':
    main()
