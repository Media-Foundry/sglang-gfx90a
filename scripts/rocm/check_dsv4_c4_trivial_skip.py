"""Exact active-score/Top-K oracle and GPU-stage ABBA for trivial C4 rows."""
import argparse
import json
import os

os.environ.setdefault('SGLANG_OPT_USE_TRITON_INDEXER_FULL', '1')
os.environ.setdefault('SGLANG_DSV4_GFX90A_CANONICAL_INDEXER_ORDER', '3')
import torch
from sglang.kernels.ops.kvcache.triton_store_cache import triton_fused_store_indexer
from sglang.kernels.ops.attention.dsv4.topk import topk_transform_512
from sglang.srt.layers.attention.dsv4.indexer import FP8_DTYPE, _fp8_paged_mqa_logits_triton


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--mutations', type=int, default=100)
    p.add_argument('--replays', type=int, default=1000)
    args = p.parse_args()
    torch.manual_seed(2709)
    for m, width in ((32, 576), (2304, 576), (32, 4096)):
        pages_n = (width + 63) // 64
        cache = torch.empty(pages_n, 64 * 132, device='cuda', dtype=torch.uint8)
        values = torch.randn(pages_n * 64, 128, device='cuda')
        loc = torch.arange(values.shape[0], device='cuda', dtype=torch.int32)
        q = torch.randn(m, 1, 64, 128, device='cuda').to(FP8_DTYPE)
        w = torch.randn(m, 64, device='cuda')
        lens = ((torch.arange(m, device='cuda', dtype=torch.int32) + 1) // 4).clamp_max(width)
        if m == 32:
            lens.copy_(torch.tensor([0, 1, 511, 512, 513, width] * 6, device='cuda', dtype=torch.int32)[:m])
        pages = torch.arange(pages_n, device='cuda', dtype=torch.int32)[None, :].expand(m, -1).contiguous()
        triton_fused_store_indexer(values, cache, loc, 64)
        graphs, outputs = {}, {}
        def run(skip):
            scores = _fp8_paged_mqa_logits_triton(q, cache.view(pages_n,64,1,132), w, lens, pages, width, skip)
            assert scores is not None
            raw = torch.empty(m,512,device='cuda',dtype=torch.int32)
            physical = torch.empty_like(raw)
            topk_transform_512(scores,lens,pages,physical,64,raw)
            return scores, raw, physical
        for skip in (0, 512):
            run(skip)
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                outputs[skip] = run(skip)
            graphs[skip] = graph
        def check():
            a,b = outputs[0],outputs[512]
            active = lens > 512
            assert torch.equal(a[0][active],b[0][active]), 'active score difference'
            assert torch.equal(a[1],b[1]), 'logical membership/order difference'
            assert torch.equal(a[2],b[2]), 'physical index difference'
            assert torch.count_nonzero(b[0][~active]).item() == 0
        for _ in range(args.replays):
            for graph in graphs.values(): graph.replay()
        check()
        samples = []
        for skip in (0,512,512,0):
            begin,end = torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True)
            begin.record()
            for _ in range(100): graphs[skip].replay()
            end.record();end.synchronize()
            samples.append({'skip':skip,'stage_us':begin.elapsed_time(end)*10})
        for iteration in range(args.mutations):
            q.copy_(torch.randn(q.shape,device='cuda').to(FP8_DTYPE))
            w.normal_();values.normal_()
            triton_fused_store_indexer(values,cache,loc,64)
            lens.random_(0,width+1)
            pages.copy_(torch.randperm(pages_n,device='cuda',dtype=torch.int32)[None,:])
            for graph in graphs.values():graph.replay()
            check()
        print(json.dumps({'M':m,'C4_width':width,'graph_replays':args.replays,
            'mutations':args.mutations,'exact':True,'ABBA':samples}),flush=True)


if __name__ == '__main__':
    main()
