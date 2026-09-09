"""Isolated H8 MFMA port; run only when the selected GPU is otherwise idle."""
import argparse
import json
import statistics

import torch
from torch.utils.cpp_extension import include_paths
from sglang.kernels.jit.utils import load_jit
from sglang.kernels.ops.attention.dsv4.unified_kv_kernels.paged_decode import (
    _sparse_attn_v4_paged_decode_triton,
)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    torch.manual_seed(909)
    module = load_jit(
        'gfx90a_dsv4_sparse_h8_oracle_v1',
        cuda_files=['deepseek_v4/gfx90a_dsv4_sparse_h8_oracle.cuh'],
        cuda_wrappers=[('run', 'sglang::Gfx90aDsv4SparseH8Oracle::run'),
                       ('run_h16', 'sglang::Gfx90aDsv4UnifiedSparseDecode::run')],
        extra_cuda_cflags=['-O3', '-std=c++20', '-DCK_ENABLE_BF16', '-DCK_USE_XDL'],
        extra_include_paths=[*include_paths(),
            '/home/pc/pytorch/third_party/aiter/3rdparty/composable_kernel/include',
            '/home/pc/pytorch/third_party/aiter/3rdparty/composable_kernel/library/include'],
    )
    results = []
    for length in (0, 17, 128, 512):
        q = torch.randn((128, 8, 512), device='cuda', dtype=torch.bfloat16) * .25
        kv = torch.randn((1024, 512), device='cuda', dtype=torch.bfloat16)
        sink = torch.randn((8,), device='cuda', dtype=torch.float32)
        lengths = [length if i % 4 else length // 2 for i in range(128)]
        ptr = [0]
        for n in lengths:
            ptr.append(ptr[-1] + n)
        indptr = torch.tensor(ptr, device='cuda', dtype=torch.int32)
        # Empty rows still need a non-null index buffer for the launch ABI;
        # indptr remains all-zero, so the sentinel is never consumed.
        indices = torch.randint(1024, (max(1, ptr[-1]),), device='cuda', dtype=torch.int32)
        out = torch.empty_like(q)
        scratch = torch.empty(128*2*8*514*4, device='cuda', dtype=torch.uint8)
        def run():
            module.run(q, kv, indices, indptr, sink, out, scratch, 512**-.5)
        max_error = 0.
        for mutation in range(10):
            q.normal_(0, .25)
            sink.normal_()
            indices.random_(0, 1024)
            run()
            for row, n in enumerate(lengths):
                if not n:
                    assert torch.count_nonzero(out[row]).item() == 0
                    continue
                keys = kv[indices[ptr[row]:ptr[row+1]].long()].float()
                logits = q[row].float() @ keys.T * (512**-.5)
                probs = torch.softmax(torch.cat([logits, sink[:, None]], dim=1), dim=1)[:, :n]
                reference = probs @ keys
                error = (out[row].float()-reference).abs().max().item()
                max_error = max(max_error, error)
                torch.testing.assert_close(out[row].float(), reference, atol=.004, rtol=.02)
        for _ in range(3):
            run()
        # Same MFMA math on the first eight heads of a full H16 tensor.
        # Both exports compile from current headers, not an old cached module.
        q16 = torch.cat([q, torch.randn_like(q)], dim=1).contiguous()
        sink16 = torch.cat([sink, torch.randn_like(sink)])
        out16 = torch.empty_like(q16)
        scratch16 = torch.empty(128*2*16*514*4, device='cuda', dtype=torch.uint8)
        module.run_h16(q16, kv, indices, indptr, sink16, out16, scratch16, 512**-.5)
        assert torch.equal(out, out16[:, :8]), 'H8 differs from corresponding H16 heads'
        torch.cuda.synchronize()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            run()
        for _ in range(10):
            q.normal_(0, .25)
            sink.normal_()
            indices.random_(0, 1024)
            run()
            eager = out.clone()
            graph.replay()
            assert torch.equal(out, eager), 'mutated graph/eager mismatch'
        graph.replay()
        expected = out.clone()
        for _ in range(1000):
            graph.replay()
        torch.cuda.synchronize()
        assert torch.equal(out, expected), 'graph replay mismatch'
        timings = []
        for _ in range(5):
            begin, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            begin.record()
            for _ in range(100):
                graph.replay()
            end.record()
            end.synchronize()
            timings.append(begin.elapsed_time(end)*10)
        results.append(dict(length=length, median_us=statistics.median(timings),
                            max_abs_error=max_error, replay_exact=True))
        def baseline():
            return _sparse_attn_v4_paged_decode_triton(
                q, kv, indices, indptr, sink, 512**-.5)
        for _ in range(3):
            baseline()
        torch.cuda.synchronize()
        control = torch.cuda.CUDAGraph()
        with torch.cuda.graph(control):
            baseline_out = baseline()
        control.replay()
        torch.testing.assert_close(out.float(), baseline_out.float(), atol=.004, rtol=.02)
        control_times = []
        for _ in range(5):
            begin, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            begin.record()
            for _ in range(100):
                control.replay()
            end.record()
            end.synchronize()
            control_times.append(begin.elapsed_time(end)*10)
        results[-1]['triton_median_us'] = statistics.median(control_times)
        print(results[-1], flush=True)
    with open(args.output, 'x') as f:
        json.dump(results, f, indent=2)


if __name__ == '__main__':
    main()
