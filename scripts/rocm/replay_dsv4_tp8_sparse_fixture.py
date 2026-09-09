"""Replay an eager H8 witness on one otherwise idle GCD (not an E2E benchmark)."""
import argparse
import json
import statistics

import torch

from sglang.kernels.ops.attention.dsv4.gfx90a_sparse_h8 import run_if_supported
from sglang.kernels.ops.attention.dsv4.unified_kv_kernels.paged_decode import (
    _sparse_attn_v4_paged_decode_triton,
)


def error_metrics(actual, reference):
    a, b = actual.float(), reference.float()
    return dict(max_abs=float((a-b).abs().max()),
                relative_l2=float(torch.linalg.vector_norm(a-b)
                                  / torch.linalg.vector_norm(b).clamp_min(1e-20)),
                exact=bool(torch.equal(a, b)))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('fixture')
    parser.add_argument('--output', required=True)
    parser.add_argument('--refined-probabilities', action='store_true',
                        help='isolated BF16 probability hi+lo PV oracle; no production changes')
    parser.add_argument('--paired-refined-probabilities', action='store_true',
                        help='Q2xH8 C128 prefix-pair oracle; no production changes')
    parser.add_argument('--rounds', type=int, default=1)
    parser.add_argument('--mutations', type=int, default=0,
                        help='vary Q/KV/indices/sink and ragged/empty rows against FP32')
    args = parser.parse_args()
    assert args.rounds > 0
    assert args.mutations >= 0
    selected_ck = run_if_supported
    if args.refined_probabilities and args.paired_refined_probabilities:
        parser.error('select at most one refined oracle')
    if args.refined_probabilities:
        from sglang.kernels.ops.debug.gfx90a_sparse_h8_refined_probability import run_if_supported as refined
        selected_ck = refined
    if args.paired_refined_probabilities:
        from sglang.kernels.ops.debug.gfx90a_sparse_h8_refined_probability import run_pair_if_supported
        selected_ck = run_pair_if_supported
    data = torch.load(args.fixture, map_location='cpu', weights_only=True)
    assert data['format'] == 'dsv4_tp8_sparse_fixture_v1'
    names = ('q', 'unified_kv', 'kv_indices', 'kv_indptr', 'attn_sink')
    q, kv, indices, ptr, sink = [data[n].cuda().contiguous() for n in names]
    scale = data['softmax_scale']
    pair_contract = None
    if args.paired_refined_probabilities:
        if data['provenance'].get('compress_ratio') != 128:
            raise ValueError('Q2xH8 oracle accepts only C128 fixtures')
        host_indices = data['kv_indices'].reshape(-1)
        host_ptr_check = data['kv_indptr'].tolist()
        mismatches = []
        split_mismatches = []
        for row in range(0, 128, 2):
            a = host_indices[host_ptr_check[row]:host_ptr_check[row + 1]]
            b = host_indices[host_ptr_check[row + 1]:host_ptr_check[row + 2]]
            if a.numel() > b.numel() or not torch.equal(a, b[:a.numel()]):
                mismatches.append(row // 2)
            if (a.numel() + 15) // 16 != (b.numel() + 15) // 16:
                split_mismatches.append(row // 2)
        if mismatches or split_mismatches:
            raise ValueError(
                f'Q2xH8 contract failed: prefix={mismatches[:8]} '
                f'tile_count={split_mismatches[:8]}')
        pair_contract = dict(prefix_pairs=64, equal_tile_count_pairs=64)
    # Preserve empty-row semantics while satisfying the non-null pointer ABI.
    if not indices.numel():
        indices = torch.zeros(1, dtype=torch.int32, device=q.device)
    if not kv.shape[0]:
        kv = torch.zeros((1, 512), dtype=torch.bfloat16, device=q.device)

    def ck():
        out = selected_ck(q, kv, indices, ptr, sink, scale)
        assert out is not None, 'fixture did not satisfy actual H8 wrapper'
        return out

    def triton():
        return _sparse_attn_v4_paged_decode_triton(q, kv, indices, ptr, sink, scale)

    a, b = triton(), ck()
    host_ptr = data['kv_indptr'].tolist()
    def fp32_reference(offsets):
        result = torch.empty_like(q, dtype=torch.float32)
        for row in range(128):
            slots = indices[offsets[row]:offsets[row+1]].long()
            keys = kv[slots].float()
            scores = q[row].float() @ keys.T * scale
            probs = torch.softmax(torch.cat((scores, sink[:, None]), dim=1), dim=1)
            result[row] = probs[:, :-1] @ keys
        return result
    reference = fp32_reference(host_ptr)
    report = dict(provenance=data['provenance'],
                  refined_probabilities=args.refined_probabilities,
                  paired_refined_probabilities=args.paired_refined_probabilities,
                  pair_contract=pair_contract,
                  q_abs_max=float(q.abs().max()), kv_abs_max=float(kv.abs().max()),
                  ck_vs_fp32=error_metrics(b, reference),
                  triton_vs_fp32=error_metrics(a, reference),
                  ck_vs_triton=error_metrics(b, a),
                  triton_vs_capture=error_metrics(a, data['baseline_output'].cuda()))
    if args.refined_probabilities or args.paired_refined_probabilities:
        original = run_if_supported(q, kv, indices, ptr, sink, scale)
        report['original_ck_vs_fp32'] = error_metrics(original, reference)
        report['refined_vs_original_ck'] = error_metrics(b, original)
    # Preserve diagnostics even on a numerical failure; no timings are accepted.
    report['numerical_pass'] = all(torch.allclose(x.float(), reference, atol=.004, rtol=.02)
                                   for x in (a, b))
    if not report['numerical_pass']:
        with open(args.output, 'x') as f:
            json.dump(report, f, indent=2)
        raise AssertionError(report)

    graphs, outputs = [], []
    for fn in (triton, ck):
        for _ in range(3):
            fn()
        torch.cuda.synchronize()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            out = fn()
        graphs.append(graph)
        outputs.append(out)
    for graph in graphs:
        graph.replay()
    expected = [x.clone() for x in outputs]
    for _ in range(1000):
        for graph in graphs:
            graph.replay()
    torch.cuda.synchronize()
    assert all(torch.equal(x, y) for x, y in zip(outputs, expected)), 'stale graph output'
    report['replay_exact'] = True

    if args.mutations:
        originals = [x.clone() for x in (q, kv, indices, ptr, sink)]
        torch.manual_seed(20909)
        worst = [0., 0.]
        for trial in range(args.mutations):
            q.copy_((originals[0].float() * (0.75 + trial % 5 * .125)).to(q.dtype))
            kv.copy_((originals[1].roll(trial % 17, 0).float()
                      * (-1 if trial % 3 == 0 else 1)).to(kv.dtype))
            indices.copy_((originals[2] + trial * 37) % kv.shape[0])
            sink.copy_(originals[4] + (trial % 7 - 3) * .25)
            offsets = list(host_ptr)
            if trial % 10 == 0:
                offsets = [0] * len(offsets)
            elif trial % 3 == 0:
                # Empty alternating rows; neighbors span two original rows.
                for row in range(0, 128, 2):
                    offsets[row+1] = offsets[row]
            ptr.copy_(torch.tensor(offsets, device=ptr.device, dtype=ptr.dtype))
            for graph in graphs:
                graph.replay()
            expected_fp32 = fp32_reference(offsets)
            checks = [bool(torch.allclose(x.float(), expected_fp32, atol=.004, rtol=.02))
                      for x in outputs]
            mutation_metrics = [error_metrics(x, expected_fp32) for x in outputs]
            for arm, metric in enumerate(mutation_metrics):
                worst[arm] = max(worst[arm], metric['relative_l2'])
            replay_refs = [x.clone() for x in outputs]
            for _ in range(10):
                for graph in graphs:
                    graph.replay()
            stable = all(torch.equal(x, y) for x, y in zip(outputs, replay_refs))
            if not all(checks) or not stable:
                report['mutation_failure'] = dict(trial=trial, numerical=checks,
                                                   stable=stable, metrics=mutation_metrics)
                with open(args.output, 'x') as f:
                    json.dump(report, f, indent=2)
                raise AssertionError(report['mutation_failure'])
        report['mutations'] = dict(count=args.mutations, passed=True,
                                   worst_relative_l2=worst, repeat_replays_per_input=10)
        for tensor, saved in zip((q, kv, indices, ptr, sink), originals):
            tensor.copy_(saved)
        for graph in graphs:
            graph.replay()
        torch.cuda.synchronize()

    def timing(graph):
        start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        start.record()
        for _ in range(100):
            graph.replay()
        end.record()
        end.synchronize()
        return start.elapsed_time(end) * 10

    blocks = []
    for _ in range(args.rounds):
        blocks.append([timing(graphs[arm]) for arm in (0, 1, 1, 0)])
    report['abba_us'] = blocks
    report['triton_median_us'] = statistics.median(v for b in blocks for v in (b[0], b[3]))
    report['ck_median_us'] = statistics.median(v for b in blocks for v in (b[1], b[2]))
    with open(args.output, 'x') as f:
        json.dump(report, f, indent=2)
    print(json.dumps(report), flush=True)


if __name__ == '__main__':
    main()
