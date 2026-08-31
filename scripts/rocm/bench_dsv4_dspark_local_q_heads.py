#!/usr/bin/env python3
"""Oracle for removing legacy 64-head padding from TP4 DSpark attention."""

import argparse
import statistics

import torch

from sglang.kernels.ops.attention.dsv4.unified_kv_kernels.paged_decode import (
    _sparse_attn_v4_paged_decode_triton,
)


def capture(fn):
    graph = torch.cuda.CUDAGraph()
    pool = torch.cuda.graph_pool_handle()
    with torch.cuda.graph(graph, pool=pool):
        out = fn()
    torch.cuda.synchronize()
    return graph, out, pool


def time_graph(graph, warmup: int, iterations: int) -> float:
    for _ in range(warmup):
        graph.replay()
    torch.cuda.synchronize()
    begin = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    begin.record()
    for _ in range(iterations):
        graph.replay()
    end.record()
    end.synchronize()
    return begin.elapsed_time(end) * 1000.0 / iterations


def check_stable_replay(graph, out, replays: int, label: str) -> None:
    # The output produced while capture is being recorded is not a replay
    # oracle: allocator/kernel initialization may still be part of that first
    # execution.  Establish the reference from the first completed replay and
    # require every subsequent replay to reproduce it bitwise.
    graph.replay()
    torch.cuda.synchronize()
    expected = out.clone()
    for replay in range(replays):
        graph.replay()
        torch.cuda.synchronize()
        if not torch.equal(expected, out):
            delta = (expected.float() - out.float()).abs()
            raise RuntimeError(
                f"{label} graph replay={replay + 1} changed output "
                f"max_abs={delta.max().item()} changed={torch.count_nonzero(delta).item()}"
            )


def trimmed(values):
    return statistics.mean(sorted(values)[1:-1])


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tokens", type=int, default=32)
    parser.add_argument("--contexts", default="128,256,512")
    parser.add_argument("--mutations", type=int, default=100)
    parser.add_argument("--graph-replays", type=int, default=1000)
    parser.add_argument("--rounds", type=int, default=7)
    parser.add_argument("--iterations", type=int, default=100)
    args = parser.parse_args()

    torch.manual_seed(20260831)
    tokens, local_heads, padded_heads, dim = args.tokens, 16, 64, 512
    for context in (int(value) for value in args.contexts.split(",")):
        q_local = torch.randn(
            (tokens, local_heads, dim), dtype=torch.bfloat16, device="cuda"
        )
        q_padded = torch.randn(
            (tokens, padded_heads, dim), dtype=torch.bfloat16, device="cuda"
        )
        kv = torch.randn(
            (tokens * context, dim), dtype=torch.bfloat16, device="cuda"
        )
        indices = torch.arange(tokens * context, dtype=torch.int32, device="cuda")
        indptr = torch.arange(
            0, (tokens + 1) * context, context, dtype=torch.int32, device="cuda"
        )
        sink_padded = torch.randn(
            (padded_heads,), dtype=torch.float32, device="cuda"
        )
        scale = 1.0 / (dim**0.5)

        def local_run():
            return _sparse_attn_v4_paged_decode_triton(
                q_local,
                kv,
                indices,
                indptr,
                sink_padded[:local_heads],
                scale,
                _oracle_num_warps=4,
                _oracle_num_stages=2,
            )

        def padded_run():
            return _sparse_attn_v4_paged_decode_triton(
                q_padded,
                kv,
                indices,
                indptr,
                sink_padded,
                scale,
                _oracle_num_warps=4,
                _oracle_num_stages=2,
            )

        for mutation in range(args.mutations):
            q_local.normal_()
            q_padded[:, :local_heads].copy_(q_local)
            local_out = local_run()
            padded_out = padded_run()
            torch.cuda.synchronize()
            if not torch.equal(local_out, padded_out[:, :local_heads]):
                delta = (
                    local_out.float() - padded_out[:, :local_heads].float()
                ).abs().max().item()
                raise RuntimeError(
                    f"ctx={context} mutation={mutation} mismatch max_abs={delta}"
                )
        print(f"CORRECT ctx={context} mutations={args.mutations} bitwise_exact=True")

        local_graph, local_out, local_pool = capture(local_run)
        check_stable_replay(
            local_graph,
            local_out,
            args.graph_replays,
            f"ctx={context} local",
        )

        padded_graph, padded_out, padded_pool = capture(padded_run)
        check_stable_replay(
            padded_graph,
            padded_out[:, :local_heads],
            args.graph_replays,
            f"ctx={context} padded",
        )
        print(
            f"GRAPH ctx={context} replays={args.graph_replays} bitwise_exact=True"
        )

        values = {"padded": [], "local": []}
        for _ in range(args.rounds):
            for name in ("padded", "local", "local", "padded"):
                graph = padded_graph if name == "padded" else local_graph
                values[name].append(time_graph(graph, 10, args.iterations))
        padded_us = trimmed(values["padded"])
        local_us = trimmed(values["local"])
        print(
            f"RESULT ctx={context} padded_us={padded_us:.3f} "
            f"local_us={local_us:.3f} saving_us={padded_us-local_us:.3f} "
            f"gain_pct={(padded_us/local_us-1)*100:.3f}"
        )


if __name__ == "__main__":
    main()
