#!/usr/bin/env python3
"""Isolated V4.1 indexer score workspace/numerics screen, not service timing."""

import argparse
import json
import statistics
from pathlib import Path

import torch

from sglang.srt.layers.attention.dsv4.dsv41_sparse import (
    DeepseekV41Indexer,
    select_candidate_blocks,
    stable_index_topk,
)
from sglang.srt.layers.attention.dsv4.torch_quant import fake_quant_fp4


def bounded_scores(q, k, weights, budget):
    """Prototype: retain BF16 operations and head sum, bound the QK slab."""
    rows = max(1, budget // max(1, q.shape[1] * len(k) * q.element_size()))
    out = torch.empty((len(q), len(k)), dtype=torch.float32, device=q.device)
    for begin in range(0, len(q), rows):
        stop = min(begin + rows, len(q))
        slab = torch.einsum("bhd,nd->bhn", q[begin:stop], k)
        slab.relu_().mul_(weights[begin:stop].unsqueeze(-1))
        out[begin:stop].copy_(slab.sum(dim=1))
    return out


def reference_scores(q, k, weights):
    # Frozen pre-tiling serving arithmetic, including each BF16 rounding point.
    slab = torch.einsum("bhd,nd->bhn", q, k)
    return (slab.relu() * weights.unsqueeze(-1)).sum(dim=1).float()


def measure(fn, rounds):
    warm = fn()
    torch.cuda.synchronize()
    del warm
    torch.cuda.empty_cache()
    times, peaks = [], []
    for _ in range(rounds):
        base = torch.cuda.memory_allocated()
        torch.cuda.reset_peak_memory_stats()
        start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        start.record()
        output = fn()
        end.record()
        end.synchronize()
        times.append(start.elapsed_time(end))
        peaks.append(torch.cuda.max_memory_allocated() - base)
        del output
    return {"median_ms": statistics.median(times), "times_ms": times,
            "peak_increment_bytes": max(peaks)}, fn()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--m", type=int, default=2304)
    p.add_argument("--lengths", default="8192,16384,32768")
    p.add_argument("--rounds", type=int, default=3)
    p.add_argument("--slab-mib", type=int, default=128)
    p.add_argument("--production", action="store_true",
                   help="Compare serving scores() instead of the isolated prototype")
    args = p.parse_args()
    if args.production and args.slab_mib != 128:
        p.error("serving uses the fixed128MiB budget; --slab-mib is prototype-only")
    lengths = [int(x) for x in args.lengths.split(",")]
    assert args.m > 0 and min(lengths) >= 512 and args.rounds > 0 and args.slab_mib > 0
    report = {"passed": False, "kind": "isolated synthetic FP4-grid inputs",
              "production": args.production, "rows": args.m,
              "slab_mib": args.slab_mib, "shapes": []}
    with args.output.open("x") as handle:
        try:
            report["gpu"] = torch.cuda.get_device_properties(0).gcnArchName
            assert report["gpu"].startswith("gfx90a")
            report["torch"] = torch.__version__
            torch.manual_seed(413)
            for length in lengths:
                # Refuse full-shape profiling on a GPU occupied by the service.
                torch.cuda.empty_cache()
                free, _ = torch.cuda.mem_get_info()
                required = 4 * args.m * 32 * length * 2 + 2**30
                if free < required:
                    raise RuntimeError(f"Insufficient isolated GPU headroom: {free=} < {required=}")
                q = fake_quant_fp4(torch.randn(args.m, 32, 128, device="cuda", dtype=torch.bfloat16))
                k = fake_quant_fp4(torch.randn(length, 128, device="cuda", dtype=torch.bfloat16))
                w = (torch.randn(args.m, 32, device="cuda") * 0.03).bfloat16()
                fn = (lambda: DeepseekV41Indexer.scores(None, q, k, w)) if args.production else (
                    lambda: bounded_scores(q, k, w, args.slab_mib * 2**20))
                baseline, ref = measure(lambda: reference_scores(q, k, w), args.rounds)
                bounded, out = measure(fn, args.rounds)
                exact = torch.equal(ref, out)
                diff = out - ref
                row = {"length": length, "reference": baseline, "bounded": bounded,
                       "scores_exact": exact, "max_abs": float(diff.abs().max()),
                       "relative_l2": float(diff.norm() / ref.norm().clamp_min(1e-30))}
                del diff
                # Cover empty/trivial and long rows, then both selection levels.
                visible = torch.linspace(0, length, args.m, device="cuda").long()[:, None]
                invalid = torch.arange(length, device="cuda")[None, :] >= visible
                ref.masked_fill_(invalid, -torch.inf)
                out.masked_fill_(invalid, -torch.inf)
                row["top512_exact"] = torch.equal(stable_index_topk(ref, 512), stable_index_topk(out, 512))
                row["candidate_blocks_exact"] = torch.equal(
                    select_candidate_blocks(ref, visible, 2048, 8),
                    select_candidate_blocks(out, visible, 2048, 8))
                report["shapes"].append(row)
                print(json.dumps(row), flush=True)
                assert exact and row["top512_exact"] and row["candidate_blocks_exact"]
                del q, k, w, ref, out, invalid, visible
            report["passed"] = True
        except Exception as exc:
            report["error"] = f"{type(exc).__name__}: {exc}"
        finally:
            json.dump(report, handle, indent=2)
    print(json.dumps(report, indent=2), flush=True)
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
