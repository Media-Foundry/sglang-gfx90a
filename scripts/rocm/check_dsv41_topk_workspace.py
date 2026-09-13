#!/usr/bin/env python3
"""Bounded stable Top-K vs unbounded stable GPU sort: memory and replay oracle."""

import argparse
import json
from pathlib import Path

import torch

from bench_dsv41_indexer_workspace import measure
from sglang.srt.layers.attention.dsv4.dsv41_sparse import (
    INDEXER_SORT_SLAB_ELEMENTS,
    stable_index_topk,
)


def full_sort(scores):
    return torch.argsort(scores, dim=-1, descending=True, stable=True)[:, :512].sort(-1).values


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--lengths", default="32768,65536")
    p.add_argument("--m", type=int, default=2304)
    p.add_argument("--graph-replays", type=int, default=100)
    args = p.parse_args()
    report = {"passed": False, "reference": "unbounded stable GPU argsort",
              "sort_slab_elements": INDEXER_SORT_SLAB_ELEMENTS,
              "shapes": []}
    with args.output.open("x") as handle:
        try:
            assert args.m > 0 and args.graph_replays > 0
            report["gpu"] = torch.cuda.get_device_properties(0).gcnArchName
            assert report["gpu"].startswith("gfx90a")
            assert torch.cuda.mem_get_info()[0] >= 8 * 2**30, "use an isolated GPU"
            torch.manual_seed(52)
            for width in [int(n) for n in args.lengths.split(",")]:
                assert width >= 512
                scores = torch.randint(0, 64, (args.m, width), dtype=torch.int16,
                                       device="cuda").float()
                visible = torch.linspace(0, width, args.m, device="cuda").long()[:, None]
                cols = torch.arange(width, device="cuda")[None, :]
                scores.masked_fill_(cols >= visible, -torch.inf)
                before, expected = measure(lambda: full_sort(scores), 3)
                after, output = measure(lambda: stable_index_topk(scores, 512), 3)
                assert torch.equal(output, expected)
                stream = torch.cuda.Stream()
                stream.wait_stream(torch.cuda.current_stream())
                with torch.cuda.stream(stream):
                    for _ in range(3):
                        stable_index_topk(scores, 512)
                torch.cuda.current_stream().wait_stream(stream)
                graph = torch.cuda.CUDAGraph()
                with torch.cuda.graph(graph, stream=stream):
                    out = stable_index_topk(scores, 512)
                errors = torch.zeros((), dtype=torch.int64, device="cuda")
                for _ in range(args.graph_replays):
                    graph.replay()
                    errors.add_((out != expected).sum())
                assert int(errors) == 0
                for _ in range(10):
                    scores.copy_(torch.randint(0, 64, scores.shape, device="cuda", dtype=torch.int16))
                    visible.copy_(torch.randint(0, width + 1, visible.shape, device="cuda"))
                    scores.masked_fill_(cols >= visible, -torch.inf)
                    expected = full_sort(scores)
                    graph.replay()
                    assert torch.equal(out, expected)
                row = {"rows": args.m, "width": width, "before": before, "after": after,
                       "selected_ids_exact": True, "graph_errors": int(errors),
                       "graph_replays": args.graph_replays, "mutated_score_and_position_replays": 10}
                report["shapes"].append(row)
                print(json.dumps(row), flush=True)
                del graph, out, output, expected, scores, visible, cols
                torch.cuda.empty_cache()
            report["passed"] = True
        except Exception as exc:
            report["error"] = f"{type(exc).__name__}: {exc}"
        finally:
            json.dump(report, handle, indent=2)
    print(json.dumps(report, indent=2), flush=True)
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
