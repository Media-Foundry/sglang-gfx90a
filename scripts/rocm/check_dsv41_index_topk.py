#!/usr/bin/env python3
"""Reproduce ROCm cutoff-tie membership drift and validate V4.1 stable Top-K."""

import argparse
import json
from pathlib import Path

import torch

from sglang.srt.layers.attention.dsv4.dsv41_sparse import stable_index_topk


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output", required=True, type=Path)
    p.add_argument("--replays", type=int, default=30)
    p.add_argument("--graph-replays", type=int, default=1000)
    args = p.parse_args()
    assert args.replays > 0 and args.graph_replays > 0
    torch.manual_seed(20260913)
    report = {"passed": False, "gpu": torch.cuda.get_device_properties(0).gcnArchName,
              "torch": torch.__version__, "shapes": []}
    assert report["gpu"].startswith("gfx90a")
    with args.output.open("x") as handle:
        try:
            for m, n in ((256, 1280), (2304, 1152), (72, 1188)):
                scores = torch.randint(0, 128, (m, n), device="cuda").float()
                # Include empty, trivial, cutoff, and nontrivial causal rows.
                visible = torch.arange(m, device="cuda") % (n + 1)
                visible[::2] = n
                scores.masked_fill_(torch.arange(n, device="cuda")[None, :] >= visible[:, None], -torch.inf)
                reference = torch.argsort(scores.cpu(), dim=-1, descending=True, stable=True)[:, :512].sort(-1).values.cuda()
                initial = scores.topk(512, dim=-1, sorted=False).indices.sort(-1).values
                initial_valid = torch.where(initial < visible[:, None], initial, -1)
                unstable, max_changed = 0, 0
                for _ in range(args.replays):
                    actual = scores.topk(512, dim=-1, sorted=False).indices.sort(-1).values
                    actual_valid = torch.where(actual < visible[:, None], actual, -1)
                    changed = int((actual_valid != initial_valid).any(-1).sum())
                    unstable += bool(changed)
                    max_changed = max(max_changed, changed)
                    assert torch.equal(stable_index_topk(scores, 512), reference)
                # Preserve the same prefix-visible masking contract as serving.
                ids = stable_index_topk(scores, 512)
                chosen = ids < visible[:, None]
                assert torch.equal(chosen.sum(-1), visible.clamp_max(512))
                stream = torch.cuda.Stream()
                stream.wait_stream(torch.cuda.current_stream())
                with torch.cuda.stream(stream):
                    for _ in range(3):
                        stable_index_topk(scores, 512)
                torch.cuda.current_stream().wait_stream(stream)
                graph = torch.cuda.CUDAGraph()
                with torch.cuda.graph(graph, stream=stream):
                    graph_out = stable_index_topk(scores, 512)
                errors = torch.zeros((), device="cuda", dtype=torch.int64)
                for _ in range(args.graph_replays):
                    graph.replay()
                    errors.add_((graph_out != reference).sum())
                assert int(errors) == 0
                # Replay against changed scores as well: repeated fixed input
                # alone cannot rule out stale graph output/workspace contents.
                for _ in range(10):
                    scores.copy_(torch.randint(0, 128, (m, n), device="cuda"))
                    scores.masked_fill_(torch.arange(n, device="cuda")[None, :] >= visible[:, None], -torch.inf)
                    reference = torch.argsort(scores.cpu(), dim=-1, descending=True, stable=True)[:, :512].sort(-1).values.cuda()
                    graph.replay()
                    assert torch.equal(graph_out, reference)
                report["shapes"].append({"m": m, "n": n, "eager_replays": args.replays,
                    "old_mismatching_replays": unstable, "old_max_changed_rows": max_changed,
                    "stable_matches_cpu_reference": True, "valid_counts_exact": True,
                    "graph_replays": args.graph_replays, "graph_errors": int(errors),
                    "mutated_score_graph_replays": 10, "mutated_score_matches_cpu": True})
                print(json.dumps(report["shapes"][-1]), flush=True)
            report["passed"] = True
        except Exception as error:
            report["error"] = f"{type(error).__name__}: {error}"
        finally:
            json.dump(report, handle, indent=2)
    print(json.dumps(report, indent=2))
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
