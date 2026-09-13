#!/usr/bin/env python3
"""GPU candidate-block oracle against the local official inference function.

Extract only the pure reference function by AST (no model/CUDA-kernel import).
Use unique reachable scores so official topk tie instability cannot hide a
contract mismatch. CPU unit tests independently cover deterministic tie order.
"""

import argparse
import ast
import hashlib
import json
from pathlib import Path

import torch
import torch.nn.functional as F

from sglang.srt.layers.attention.dsv4.dsv41_sparse import select_candidate_blocks


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--reference", type=Path, default=Path(
        "/media/PM983/deepseek-v4.1-flash/inference/model.py"))
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--graph-replays", type=int, default=1000)
    args = p.parse_args()
    if args.graph_replays < 1:
        p.error("graph replays must be positive")
    source = args.reference.read_text()
    definition = next(n for n in ast.parse(source).body
                      if isinstance(n, ast.FunctionDef) and n.name == "select_candidate_blocks")
    namespace = {"torch": torch, "F": F}
    exec(compile(ast.Module(body=[definition], type_ignores=[]), str(args.reference), "exec"), namespace)
    reference = namespace["select_candidate_blocks"]
    reference_source = ast.get_source_segment(source, definition)
    report = {"passed": False, "reference": str(args.reference),
              "reference_function_sha256": hashlib.sha256(reference_source.encode()).hexdigest(),
              "torch": torch.__version__, "shapes": []}
    with args.output.open("x") as handle:
        try:
            report["gpu"] = torch.cuda.get_device_properties(0).gcnArchName
            assert report["gpu"].startswith("gfx90a")
            torch.manual_seed(93)
            for width in (16383, 16384, 16385, 32769):
                scores = torch.randperm(width, device="cuda").float().repeat(6, 1)
                visible = torch.tensor([0, 1, 7, width // 2, width - 1, width],
                                       device="cuda")[:, None]
                ids = torch.arange(width, device="cuda")[None, :]
                scores.masked_fill_(ids >= visible, -torch.inf)

                def expected():
                    return reference(scores.cpu(), visible.cpu(), 2048, 8).cuda()

                def expanded(mask):
                    return mask.repeat_interleave(8, dim=-1)[:, :width]

                ref = expected()
                actual = select_candidate_blocks(scores, visible, 2048, 8)
                assert torch.equal(expanded(actual), ref)
                assert torch.equal(actual.sum(-1), ((visible[:, 0] + 7) // 8).clamp_max(2048))
                stream = torch.cuda.Stream()
                stream.wait_stream(torch.cuda.current_stream())
                with torch.cuda.stream(stream):
                    for _ in range(3):
                        select_candidate_blocks(scores, visible, 2048, 8)
                torch.cuda.current_stream().wait_stream(stream)
                graph = torch.cuda.CUDAGraph()
                with torch.cuda.graph(graph, stream=stream):
                    out = select_candidate_blocks(scores, visible, 2048, 8)
                errors = torch.zeros((), dtype=torch.int64, device="cuda")
                for _ in range(args.graph_replays):
                    graph.replay()
                    errors.add_((out != actual).sum())
                assert int(errors) == 0
                for _ in range(10):
                    scores.copy_(torch.randperm(width, device="cuda").float()[None, :])
                    visible.copy_(torch.randint(0, width + 1, visible.shape, device="cuda"))
                    scores.masked_fill_(ids >= visible, -torch.inf)
                    ref = expected()
                    graph.replay()
                    assert torch.equal(expanded(out), ref)
                row = {"width": width, "rows": 6, "official_mask_exact": True,
                       "graph_replays": args.graph_replays, "graph_errors": int(errors),
                       "score_and_position_mutations": 10, "mutations_official_exact": True,
                       "mask_bytes": out.numel() * out.element_size(),
                       "full_position_mask_bytes": ref.numel() * ref.element_size()}
                report["shapes"].append(row)
                print(json.dumps(row), flush=True)
            report["passed"] = True
        except Exception as exc:
            report["error"] = f"{type(exc).__name__}: {exc}"
        finally:
            json.dump(report, handle, indent=2)
    print(json.dumps(report, indent=2), flush=True)
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
