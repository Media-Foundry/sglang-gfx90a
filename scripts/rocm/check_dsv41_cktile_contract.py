#!/usr/bin/env python3
"""Isolate V4.1 A16W4 padding/row-layout contracts with real selected experts.

Does not change the checkpoint or serving process. Run with a single visible
GPU. References use independently expanded checkpoint weights and BF16 GEMM.
"""

import argparse
import importlib
import json
from pathlib import Path

import torch
import torch.nn.functional as F
from safetensors import safe_open


def metrics(actual, expected):
    a, b = actual.float().flatten(), expected.float().flatten()
    return {
        "finite": bool(torch.isfinite(a).all()),
        "actual_rms": a.square().mean().sqrt().item(),
        "reference_rms": b.square().mean().sqrt().item(),
        "relative_l2": ((a - b).norm() / b.norm().clamp_min(1e-20)).item(),
        "cosine": F.cosine_similarity(a, b, dim=0).item(),
        "max_abs": (a - b).abs().max().item(),
    }


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--trace", type=Path, required=True)
    p.add_argument("--model-dir", type=Path, default=Path("/media/PM983/deepseek-v4.1-flash"))
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--rank", type=int, default=0)
    p.add_argument("--tp", type=int, default=8)
    p.add_argument("--padding", type=int, choices=[384, 512], required=True)
    p.add_argument("--fix-w2-rows", action="store_true")
    p.add_argument("--compact-down", action="store_true")
    p.add_argument("--rows", type=int, default=1)
    p.add_argument("--graph-replays", type=int, default=0)
    args = p.parse_args()
    if args.compact_down and (args.padding != 384 or args.fix_w2_rows):
        p.error("compact down consumes uncorrected A16W4-v1 rows with padding384")
    if args.output.exists():
        raise FileExistsError(args.output)
    torch.set_num_threads(4)
    from aiter import ActivationType, QuantType
    from sglang.srt.layers.quantization.fp8 import (
        _gfx90a_cktile_reorder_w2_rows, shuffle_scale, shuffle_weight,
    )

    fm = importlib.import_module("aiter.fused_moe")
    index = json.loads((args.model_dir / "model.safetensors.index.json").read_text())["weight_map"]
    paths = list(args.trace.glob(f"rank{args.rank}-firstdiv-FusedMoE-*-call0-*.pt"))
    assert len(paths) == 1, paths
    trace = torch.load(paths[0], weights_only=True)
    assert 1 <= args.rows <= 8192
    source = trace["args"][0]
    if args.rows <= source.shape[0]:
        x = source[-args.rows:].cuda()
    else:
        # Geometry oracle only: tile real captured rows, do not label this
        # a diverse-request service benchmark.
        x = source.repeat((args.rows + source.shape[0] - 1) // source.shape[0], 1)[:args.rows].cuda()
    route_weights = trace["args"][1][0][-1:].expand(args.rows, -1).contiguous().cuda()
    expert_ids = trace["args"][1][1][-1].tolist()
    h, inter, e = x.shape[-1], 2304 // args.tp, 8
    assert inter <= args.padding
    lut = torch.tensor([0., .5, 1., 1.5, 2., 3., 4., 6., -0., -.5, -1., -1.5, -2., -3., -4., -6.], device="cuda")

    def raw(key, projection):
        def read(suffix):
            name = key + suffix
            with safe_open(args.model_dir / index[name], framework="pt", device="cpu") as f:
                return f.get_tensor(name).view(torch.uint8)
        w, s = read(".weight"), read(".scale")
        if projection == "w2":
            w = w[:, args.rank * inter // 2:(args.rank + 1) * inter // 2]
            s = s[:, args.rank * inter // 32:(args.rank + 1) * inter // 32]
        else:
            w = w[args.rank * inter:(args.rank + 1) * inter]
            s = s[args.rank * inter:(args.rank + 1) * inter]
        return w.cuda().contiguous(), s.cuda().contiguous()

    def expand(w, s):
        ids = torch.stack((w & 15, w >> 4), dim=-1).flatten(-2).long()
        scales = s.view(torch.float8_e8m0fnu).float().repeat_interleave(32, -1)
        return (lut[ids] * scales).bfloat16()

    w13 = torch.zeros((e, 2 * args.padding, h // 2), dtype=torch.uint8, device="cuda")
    w2 = torch.zeros((e, h, args.padding // 2), dtype=torch.uint8, device="cuda")
    s13 = torch.full((e, 2 * args.padding, h // 32), 127, dtype=torch.uint8, device="cuda")
    s2 = torch.full((e, h, args.padding // 32), 127, dtype=torch.uint8, device="cuda")
    ref_stage1, down_weights = [], []
    for slot, eid in enumerate(expert_ids):
        matrices = {}
        for name in ("w1", "w3", "w2"):
            w, s = raw(f"layers.0.ffn.experts.{eid}.{name}", name)
            matrices[name] = expand(w, s)
            if name == "w2":
                w2[slot, :, :inter // 2], s2[slot, :, :inter // 32] = w, s
            else:
                offset = 0 if name == "w1" else args.padding
                w13[slot, offset:offset + inter] = w
                s13[slot, offset:offset + inter] = s
        g = F.linear(x, matrices["w1"]).float().clamp(max=10)
        u = F.linear(x, matrices["w3"]).float().clamp(-10, 10)
        ref_stage1.append((F.silu(g) * u).bfloat16())
        down_weights.append(matrices["w2"])
    ref_stage1 = torch.stack(ref_stage1, dim=1)
    ref_out = sum(F.linear(ref_stage1[:, i], down_weights[i]).float() * route_weights[:, i:i + 1]
                  for i in range(len(expert_ids)))
    if args.fix_w2_rows:
        w2 = _gfx90a_cktile_reorder_w2_rows(w2)
        s2 = _gfx90a_cktile_reorder_w2_rows(s2)
    s13 = shuffle_scale(s13.reshape(-1, s13.shape[-1]), e, True, True)
    s2 = shuffle_scale(s2.reshape(-1, s2.shape[-1]), e, True, False)
    w13 = shuffle_weight(w13.view(torch.float4_e2m1fn_x2), gate_up=True)
    w2 = shuffle_weight(w2.view(torch.float4_e2m1fn_x2), gate_up=False)
    w13.is_shuffled = w2.is_shuffled = True
    ids = torch.arange(len(expert_ids), dtype=torch.int32, device="cuda").view(1, -1).repeat(args.rows, 1)
    captured = {}
    original = fm.cktile_moe_stage1

    def capture_stage1(*a, **kw):
        out = original(*a, **kw)
        captured["stage1"] = out.clone()
        return out

    fm.cktile_moe_stage1 = capture_stage1
    # Match the production ksplit=0 metadata while limiting the fixture to
    # eight experts. Bypass the missing gfx90a heuristic table, not the actual
    # CKTile selection. Do not write AIter's global tuning files.
    key = (fm.get_cu_num(), fm.get_padded_M(x.shape[0]), h, args.padding,
           e, len(expert_ids), str(ActivationType.Dsv4Silu), str(x.dtype),
           str(x.dtype), str(w13.dtype), str(QuantType.per_1x32), True, False)
    fm.cfg_2stages = {key: {"block_m": 32, "ksplit": 0,
                          "kernelName1": "", "kernelName2": "", "run_1stage": False}}
    fm.get_2stage_cfgs.cache_clear()
    try:
        outputs = []
        for _ in range(3):
            if args.compact_down:
                from sglang.kernels.ops.moe.gfx90a_dsv41_compact_ck import compact_ck_moe
                actual = compact_ck_moe(x, w13, s13, w2, s2, ids, route_weights)
            else:
                actual = fm.fused_moe(x, w13, w2, route_weights, ids,
                                 activation=ActivationType.Dsv4Silu,
                                 quant_type=QuantType.per_1x32,
                                 w1_scale=s13, w2_scale=s2,
                                 intermediate_pad=args.padding - inter,
                                 preshuffle=True)
            torch.cuda.synchronize()
            outputs.append(actual.clone())
    finally:
        fm.cktile_moe_stage1 = original
    graph_exact = None
    if args.graph_replays:
        assert args.compact_down
        from sglang.kernels.ops.moe.gfx90a_dsv41_compact_ck import compact_ck_moe
        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            for _ in range(3):
                compact_ck_moe(x, w13, s13, w2, s2, ids, route_weights)
        torch.cuda.current_stream().wait_stream(stream)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            graph_out = compact_ck_moe(x, w13, s13, w2, s2, ids, route_weights)
        graph.replay()
        expected_graph = graph_out.clone()
        any_mismatch = torch.zeros((), dtype=torch.bool, device=x.device)
        for _ in range(args.graph_replays):
            graph.replay()
            any_mismatch.logical_or_((graph_out != expected_graph).any())
        torch.cuda.synchronize()
        graph_exact = not bool(any_mismatch) and torch.equal(graph_out, actual)
    report = {
        "padding": args.padding, "fix_w2_rows": args.fix_w2_rows,
        "compact_down": args.compact_down,
        "rows": args.rows, "graph_replays": args.graph_replays, "graph_exact": graph_exact,
        "tiled_input_rows": args.rows > source.shape[0],
        "rank": args.rank, "raw_expert_ids": expert_ids,
        "weight_shapes": [list(w13.shape), list(w2.shape)],
        "scale_shapes": [list(s13.shape), list(s2.shape)],
        "stage1": metrics(captured["stage1"][..., :inter], ref_stage1),
        "output": metrics(actual, ref_out),
        "replay_exact": all(torch.equal(outputs[0], o) for o in outputs[1:]),
    }
    with args.output.open("x") as f:
        json.dump(report, f, indent=2)
        f.write("\n")
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
