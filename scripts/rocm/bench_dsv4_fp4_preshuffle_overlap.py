#!/usr/bin/env python3
"""Measure whether next-layer raw-FP4 preshuffle hides under M4608 AIter MoE."""

import statistics

import torch
import aiter
import aiter.fused_moe as fused_moe_module
from aiter.fused_moe import fused_moe

from sglang.kernels.ops.moe.gfx90a_dsv4_fp4_preshuffle import preshuffle_into


def center(values):
    return statistics.mean(sorted(values)[1:-1])


def main():
    assert torch.cuda.get_device_properties(0).gcnArchName.split(":", 1)[0] == "gfx90a"
    torch.manual_seed(20260902)
    fused_moe_module.fused_moe_1stage_dict.setdefault("gfx90a", set())
    dev = "cuda"
    e, m, topk, h, i = 256, 4608, 6, 4096, 512

    def raw_layer():
        return (
            torch.randint(0, 256, (e, 2 * i, h // 2), dtype=torch.uint8, device=dev),
            torch.randint(112, 136, (e, 2 * i, h // 32), dtype=torch.uint8, device=dev),
            torch.randint(0, 256, (e, h, i // 2), dtype=torch.uint8, device=dev),
            torch.randint(112, 136, (e, h, i // 32), dtype=torch.uint8, device=dev),
        )

    raw_a, raw_b = raw_layer(), raw_layer()
    buf_a = tuple(torch.empty_like(x) for x in raw_a)
    buf_b = tuple(torch.empty_like(x) for x in raw_b)
    preshuffle_into(*raw_a, buf_a, blocks=416)
    torch.cuda.synchronize()

    x = torch.randn((m, h), dtype=torch.bfloat16, device=dev)
    scores = torch.randn((m, e), dtype=torch.float32, device=dev)
    weights, ids = torch.topk(scores, topk, dim=-1)
    weights = torch.softmax(weights, dim=-1)
    ids = ids.to(torch.int32)
    fp4 = torch.float4_e2m1fn_x2
    e8m0 = torch.float8_e8m0fnu

    def compute():
        return fused_moe(
            x,
            buf_a[0].view(fp4),
            buf_a[2].view(fp4),
            weights,
            ids,
            activation=aiter.ActivationType.Dsv4Silu,
            quant_type=aiter.QuantType.per_1x32,
            w1_scale=buf_a[1].view(e8m0),
            w2_scale=buf_a[3].view(e8m0),
            block_size_M=32,
            splitk=0,
            preshuffle=True,
        )

    def shuffle_next():
        preshuffle_into(*raw_b, buf_b, blocks=416)

    for _ in range(3):
        y = compute()
    torch.cuda.synchronize()
    if not torch.isfinite(y).all():
        raise RuntimeError("AIter output is non-finite")

    alt = torch.cuda.Stream()

    def measure(kind, repeats=5):
        values = []
        for _ in range(repeats):
            torch.cuda.synchronize()
            start, finish = torch.cuda.Event(True), torch.cuda.Event(True)
            start.record()
            if kind == "compute":
                compute()
            elif kind == "shuffle":
                shuffle_next()
            elif kind == "sequential":
                shuffle_next()
                compute()
            elif kind == "overlap":
                ready = torch.cuda.Event()
                ready.record()
                with torch.cuda.stream(alt):
                    alt.wait_event(ready)
                    shuffle_next()
                    alt_done = torch.cuda.Event()
                    alt_done.record()
                compute()
                torch.cuda.current_stream().wait_event(alt_done)
            else:
                raise AssertionError(kind)
            finish.record()
            finish.synchronize()
            values.append(start.elapsed_time(finish) * 1000)
        print(f"RESULT arm={kind} samples_us={values} trimmed_us={center(values):.3f}")

    for kind in ("compute", "shuffle", "sequential", "overlap"):
        measure(kind, repeats=7)


if __name__ == "__main__":
    main()
