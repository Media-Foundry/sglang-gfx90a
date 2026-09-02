#!/usr/bin/env python3
import statistics

import torch

from aiter.ops.shuffle import shuffle_scale_a16w4, shuffle_weight_a16w4
from sglang.kernels.ops.moe.gfx90a_dsv4_fp4_preshuffle import preshuffle_into


def time_us(fn, iterations=20):
    for _ in range(5):
        fn()
    torch.cuda.synchronize()
    a, b = torch.cuda.Event(True), torch.cuda.Event(True)
    a.record()
    for _ in range(iterations):
        fn()
    b.record()
    b.synchronize()
    return a.elapsed_time(b) * 1000 / iterations


def main():
    assert torch.cuda.get_device_properties(0).gcnArchName.split(":", 1)[0] == "gfx90a"
    torch.manual_seed(20260902)
    dev = "cuda"
    w13 = torch.randint(0, 256, (256, 1024, 2048), dtype=torch.uint8, device=dev)
    s13 = torch.randint(0, 256, (256, 1024, 128), dtype=torch.uint8, device=dev)
    w2 = torch.randint(0, 256, (256, 4096, 256), dtype=torch.uint8, device=dev)
    s2 = torch.randint(0, 256, (256, 4096, 16), dtype=torch.uint8, device=dev)
    out = tuple(torch.empty_like(x) for x in (w13, s13, w2, s2))

    inv = torch.tensor([0, 4, 1, 5, 2, 6, 3, 7], device=dev)
    rw2 = w2.view(256, 32, 8, 16, 256).index_select(2, inv).reshape_as(w2)
    rs2 = s2.view(256, 32, 8, 16, 16).index_select(2, inv).reshape_as(s2)
    refs = (
        shuffle_weight_a16w4(w13, NLane=16, gate_up=True).view(torch.uint8),
        shuffle_scale_a16w4(s13.reshape(-1, 128), 256, True).reshape_as(s13).view(torch.uint8),
        shuffle_weight_a16w4(rw2, NLane=16, gate_up=False).view(torch.uint8),
        shuffle_scale_a16w4(rs2.reshape(-1, 16), 256, False).reshape_as(s2).view(torch.uint8),
    )
    preshuffle_into(w13, s13, w2, s2, out)
    torch.cuda.synchronize()
    names = ("w13", "s13", "w2", "s2")
    for name, got, ref in zip(names, out, refs):
        if not torch.equal(got, ref):
            neq = torch.nonzero(got != ref)
            raise RuntimeError(f"{name} mismatch count={neq.shape[0]} first={neq[:8].cpu().tolist()}")
    print("CORRECT byte_exact=True")

    def hip():
        preshuffle_into(w13, s13, w2, s2, out)

    def torch_ref():
        shuffle_weight_a16w4(w13, NLane=16, gate_up=True)
        shuffle_scale_a16w4(s13.reshape(-1, 128), 256, True)
        shuffle_weight_a16w4(rw2, NLane=16, gate_up=False)
        shuffle_scale_a16w4(rs2.reshape(-1, 16), 256, False)

    arms = {"torch": torch_ref}
    for blocks in (104, 208, 416, 832, 1664):
        arms[f"hip_b{blocks}"] = lambda b=blocks: preshuffle_into(w13, s13, w2, s2, out, blocks=b)
    samples = {name: [] for name in arms}
    order = tuple(arms) + tuple(reversed(tuple(arms)))
    for _ in range(7):
        for name in order:
            fn = arms[name]
            samples[name].append(time_us(fn))
    for name, values in samples.items():
        trimmed = statistics.mean(sorted(values)[1:-1])
        print(f"RESULT arm={name} median_us={statistics.median(values):.3f} trimmed_us={trimmed:.3f}")


if __name__ == "__main__":
    main()
