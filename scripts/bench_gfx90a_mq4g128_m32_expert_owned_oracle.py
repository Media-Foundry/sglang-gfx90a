"""M32 EP4 expert-owned MQ4G128 sorter/projection oracle."""

from __future__ import annotations

import statistics

import torch

from sglang.kernels.ops.moe.gfx90a_mq4g128_moe import (
    _expert_owned_module,
    _expert_owned_sorter_module,
    _indexed_module,
    _masked_weighted_reduce_module,
)

E = 128


def packed_weight(n: int, k: int, seed: int) -> torch.Tensor:
    torch.manual_seed(seed)
    weight = torch.zeros(E, n, k // 128, 72, dtype=torch.uint8, device="cuda")
    rows = E * n * (k // 128)
    scale = torch.full((rows, 1), 0.0078125, dtype=torch.float32, device="cuda")
    zero = torch.full((rows, 1), -0.05859375, dtype=torch.float32, device="cuda")
    flat = weight.view(rows, 72)
    flat[:, :4].copy_(scale.view(torch.uint8).view(rows, 4))
    flat[:, 4:8].copy_(zero.view(torch.uint8).view(rows, 4))
    flat[:, 8:].random_(0, 256)
    return weight


def elapsed(fn) -> float:
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    fn()
    end.record()
    end.synchronize()
    return start.elapsed_time(end) * 1000.0


def summary(values: list[float]) -> tuple[float, float]:
    ordered = sorted(values)
    return statistics.median(values), statistics.fmean(ordered[2:-2])


def run_shape(m: int, t: int, n: int, k: int, seed: int) -> None:
    torch.manual_seed(seed)
    x = torch.randn(m, k, dtype=torch.float32, device="cuda") * 0.1
    weight = packed_weight(n, k, seed + 100)
    ids = torch.full((m, t), -1, dtype=torch.int32, device="cuda")
    local_slots = torch.randperm(m * t, device="cuda")[: m * t // 4]
    ids.view(-1)[local_slots] = torch.randint(
        0, E, (local_slots.numel(),), dtype=torch.int32, device="cuda"
    )
    offsets = torch.empty(E + 1, dtype=torch.int32, device="cuda")
    assignments = torch.empty(m * t, dtype=torch.int32, device="cuda")
    indexed_out = torch.empty(m, t, n, dtype=torch.float32, device="cuda")
    owned_out = torch.empty_like(indexed_out)
    indexed = _indexed_module(E, m, t, n, k)
    sorter = _expert_owned_sorter_module(E, m, t)
    owned_variants = {w: _expert_owned_module(E, m, t, n, k, w) for w in (2, 4, 8)}
    owned = owned_variants[2]

    def baseline():
        indexed.run(x, weight, ids, indexed_out)

    def candidate():
        owned_out.zero_()
        sorter.run(ids, offsets, assignments)
        owned.run(x, weight, offsets, assignments, owned_out)

    def sort_only():
        sorter.run(ids, offsets, assignments)

    def project_only():
        owned.run(x, weight, offsets, assignments, owned_out)

    for _ in range(30):
        baseline()
        candidate()
    torch.cuda.synchronize()
    baseline()
    candidate()
    torch.cuda.synchronize()
    offsets_cpu = offsets.cpu()
    assignments_cpu = assignments.cpu()
    ids_cpu = ids.view(-1).cpu()
    assert offsets_cpu[0].item() == 0
    assert offsets_cpu[-1].item() == local_slots.numel()
    assert bool(torch.all(offsets_cpu[1:] >= offsets_cpu[:-1]))
    valid_assignments = assignments_cpu[: offsets_cpu[-1].item()]
    assert torch.equal(
        torch.sort(valid_assignments).values,
        torch.sort(local_slots.cpu().to(torch.int32)).values,
    )
    for expert in range(E):
        begin, end = offsets_cpu[expert : expert + 2].tolist()
        if begin != end:
            assert bool(
                torch.all(ids_cpu[assignments_cpu[begin:end].long()] == expert)
            )
    delta = (indexed_out - owned_out).abs()
    print(
        f"shape=M{m}/T{t}/N{n}/K{k} valid={int((ids >= 0).sum())}/{m*t} "
        f"offsets_ok=True "
        f"bitwise={bool(torch.equal(indexed_out, owned_out))} "
        f"finite={bool(torch.isfinite(owned_out).all())} "
        f"max_abs={delta.max().item():.9g} mean_abs={delta.mean().item():.9g}"
    )

    samples = {"indexed": [], "owned": [], "sort": [], "project": []}
    for _ in range(31):
        samples["indexed"].append(elapsed(baseline))
        samples["owned"].append(elapsed(candidate))
        samples["sort"].append(elapsed(sort_only))
        samples["project"].append(elapsed(project_only))
        samples["project"].append(elapsed(project_only))
        samples["sort"].append(elapsed(sort_only))
        samples["owned"].append(elapsed(candidate))
        samples["indexed"].append(elapsed(baseline))
    stats = {name: summary(values) for name, values in samples.items()}
    print(f"timings_us={stats}")
    print(
        f"median_speedup={stats['indexed'][0] / stats['owned'][0] - 1:.3%} "
        f"trim_speedup={stats['indexed'][1] / stats['owned'][1] - 1:.3%}"
    )
    for waves, variant in owned_variants.items():
        def wave_candidate():
            owned_out.zero_()
            sorter.run(ids, offsets, assignments)
            variant.run(x, weight, offsets, assignments, owned_out)

        values = []
        for _ in range(31):
            values.append(elapsed(wave_candidate))
            values.append(elapsed(wave_candidate))
        wave_candidate()
        torch.cuda.synchronize()
        print(
            f"waves={waves} complete_us={summary(values)} "
            f"bitwise={bool(torch.equal(indexed_out, owned_out))}"
        )

    best_waves = 4 if (m, t) == (32, 10) else 8
    best_owned = owned_variants[best_waves]

    def best_candidate():
        owned_out.zero_()
        sorter.run(ids, offsets, assignments)
        best_owned.run(x, weight, offsets, assignments, owned_out)

    best_candidate()
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        best_candidate()
    graph.replay()
    torch.cuda.synchronize()
    reference = owned_out.clone()
    for _ in range(1000):
        graph.replay()
    torch.cuda.synchronize()
    print(
        f"graph_replay=1000 finite={bool(torch.isfinite(owned_out).all())} "
        f"stable={bool(torch.equal(owned_out, reference))}"
    )

    if (m, t, n, k) == (320, 1, 2560, 640):
        partials = owned_out.reshape(32, 10, n)
        reduce_ids = ids.reshape(32, 10)
        router_weights = torch.rand(32, 10, dtype=torch.float32, device="cuda")
        reduced = torch.empty(32, n, dtype=torch.bfloat16, device="cuda")
        reducer = _masked_weighted_reduce_module(32, 10, n)

        def aten_reduce():
            return (partials * router_weights.unsqueeze(-1)).sum(dim=1).to(
                torch.bfloat16
            )

        def masked_reduce():
            reducer.run(partials, router_weights, reduce_ids, reduced)

        for _ in range(30):
            aten_reduce()
            masked_reduce()
        torch.cuda.synchronize()
        expected = aten_reduce()
        masked_reduce()
        torch.cuda.synchronize()
        aten_samples, masked_samples = [], []
        for _ in range(31):
            aten_samples.append(elapsed(aten_reduce))
            masked_samples.append(elapsed(masked_reduce))
            masked_samples.append(elapsed(masked_reduce))
            aten_samples.append(elapsed(aten_reduce))
        print(
            f"masked_reduce bitwise={bool(torch.equal(expected, reduced))} "
            f"max_abs={(expected.float() - reduced.float()).abs().max().item()} "
            f"aten_us={summary(aten_samples)} masked_us={summary(masked_samples)}"
        )


def main() -> None:
    if "gfx90a" not in torch.cuda.get_device_properties(0).gcnArchName:
        raise RuntimeError("gfx90a required")
    run_shape(16, 10, 1280, 2560, 5)
    run_shape(160, 1, 2560, 640, 9)
    run_shape(32, 10, 1280, 2560, 7)
    run_shape(320, 1, 2560, 640, 11)


if __name__ == "__main__":
    main()
