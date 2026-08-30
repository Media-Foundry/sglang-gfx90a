#!/usr/bin/env python3
"""Cold-resident packed-byte roofline for one DSV4 TP4 routed-expert rank."""

from __future__ import annotations

import argparse
import json
import random
import statistics
import time
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import torch
from safetensors import safe_open

from sglang.kernels.ops.debug.gfx90a_packed_weight_roofline import (
    packed_weight_roofline,
)

EXPERTS = 256
BYTES_PER_EXPERT = 3 * 1024 * 1024


def load_tp4_rank(model: Path, layer: int, rank: int) -> torch.Tensor:
    index = json.loads((model / "model.safetensors.index.json").read_text())
    weight_map = index["weight_map"]
    by_shard: dict[str, list[tuple[int, str]]] = defaultdict(list)
    for expert in range(EXPERTS):
        for name in ("w1", "w3", "w2"):
            key = f"layers.{layer}.ffn.experts.{expert}.{name}.weight"
            by_shard[weight_map[key]].append((expert, name))
    packed = torch.empty((EXPERTS, BYTES_PER_EXPERT), dtype=torch.uint8)
    row = 512
    packed_col = 256
    for shard, entries in sorted(by_shard.items()):
        with safe_open(model / shard, framework="pt", device="cpu") as f:
            for expert, name in entries:
                key = f"layers.{layer}.ffn.experts.{expert}.{name}.weight"
                raw = f.get_tensor(key).view(torch.uint8)
                if name == "w1":
                    part = raw[rank * row : (rank + 1) * row, :]
                    begin = 0
                elif name == "w3":
                    part = raw[rank * row : (rank + 1) * row, :]
                    begin = 1024 * 1024
                else:
                    part = raw[:, rank * packed_col : (rank + 1) * packed_col]
                    begin = 2 * 1024 * 1024
                flat = part.contiguous().reshape(-1)
                assert flat.numel() == 1024 * 1024, (key, part.shape)
                packed[expert, begin : begin + flat.numel()].copy_(flat)
    return packed


def expert_checksums(packed: torch.Tensor) -> list[int]:
    view = packed.numpy().view(np.uint64).reshape(EXPERTS, -1)
    return [int(np.bitwise_xor.reduce(view[e])) for e in range(EXPERTS)]


def expected_checksum(order: list[int], sums: list[int]) -> int:
    value = 0
    for expert in order:
        value ^= sums[expert]
    return value


def rotated(order: list[int], shift: int) -> list[int]:
    return [int((expert + shift) % EXPERTS) for expert in order]


def timed_ms(fn) -> float:
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    fn()
    end.record()
    end.synchronize()
    return start.elapsed_time(end)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="/home/pc/models/modelscope")
    parser.add_argument("--route", default="/tmp/dsv4_tp4_m64_real_route.pt")
    parser.add_argument("--layer", type=int, default=34)
    parser.add_argument("--rank", type=int, default=0)
    parser.add_argument("--rounds", type=int, default=9)
    args = parser.parse_args()

    torch.cuda.set_device(0)
    snapshot = torch.load(args.route, map_location="cpu", weights_only=False)
    ids = snapshot["topk_ids"].to(torch.int64).flatten().tolist()
    counts = Counter(ids)
    real_unique = sorted(counts)
    a4 = []
    for expert in real_unique:
        a4.extend([expert] * ((counts[expert] + 3) // 4))
    assert len(real_unique) == 166, len(real_unique)
    # The authoritative saved pass-20/layer-34 snapshot has 182 padded A4
    # scans.  An earlier transcript quoted 174 from a different histogram;
    # retain the exact file-derived value here instead of forcing that number.
    assert len(a4) == sum((value + 3) // 4 for value in counts.values())

    print("loading one layer / TP4 rank packed weights to host ...", flush=True)
    host = load_tp4_rank(Path(args.model), args.layer, args.rank)
    sums = expert_checksums(host)
    print("copying 768 MiB once to GPU-resident buffer ...", flush=True)
    weights = host.cuda(non_blocking=False)
    del host
    order_gpu = torch.zeros(EXPERTS, dtype=torch.int32, device="cuda")
    order_len = torch.zeros(1, dtype=torch.int32, device="cuda")
    checksum = torch.zeros(2080, dtype=torch.int64, device="cuda")

    def checksum_value() -> int:
        values = checksum.cpu().numpy().view(np.uint64)
        return int(np.bitwise_xor.reduce(values))

    def launch(order: list[int]) -> int:
        staging = torch.full((EXPERTS,), -1, dtype=torch.int32)
        staging[: len(order)] = torch.tensor(order, dtype=torch.int32)
        order_gpu.copy_(staging)
        order_len.fill_(len(order))
        checksum.zero_()
        packed_weight_roofline(weights, order_gpu, order_len, checksum)
        torch.cuda.synchronize()
        return checksum_value()

    # Compile and validate the exact resident bytes before timing.
    assert launch(real_unique) == expected_checksum(real_unique, sums)
    rng = random.Random(20260830)
    for _ in range(100):
        shift = rng.randrange(EXPERTS)
        base = rng.choice((list(range(166)), real_unique, a4))
        order = rotated(base, shift)
        got = launch(order)
        want = expected_checksum(order, sums)
        assert got == want, (shift, got, want)
    print("correctness: 100 rotation mutations CPU/GPU xor exact", flush=True)

    # Capture only the kernel/reset; order metadata remains at stable addresses.
    stable = real_unique
    staging = torch.full((EXPERTS,), -1, dtype=torch.int32)
    staging[: len(stable)] = torch.tensor(stable, dtype=torch.int32)
    order_gpu.copy_(staging)
    order_len.fill_(len(stable))
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        checksum.zero_()
        packed_weight_roofline(weights, order_gpu, order_len, checksum)
    expected = expected_checksum(stable, sums)
    for _ in range(1000):
        graph.replay()
    torch.cuda.synchronize()
    assert checksum_value() == expected
    print("correctness: 1000 HIP Graph replays bitwise stable", flush=True)

    cases = {
        "A_contiguous_166_cold_rotate": (list(range(166)), True),
        "B_real_unique_166_cold_rotate": (real_unique, True),
        "B_real_unique_166_same_address_warm": (real_unique, False),
        f"C_real_A4_scans_{len(a4)}_cold_rotate": (a4, True),
        f"C_real_A4_scans_{len(a4)}_same_address_warm": (a4, False),
        "A_contiguous_favorable_146_cold_rotate": (list(range(146)), True),
    }
    results = {}
    for name, (base, rotate_each_round) in cases.items():
        samples = []
        # Every timed pass streams 438--522 MiB and rotates addresses. Copy and
        # scalar reset happen before the event pair and are not measured.
        for round_index in range(args.rounds):
            shift = (37 * round_index + 19) % EXPERTS if rotate_each_round else 0
            order = rotated(base, shift)
            staging.fill_(-1)
            staging[: len(order)] = torch.tensor(order, dtype=torch.int32)
            order_gpu.copy_(staging)
            order_len.fill_(len(order))
            checksum.zero_()
            ms = timed_ms(
                lambda: packed_weight_roofline(
                    weights, order_gpu, order_len, checksum
                )
            )
            got = checksum_value()
            assert got == expected_checksum(order, sums), (name, round_index)
            samples.append(ms * 1000.0)
        ordered = sorted(samples)
        trimmed = ordered[1:-1] if len(ordered) >= 7 else ordered
        us = statistics.mean(trimmed)
        byte_count = len(base) * BYTES_PER_EXPERT
        gbps = byte_count / (us * 1e-6) / 1e9
        results[name] = {
            "entries": len(base),
            "bytes": byte_count,
            "samples_us": samples,
            "trimmed_mean_us": us,
            "median_us": statistics.median(samples),
            "effective_GBps": gbps,
        }
        print(
            f"{name}: {byte_count / 2**20:.1f} MiB, "
            f"trim={us:.2f} us median={statistics.median(samples):.2f} us, "
            f"{gbps:.1f} GB/s"
        )

    out = Path("/tmp/dsv4_tp4_m64_packed_weight_roofline.json")
    out.write_text(json.dumps(results, indent=2))
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
