"""Validate that gfx90a s_memrealtime markers update during graph replay."""

import torch

from sglang.kernels.ops.debug.gfx90a_realtime_marker import (
    gfx90a_realtime_marker,
)


def main() -> None:
    device = torch.device("cuda", 0)
    markers = torch.zeros(2, dtype=torch.uint64, device=device)
    value = torch.ones(1 << 20, dtype=torch.float32, device=device)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        gfx90a_realtime_marker(markers, 0)
        for _ in range(32):
            value.add_(1.0)
        gfx90a_realtime_marker(markers, 1)

    previous = 0
    deltas = []
    for _ in range(5):
        graph.replay()
        torch.cuda.synchronize()
        current = markers.cpu().tolist()
        assert current[0] > previous, (previous, current)
        assert current[1] > current[0], current
        previous = current[1]
        deltas.append(current[1] - current[0])
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(100):
        graph.replay()
    end.record()
    end.synchronize()
    marker_delta = markers.cpu().tolist()[1] - markers.cpu().tolist()[0]
    event_us = start.elapsed_time(end) * 1000.0 / 100.0
    ns_per_tick = event_us * 1000.0 / marker_delta
    print(
        f"PASS: replay markers update; raw deltas={deltas}; "
        f"event={event_us:.3f} us, calibration={ns_per_tick:.3f} ns/tick"
    )


if __name__ == "__main__":
    main()
