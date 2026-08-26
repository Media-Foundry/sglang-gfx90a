#!/usr/bin/env python3
"""Summarize TP8 BS32 expert occupancy from an SGLang recorder dump.

The recorder stores the logical count after an eight-rank sum, so counts are
divided by eight.  BS32 passes are selected by their exact 32*6*43*8 checksum.
The script is CPU-only and writes CSV to stdout.
"""

import argparse
import glob

import torch


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "recorder",
        nargs="?",
        default=None,
        help="expert_distribution_recorder_*.pt (defaults to the newest /tmp dump)",
    )
    args = parser.parse_args()
    path = args.recorder
    if path is None:
        candidates = sorted(glob.glob("/tmp/expert_distribution_recorder_*.pt"))
        if not candidates:
            raise FileNotFoundError("no /tmp expert distribution recorder found")
        path = candidates[-1]

    payload = torch.load(path, map_location="cpu", weights_only=False)
    counts = payload["logical_count"]
    mask = counts.sum((1, 2)) == 32 * 6 * 43 * 8
    occupancy = (counts[mask] // 8).to(torch.int64)
    if occupancy.shape[0] == 0:
        raise RuntimeError("recorder contains no complete BS32 passes")
    if not torch.all(occupancy.sum((1, 2)) == 32 * 6 * 43):
        raise RuntimeError("BS32 pass checksum mismatch")

    print(f"# recorder={path},passes={occupancy.shape[0]}")
    print(
        "layer,active_mean,active_p05,active_p50,active_p95,"
        "maxocc_mean,maxocc_p50,maxocc_p95,maxocc_max,"
        "A4_blocks,A4_util_pct,A8_blocks,A8_util_pct,"
        "A16_blocks,A16_util_pct,M4_full_assign_pct,"
        "M8_full_assign_pct,M16_full_assign_pct"
    )
    for layer in range(occupancy.shape[1]):
        values = occupancy[:, layer, :]
        active = (values > 0).sum(1).float()
        max_occupancy = values.max(1).values.float()
        active_q = torch.quantile(active, torch.tensor([0.05, 0.5, 0.95]))
        max_q = torch.quantile(max_occupancy, torch.tensor([0.5, 0.95]))
        tile_stats = []
        for tile in (4, 8, 16):
            blocks = ((values + tile - 1) // tile).sum(1).float()
            utilization = 100 * values.sum().item() / (blocks.sum().item() * tile)
            full = 100 * ((values // tile) * tile).sum().item() / values.sum().item()
            tile_stats.append((blocks.mean().item(), utilization, full))
        print(
            f"{layer},{active.mean():.4f},{active_q[0]:.1f},{active_q[1]:.1f},"
            f"{active_q[2]:.1f},{max_occupancy.mean():.4f},{max_q[0]:.1f},"
            f"{max_q[1]:.1f},{max_occupancy.max().item():.0f},"
            f"{tile_stats[0][0]:.4f},{tile_stats[0][1]:.4f},"
            f"{tile_stats[1][0]:.4f},{tile_stats[1][1]:.4f},"
            f"{tile_stats[2][0]:.4f},{tile_stats[2][1]:.4f},"
            f"{tile_stats[0][2]:.4f},{tile_stats[1][2]:.4f},"
            f"{tile_stats[2][2]:.4f}"
        )


if __name__ == "__main__":
    main()
