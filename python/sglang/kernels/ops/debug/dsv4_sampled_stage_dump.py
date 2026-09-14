"""Optional sampled stage dump; input IDs/positions are always saved in full."""
import os
from pathlib import Path

import torch


def sampled_stage_value(name, value, positions, directory, prefix):
    selected = os.getenv("SGLANG_DSV4_DEBUG_STAGE_SAMPLE_POSITIONS", "")
    if not selected or name in ("input_ids", "positions"):
        return value.detach().cpu()
    if value.ndim == 0 or value.shape[0] != positions.numel():
        return value.detach().cpu()
    wanted = [int(p) for p in selected.split(",")]
    mask = torch.zeros_like(positions, dtype=torch.bool)
    for p in wanted:
        mask |= positions == p
    rows = mask.nonzero().flatten()
    torch.save(rows.cpu(), Path(directory) / f"{prefix}_sample_rows.pt")
    return value.detach().index_select(0, rows).cpu()
