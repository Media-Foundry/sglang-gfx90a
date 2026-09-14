"""Optional sampled stage dump; input IDs/positions are always saved in full."""
import os
from contextlib import contextmanager
from pathlib import Path

import torch


def parameter_stage(name):
    return name.startswith(("hc_", "projection_")) or name.endswith("_weight")


def should_dump_stage(name):
    return not (os.getenv("SGLANG_DSV4_DEBUG_STAGE_SKIP_WEIGHTS", "0") == "1"
                and parameter_stage(name))


@contextmanager
def stage_dump_scope(module, callback):
    """Temporarily expose the owning V4 layer's dump callback to its MoE.

    Only used for synchronous eager diagnostics, never a production stream or
    graph protocol. Restore ownership even if the forward raises.
    """
    sentinel = object()
    old = getattr(module, "_dsv4_stage_dump", sentinel)
    module._dsv4_stage_dump = callback
    try:
        yield
    finally:
        if old is sentinel:
            del module._dsv4_stage_dump
        else:
            module._dsv4_stage_dump = old


def sampled_stage_value(name, value, positions, directory, prefix):
    selected = os.getenv("SGLANG_DSV4_DEBUG_STAGE_SAMPLE_POSITIONS", "")
    if not selected or name in ("input_ids", "positions") or parameter_stage(name):
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
