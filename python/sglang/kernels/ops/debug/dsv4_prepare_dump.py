"""Optional original-V4 prefill query/KV input diagnostics; no arithmetic changes."""
import os
from pathlib import Path

import torch

from .dsv4_sampled_stage_dump import sampled_stage_value


def make_prepare_dump(layer, rank, batch, positions):
    directory = os.getenv("SGLANG_DSV4_DEBUG_ATTN_DUMP_DIR")
    if os.getenv("SGLANG_DSV4_DEBUG_PREPARE_DUMP", "0") != "1" or not directory:
        return None
    spec = batch.spec_algorithm
    if not batch.forward_mode.is_extend_without_speculative() or not (spec is None or spec.is_none()):
        return None
    target_layer = int(os.getenv("SGLANG_DSV4_DEBUG_STAGE_LAYER", "-1"))
    target_rank = int(os.getenv("SGLANG_DSV4_DEBUG_STAGE_RANK", "-1"))
    if (target_layer >= 0 and layer != target_layer) or (target_rank >= 0 and rank != target_rank):
        return None
    prefix = f"layer_{layer}_rank_{rank}"
    root = Path(directory)

    def dump(name, value, *, full=False, rank0_only=False):
        if rank0_only and rank != 0:
            return
        if not isinstance(value, torch.Tensor):
            raise TypeError(f"Prepare dump requires a tensor: {name}, {type(value)}")
        root.mkdir(parents=True, exist_ok=True)
        snapshot = value.detach().cpu() if full else sampled_stage_value(
            name, value, positions, root, prefix)
        torch.save(snapshot, root / f"{prefix}_{name}.pt")

    return dump
