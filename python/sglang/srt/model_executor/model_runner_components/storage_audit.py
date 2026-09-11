"""Opt-in post-load accounting for live model tensor storage.

This intentionally counts allocator state, tensor storage and host table
mappings separately.  Summing ``tensor.numel()`` is misleading for views,
aliases and mmap-backed parameters, while allocator ``reserved`` bytes also
include free cache blocks that are not live model tensors.
"""

from __future__ import annotations

import logging
from collections import defaultdict
from typing import Iterable

import torch

from sglang.srt.utils import get_bool_env_var


def _iter_named_tensors(model: torch.nn.Module) -> Iterable[tuple[str, torch.Tensor]]:
    yield from model.named_parameters(recurse=True, remove_duplicate=False)
    yield from model.named_buffers(recurse=True, remove_duplicate=False)


def _storage_key(tensor: torch.Tensor) -> tuple[str, int | None, int, int]:
    storage = tensor.untyped_storage()
    return (
        tensor.device.type,
        tensor.device.index,
        storage.data_ptr(),
        storage.nbytes(),
    )


def _gib(nbytes: int) -> float:
    return nbytes / 2**30


def maybe_log_model_storage_audit(
    model: torch.nn.Module,
    *,
    device: str,
    rank: int,
    logger: logging.Logger,
) -> None:
    """Log a zero-copy live-storage census when explicitly requested.

    Set ``SGLANG_DEBUG_MODEL_STORAGE_AUDIT=1`` for a diagnostic startup.  The
    walk neither clones tensors nor synchronizes the device.
    """

    if not get_bool_env_var("SGLANG_DEBUG_MODEL_STORAGE_AUDIT"):
        return

    storages: dict[tuple[str, int | None, int, int], dict[str, object]] = {}
    logical_bytes = defaultdict(int)
    logical_tensors = defaultdict(int)

    for name, tensor in _iter_named_tensors(model):
        if tensor is None:
            continue
        dev = str(tensor.device)
        logical_bytes[dev] += tensor.numel() * tensor.element_size()
        logical_tensors[dev] += 1
        try:
            key = _storage_key(tensor)
        except (RuntimeError, NotImplementedError):
            continue
        entry = storages.setdefault(
            key,
            {
                "nbytes": key[3],
                "device": dev,
                "dtype": str(tensor.dtype),
                "names": [],
            },
        )
        entry["names"].append(name)

    unique_bytes = defaultdict(int)
    unique_count = defaultdict(int)
    dtype_bytes = defaultdict(int)
    alias_storages = 0
    routed_weight_bytes = 0
    routed_scale_bytes = 0
    direct_scale_clone_bytes = 0
    gpu_engram_bytes = 0

    for entry in storages.values():
        nbytes = int(entry["nbytes"])
        dev = str(entry["device"])
        names = list(entry["names"])
        unique_bytes[dev] += nbytes
        unique_count[dev] += 1
        dtype_bytes[(dev, str(entry["dtype"]))] += nbytes
        alias_storages += len(names) > 1

        if any(name.endswith(("w13_weight", "w2_weight")) for name in names):
            routed_weight_bytes += nbytes
        if any(
            name.endswith(("w13_weight_scale_inv", "w2_weight_scale_inv"))
            for name in names
        ):
            routed_scale_bytes += nbytes
        if any(name.endswith("w2_weight_scale_logical") for name in names):
            direct_scale_clone_bytes += nbytes
        if any(".engram.embed." in name for name in names) and dev.startswith(
            ("cuda", "hip")
        ):
            gpu_engram_bytes += nbytes

    # A host table owns one mmap.  Its weight and scale Parameters are views of
    # that mapping, so deduplicate by the mmap object rather than counting both.
    host_maps: dict[int, int] = {}
    for module in model.modules():
        table = getattr(module, "host_table", None)
        mm = getattr(table, "mm", None)
        if mm is not None:
            host_maps.setdefault(id(mm), int(table.nbytes))

    dtype_summary = ", ".join(
        f"{dev}/{dtype}={_gib(nbytes):.3f}GiB"
        for (dev, dtype), nbytes in sorted(dtype_bytes.items())
    )
    device_summary = ", ".join(
        f"{dev}:logical={_gib(logical_bytes[dev]):.3f}GiB/"
        f"unique={_gib(unique_bytes[dev]):.3f}GiB/"
        f"storages={unique_count[dev]}/tensors={logical_tensors[dev]}"
        for dev in sorted(set(logical_bytes) | set(unique_bytes))
    )

    allocated = reserved = peak = 0
    if device != "cpu" and torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated()
        reserved = torch.cuda.memory_reserved()
        peak = torch.cuda.max_memory_allocated()

    logger.info(
        "model storage audit rank=%d: %s; aliases=%d; dtypes=[%s]; "
        "routed_weights=%.3fGiB routed_scales=%.3fGiB "
        "direct_scale_clones=%.3fGiB gpu_engram=%.3fGiB "
        "host_mmaps=%.3fGiB/%d allocator=allocated:%.3fGiB,"
        "reserved:%.3fGiB,peak:%.3fGiB",
        rank,
        device_summary,
        alias_storages,
        dtype_summary,
        _gib(routed_weight_bytes),
        _gib(routed_scale_bytes),
        _gib(direct_scale_clone_bytes),
        _gib(gpu_engram_bytes),
        _gib(sum(host_maps.values())),
        len(host_maps),
        _gib(allocated),
        _gib(reserved),
        _gib(peak),
    )
