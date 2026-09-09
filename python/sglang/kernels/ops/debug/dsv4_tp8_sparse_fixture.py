"""Offline/eager TP8 sparse-attention fixture writer, never a timing path.

The caller must select the desired DSpark target layer and disable graphs.
Only referenced KV slots are copied; physical-slot identity is retained.
"""
from pathlib import Path

import torch


def save_fixture(path, *, q, kv, indices, indptr, sink, output, scale,
                 provenance):
    if q.is_cuda and torch.cuda.is_current_stream_capturing():
        raise RuntimeError("TP8 sparse fixture requires eager execution")
    if (q.shape != (128, 8, 512) or q.dtype != torch.bfloat16
            or kv.ndim != 2 or kv.shape[1] != 512 or kv.dtype != q.dtype
            or output.shape != q.shape or output.dtype != q.dtype
            or indptr.shape != (129,) or indptr.dtype != torch.int32
            or indices.ndim != 1 or indices.dtype != torch.int32
            or sink.ndim != 1 or sink.numel() < 8 or sink.dtype != torch.float32):
        raise ValueError("expected TP8 M128/H8/D512 BF16 sparse fixture")
    if any(t.device != q.device for t in (kv, indices, indptr, sink, output)):
        raise ValueError("fixture tensors must share one device")
    ptr_cpu = indptr.detach().cpu().contiguous()
    total = int(ptr_cpu[-1])
    if (int(ptr_cpu[0]) != 0 or total > indices.numel()
            or bool((ptr_cpu[1:] < ptr_cpu[:-1]).any())):
        raise ValueError("invalid ragged offsets")
    selected = indices[:total].detach().long()
    # Unified sparse decode treats negative slots as ragged padding/sentinels
    # (see the production CK kernels' ``slot >= 0`` guard).  Preserve those
    # entries in the fixture and compact only referenced physical KV rows.
    if total and int(selected.max()) >= kv.shape[0]:
        raise ValueError("invalid physical KV slot")
    valid = selected >= 0
    valid_slots = selected[valid]
    slots, compact = torch.unique(valid_slots, sorted=True, return_inverse=True)
    remap = selected.to(torch.int32)
    remap[valid] = compact.to(torch.int32)
    payload = dict(
        format="dsv4_tp8_sparse_fixture_v1",
        q=q.detach().cpu().contiguous(),
        unified_kv=kv.index_select(0, slots).detach().cpu().contiguous(),
        kv_indices=remap.to(torch.int32).cpu(),
        kv_indptr=ptr_cpu,
        attn_sink=sink[:8].detach().cpu().contiguous(),
        baseline_output=output.detach().cpu().contiguous(),
        physical_slots=slots.cpu(),
        softmax_scale=float(scale),
        provenance=dict(provenance, original_pool_slots=kv.shape[0]),
    )
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    # Exclusive creation prevents silently replacing a previous witness.
    with path.open("xb") as handle:
        torch.save(payload, handle)
    return payload
