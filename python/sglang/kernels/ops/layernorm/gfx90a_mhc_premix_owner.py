"""Opt-in TP8 ordinary-prefill coefficient ownership, not residual sharding.

The caller validates the input layout. The scoped model-runner predicate admits
only original V4 large eager prefill. No changed per-row arithmetic or persistent
weight/workspace cache. All ranks must use the same launch configuration.
"""
import os

import torch
import torch.distributed as dist
import triton

from .gfx90a_mhc_premix_pair import premix8_pair

_logged = False


def partition(m, rank):
    if m <= 0 or not 0 <= rank < 8:
        raise ValueError((m, rank))
    capacity = triton.cdiv(m, 64) * 8
    return min(rank * capacity, m), min((rank + 1) * capacity, m), capacity


def try_premix_owner(residual, fn, rms, eps):
    global _logged
    if os.getenv('SGLANG_DSV4_DEBUG_PREFILL_MIX_OWNER', '0') != '1':
        return None
    from sglang.srt.layers.dsv4_prefill_experiments import mix_pair_active
    if not mix_pair_active() or not 8192 <= residual.shape[0] <= 65536:
        return None
    if torch.cuda.is_current_stream_capturing():
        return None
    if os.getenv('SGLANG_DSV4_DEBUG_PREFILL_MIX_MFMA', '0') == '1':
        raise ValueError('pre-mix owner and MFMA are independent candidates; select only one')
    from sglang.srt.distributed import get_tp_group
    tp = get_tp_group()
    assert tp.world_size == 8
    rank = tp.rank_in_group
    m = residual.shape[0]
    start, end, capacity = partition(m, rank)
    n = end - start
    local = torch.empty((capacity, 24), dtype=torch.float32, device=residual.device)
    # Initialize padding only. No global all-output zeroing or H16384 packing.
    if n < capacity:
        local[n:].zero_()
    gathered = torch.empty((8 * capacity, 24), dtype=torch.float32, device=residual.device)
    premix8_pair[(12, triton.cdiv(n, 8))](residual[start:end], fn, rms[start:end],
                                       local, n, float(eps), num_warps=1)
    # Same explicit RCCL path as the oracle; exchange FP32 values as bytes, no reduction.
    dist.all_gather_into_tensor(gathered, local, group=tp.device_group)
    output = gathered[:m].view(m, 1, 24)
    checking = os.getenv('SGLANG_DSV4_DEBUG_PREFILL_MIX_OWNER_CHECK', '0') == '1'
    if checking:
        reference = torch.empty_like(output)
        premix8_pair[(12, triton.cdiv(m, 8))](residual, fn, rms, reference,
                                           m, float(eps), num_warps=1)
        exact = torch.equal(output.view(torch.int32), reference.view(torch.int32))
        assert exact, ('owner mixes differ from this rank full-input reference', rank, m)
        print(f'[TP{rank}] pre-mix owner full-reference exact: rows={m}', flush=True)
    if not _logged:
        print(f'[TP{rank}] pre-mix owner selected: rows={m} local_rows={n} '
              f'gather_bytes={gathered.numel()*4} check={int(checking)}', flush=True)
        _logged = True
    return output
