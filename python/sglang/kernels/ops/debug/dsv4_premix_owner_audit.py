"""Default-off observation only: never replace replicated model state.

Full-tensor SHA256 and local-slice oracle at selected large-prefill calls.
CPU copies deliberately perturb timing; results are not performance measurements.
"""
import hashlib
import json
import os
from pathlib import Path

import torch
import triton

_call = 0


def owned_rows(m, rank, size=8):
    if m <= 0 or not 0 <= rank < size:
        raise ValueError((m, rank, size))
    per_rank = ((m + 8 * size - 1) // (8 * size)) * 8
    return min(rank * per_rank, m), min((rank + 1) * per_rank, m)


def selected_calls():
    # Four forwards, 85 paired boundaries/forward. Records expose actual M/Fn
    # hashes, so this schedule is a sampling policy, not an inferred layer label.
    value = os.getenv('SGLANG_DSV4_DEBUG_PREMIX_OWNER_AUDIT_CALLS',
                      '0,40,84,85,125,169,170,210,254,255,295,339')
    calls = frozenset(int(s) for s in value.split(','))
    if not calls or min(calls) < 0:
        raise ValueError('nonnegative premix call IDs required')
    return calls


def audit(residual, fn, rms, reference, eps):
    global _call
    directory = os.getenv('SGLANG_DSV4_DEBUG_PREMIX_OWNER_AUDIT_DIR')
    if not directory:
        return
    from sglang.srt.layers.dsv4_prefill_experiments import mix_pair_active
    if not mix_pair_active() or torch.cuda.is_current_stream_capturing():
        return
    call = _call
    _call += 1
    if call not in selected_calls():
        return
    from sglang.srt.distributed import get_tp_group
    from sglang.kernels.ops.layernorm.gfx90a_mhc_premix_pair import premix8_pair
    tp = get_tp_group()
    assert tp.world_size == 8
    rank = tp.rank_in_group
    m = len(residual)
    assert 8192 <= m <= 65536
    start, end = owned_rows(m, rank)
    n = end - start
    local = torch.empty((n, 1, 24), dtype=torch.float32, device=residual.device)
    premix8_pair[(12, triton.cdiv(n, 8))](residual[start:end], fn, rms[start:end],
                                       local, n, float(eps), num_warps=1)
    exact = torch.equal(local.view(torch.int32), reference[start:end].view(torch.int32))
    result = dict(call=call, rank=rank, rows=m, start=start, end=end, eps=float(eps),
                  local_byte_exact=exact, diagnostic_only=True,
                  output_replaced=False, tensors={})
    # Copy/hash one tensor at a time, do not retain full activations on disk.
    for name, tensor in [('residual', residual), ('fn', fn), ('rms', rms), ('mix', reference)]:
        cpu = tensor.detach().contiguous().view(torch.uint8).cpu()
        result['tensors'][name] = dict(shape=list(tensor.shape), dtype=str(tensor.dtype),
            sha256=hashlib.sha256(memoryview(cpu.numpy()).cast('B')).hexdigest())
        del cpu
    root = Path(directory)
    root.mkdir(parents=True, exist_ok=True)
    with (root/f'call-{call:04d}-rank-{rank}.json').open('x') as stream:
        json.dump(result, stream, indent=2)
        stream.write('\n')
    print(f'[TP{rank}] pre-mix owner audit: call={call} rows={m} local={n} exact={exact}', flush=True)
