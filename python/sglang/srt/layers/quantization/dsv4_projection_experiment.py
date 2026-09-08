"""Opt-in native TP8 prefill numerical diagnostic; never affects decode."""
from contextlib import contextmanager
from contextvars import ContextVar
import torch
from sglang.srt.environ import envs

_active = ContextVar('dsv4_row_stable_prefill', default=False)


def eligible(*, enabled, hip, arch, native, extend, tp, ep):
    return bool(enabled and hip and arch == 'gfx90a' and native and extend
                and tp == 8 and ep == 1)


@contextmanager
def projection_scope(batch, device):
    enabled = envs.SGLANG_DSV4_GFX90A_ROW_STABLE_PREFILL.get()
    if not enabled:
        yield
        return
    from sglang.srt.runtime_context import get_parallel
    parallel = get_parallel()
    spec = batch.spec_algorithm
    active = eligible(enabled=enabled, hip=bool(torch.version.hip),
        arch=torch.cuda.get_device_properties(device).gcnArchName.split(':')[0]
        if torch.version.hip else '', native=spec is None or spec.is_none(),
        extend=batch.forward_mode.is_extend_without_speculative(),
        tp=parallel.tp_size if parallel.attn_tp_size == 8 else 0,
        ep=parallel.moe_ep_size)
    token = _active.set(active)
    try:
        yield
    finally:
        _active.reset(token)


def maybe_row_stable_linear(x, weight):
    if not _active.get():
        return None
    if (x.ndim != 2 or weight.ndim != 2 or not 4 < x.shape[0] <= 4096
            or x.dtype != torch.bfloat16 or weight.dtype != torch.bfloat16
            or x.shape[1] != weight.shape[1] or weight.shape[1] % 64
            or weight.shape[0] % 64 or not x.is_contiguous()
            or not weight.is_contiguous()):
        return None
    from sglang.kernels.ops.quantization.gfx90a_row_stable_linear import row_stable_linear
    return row_stable_linear(x, weight)
