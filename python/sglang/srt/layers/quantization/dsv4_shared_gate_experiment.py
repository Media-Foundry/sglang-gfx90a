"""Default-off TP8 C1 native-decode shared gate experiment."""
from contextlib import contextmanager
from contextvars import ContextVar
import torch
from sglang.srt.environ import envs

_active = ContextVar("dsv4_shared_gate_round", default=False)


def eligible(*, enabled, hip, arch, native, decode, batch_size, tp, ep):
    return bool(enabled and hip and arch == "gfx90a" and native and decode
                and batch_size == 1 and tp == 8 and ep == 1)


@contextmanager
def shared_gate_scope(batch, device):
    if not envs.SGLANG_DSV4_GFX90A_TP8_C1_SHARED_GATE_ROUND.get():
        token = _active.set(False)
        try:
            yield
        finally:
            _active.reset(token)
        return
    from sglang.srt.runtime_context import get_parallel
    p = get_parallel()
    spec = batch.spec_algorithm
    active = eligible(enabled=True, hip=bool(torch.version.hip),
        arch=torch.cuda.get_device_properties(device).gcnArchName.split(':')[0]
        if torch.version.hip else '', native=spec is None or spec.is_none(),
        decode=batch.forward_mode.is_decode(), batch_size=batch.batch_size,
        tp=p.tp_size if p.attn_tp_size == 8 else 0, ep=p.moe_ep_size)
    token = _active.set(active)
    try:
        yield
    finally:
        _active.reset(token)


def maybe_shared_gate(x, weight, limit):
    if not _active.get():
        return None
    if (x.shape != (1,4096) or weight.shape != (512,4096)
        or x.dtype != torch.bfloat16 or weight.dtype != torch.bfloat16
        or not x.is_cuda or x.device != weight.device
        or not x.is_contiguous() or not weight.is_contiguous() or limit != 10):
        return None
    from sglang.kernels.ops.quantization.gfx90a_bf16_gated_gemv import (
        _jit_gfx90a_tp8_shared_gate_round_module,
    )
    output = torch.empty((1,256),device=x.device,dtype=x.dtype)
    _jit_gfx90a_tp8_shared_gate_round_module().run(x,weight,output,10.)
    return output
