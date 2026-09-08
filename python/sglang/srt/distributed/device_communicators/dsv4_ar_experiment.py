"""Default-off DSV4 decode scope for the isolated TP8 all-reduce experiment."""
from contextlib import contextmanager
from contextvars import ContextVar
import logging

import torch

from sglang.srt.environ import envs

_active = ContextVar('dsv4_tp8_m32_legacy_ar', default=False)


def eligible(*, enabled, hip, arch, decode, batch_size, native):
    return bool(enabled and hip and arch == 'gfx90a' and decode
                and batch_size == 32 and native)


@contextmanager
def dsv4_ar_scope(batch, device):
    enabled = envs.SGLANG_DSV4_GFX90A_TP8_M32_LEGACY_AR.get()
    if not enabled:
        yield
        return
    spec = batch.spec_algorithm
    native = spec is None or spec.is_none()
    active = eligible(enabled=enabled, hip=bool(torch.version.hip),
                      arch=torch.cuda.get_device_properties(device).gcnArchName.split(':')[0]
                      if torch.version.hip else '',
                      decode=batch.forward_mode.is_decode(),
                      batch_size=batch.batch_size, native=native)
    token = _active.set(active)
    try:
        yield
    finally:
        _active.reset(token)


def adapt_dsv4_ar(aiter_cls):
    if not envs.SGLANG_DSV4_GFX90A_TP8_M32_LEGACY_AR.get():
        return aiter_cls

    class Dsv4AR(aiter_cls):
        def all_reduce(self, inp, *, out=None, use_new=True,
                       open_fp8_quant=False, registered=False):
            if (_active.get() and self.world_size == 8
                    and inp.shape == (32, 4096) and inp.dtype == torch.bfloat16
                    and inp.is_contiguous() and not open_fp8_quant):
                use_new = False
                if not getattr(self, '_dsv4_legacy_ar_logged', False):
                    logging.getLogger(__name__).info(
                        'DSV4 native TP8 M32 BF16 legacy AIter AR selected')
                    self._dsv4_legacy_ar_logged = True
            return super().all_reduce(inp, out=out, use_new=use_new,
                                      open_fp8_quant=open_fp8_quant, registered=registered)

    return Dsv4AR
