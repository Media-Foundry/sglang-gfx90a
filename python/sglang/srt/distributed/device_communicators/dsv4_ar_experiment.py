"""Default-off native DSV4 decode scope shared by narrow TP8 experiments."""
from contextlib import contextmanager
from contextvars import ContextVar
import logging

import torch

from sglang.srt.environ import envs

_active = ContextVar('dsv4_tp8_m32_legacy_ar', default=False)
_native_active = ContextVar('dsv4_native_m32_experiment', default=False)
_attention_active = ContextVar('dsv4_native_decode_attention', default=False)
_dspark_m128_active = ContextVar('dsv4_dspark_tp8_m128_ar', default=False)
_down_uniform_override = ContextVar('dsv4_down_uniform_capture', default=None)


def down_uniform_requested():
    override = _down_uniform_override.get()
    return (envs.SGLANG_DSV4_GFX90A_TP8_M32_DOWN_UNIFORM.get()
            if override is None else override)


@contextmanager
def down_uniform_capture(enabled):
    """Override only kernel selection during a controlled startup capture.

    This is not an eligibility bypass: the native M32 and TP8/EP1/geometry
    predicates still apply. The caller must not change this around live eager
    requests; use separately captured graph/output pairs for runtime A/B.
    """
    if type(enabled) is not bool:
        raise TypeError('capture arm must be a boolean')
    token = _down_uniform_override.set(enabled)
    try:
        yield
    finally:
        _down_uniform_override.reset(token)


def native_attention_active():
    return _attention_active.get()


def native_m32_active():
    return _native_active.get()


def down_uniform_eligible(*, native_scope, tp_size, ep_size, gfx90a,
                          hidden_shape, topk_shape, weight_shape, geometry,
                          incompatible):
    return bool(native_scope and tp_size == 8 and ep_size == 1 and gfx90a
                and tuple(hidden_shape) == (32, 4096)
                and tuple(topk_shape) == (32, 6)
                and tuple(weight_shape) == (256, 4096, 128)
                and tuple(geometry) == (4, 2, 8, 832, True)
                and not incompatible)


def eligible(*, enabled, hip, arch, decode, batch_size, native):
    return bool(enabled and hip and arch == 'gfx90a' and decode
                and batch_size == 32 and native)


def shared_after_topk_eligible(*, enabled, hip, arch, decode, batch_size,
                               native, tp_size, ep_size, dsv4):
    return bool(dsv4 and tp_size == 8 and ep_size == 1 and eligible(
        enabled=enabled, hip=hip, arch=arch, decode=decode,
        batch_size=batch_size, native=native))


@contextmanager
def dsv4_ar_scope(batch, device):
    enabled = envs.SGLANG_DSV4_GFX90A_TP8_M32_LEGACY_AR.get()
    gate_enabled = envs.SGLANG_DSV4_GFX90A_TP8_M32_GATE_PREFETCH.get()
    down_enabled = down_uniform_requested()
    attention_enabled = envs.SGLANG_DSV4_GFX90A_TP8_DECODE_ATTN_WARPS2.get()
    c1_attention_enabled = envs.SGLANG_DSV4_GFX90A_TP8_C1_ATTN_WARPS2.get()
    dspark_enabled = envs.SGLANG_DSV4_GFX90A_DSPARK_TP8_M128_AR_BLOCKS.get() != 0
    if not (enabled or gate_enabled or down_enabled or attention_enabled or c1_attention_enabled or dspark_enabled):
        yield
        return
    spec = batch.spec_algorithm
    native = spec is None or spec.is_none()
    active = eligible(enabled=enabled or gate_enabled or down_enabled, hip=bool(torch.version.hip),
                      arch=torch.cuda.get_device_properties(device).gcnArchName.split(':')[0]
                      if torch.version.hip else '',
                      decode=batch.forward_mode.is_decode(),
                      batch_size=batch.batch_size, native=native)
    token = _active.set(active and enabled)
    native_token = _native_active.set(active)
    attention_token = _attention_active.set(
        native and batch.forward_mode.is_decode()
        and ((attention_enabled and batch.batch_size in (1,32))
             or (c1_attention_enabled and batch.batch_size == 1)))
    dspark_token = _dspark_m128_active.set(
        dspark_enabled and spec is not None and spec.is_dspark()
        and batch.forward_mode.is_target_verify() and batch.batch_size == 32
        and getattr(batch.spec_info, 'num_tokens_per_req', None) == 4)
    try:
        yield
    finally:
        _active.reset(token)
        _native_active.reset(native_token)
        _attention_active.reset(attention_token)
        _dspark_m128_active.reset(dspark_token)


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
