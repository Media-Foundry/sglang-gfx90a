"""Independent, default-off FP32/20 MHC policy for small original-V4 prefill.

Never broaden the large-kernel mix/post scopes merely to select tail arithmetic.
"""
from contextvars import ContextVar
from functools import wraps
import os

import torch

ENV = "SGLANG_DSV4_PREFILL_MHC_COMMON_SMALL"
COMMON_ENV = "SGLANG_DSV4_PREFILL_MHC_COMMON_FP32"
_active = ContextVar("dsv4_small_prefill_common_mhc", default=False)


def active():
    return _active.get()


def eligible(runner, batch):
    cfg = runner.model_config.hf_text_config
    ps = runner.ps
    return (
        getattr(cfg, "model_type", None) == "deepseek_v4"
        and getattr(cfg, "num_hidden_layers", None) == 43
        and getattr(cfg, "hidden_size", None) == 4096
        and ps.tp_size == ps.attn_tp_size == 8
        and ps.moe_ep_size == ps.attn_cp_size == 1
        and getattr(ps, "pp_size", 1) == 1
        and getattr(ps, "attn_dcp_size", 1) == 1
        and not getattr(runner, "is_draft_worker", False)
        and (batch.spec_algorithm is None or batch.spec_algorithm.is_none())
        and getattr(batch.forward_mode, "name", None) == "EXTEND"
        and getattr(batch, "tbo_parent_token_range", None) is None
        and getattr(batch, "_original_forward_mode", None) is None
        and 0 < batch.input_ids.shape[0] < 8192
    )


def instrument(fn):
    if os.getenv(ENV, "0") != "1":
        return fn
    if os.getenv(COMMON_ENV, "0") != "1":
        raise ValueError("Small-prefill MHC common policy requires large-prefill common FP32/20")

    @wraps(fn)
    def wrapped(self, forward_batch, *args, **kwargs):
        enabled = eligible(self.model_runner, forward_batch)
        if enabled:
            enabled = (
                bool(torch.version.hip)
                and forward_batch.input_ids.device.type == "cuda"
                and "gfx90a" in torch.cuda.get_device_properties(forward_batch.input_ids.device).gcnArchName
                and not torch.cuda.is_current_stream_capturing()
            )
        token = _active.set(enabled)
        try:
            return fn(self, forward_batch, *args, **kwargs)
        finally:
            _active.reset(token)

    return wrapped
