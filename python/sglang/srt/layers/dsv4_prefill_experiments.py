"""Default-off original-V4 native prefill scopes; never infer mode from M alone."""

import os
from contextvars import ContextVar
from functools import wraps

import torch

POST_REUSE_ENV = "SGLANG_DSV4_PREFILL_POST_FUSED4"
MIX_REUSE_ENV = "SGLANG_DSV4_PREFILL_MIX_REUSE4"
MIX_PAIR_ENV = "SGLANG_DSV4_PREFILL_MIX_PAIR_COLUMNS"
_post_reuse = ContextVar("dsv4_prefill_post_fused4", default=False)
_mix_reuse = ContextVar("dsv4_prefill_mix_reuse4", default=False)
_mix_pair = ContextVar("dsv4_prefill_mix_pair_columns", default=False)


def post_reuse_active():
    return _post_reuse.get()


def mix_reuse_active():
    return _mix_reuse.get()


def mix_pair_active():
    return _mix_pair.get()


def mix_pair_eligible(runner, batch):
    # Do not inherit MIXED/DLLM modes admitted by the older reuse predicate.
    return (
        post_reuse_eligible(runner, batch)
        and getattr(batch.forward_mode, "name", None) == "EXTEND"
        and not getattr(runner, "is_draft_worker", False)
        and getattr(runner.ps, "attn_dcp_size", 1) == 1
    )


def post_reuse_eligible(runner, batch):
    cfg = runner.model_config.hf_text_config
    ps = runner.ps
    return (
        getattr(cfg, "model_type", None) == "deepseek_v4"
        and getattr(cfg, "num_hidden_layers", None) == 43
        and getattr(cfg, "hidden_size", None) == 4096
        and ps.tp_size == ps.attn_tp_size == 8
        and ps.moe_ep_size == ps.attn_cp_size == 1
        and getattr(ps, "pp_size", 1) == 1
        and (batch.spec_algorithm is None or batch.spec_algorithm.is_none())
        and batch.forward_mode.is_extend_without_speculative()
        and getattr(batch, "tbo_parent_token_range", None) is None
        and getattr(batch, "_original_forward_mode", None) is None
        and 8192 <= batch.input_ids.shape[0] <= 65536
    )


def instrument_prefill_post_reuse(fn):
    # Disabled deployment keeps the original callable and its call overhead.
    if os.getenv(POST_REUSE_ENV, "0") != "1":
        return fn

    @wraps(fn)
    def wrapped(self, forward_batch, *args, **kwargs):
        batch = forward_batch
        enabled = post_reuse_eligible(self.model_runner, batch)
        if enabled:
            enabled = (
                bool(torch.version.hip)
                and batch.input_ids.device.type == "cuda"
                and "gfx90a" in torch.cuda.get_device_properties(
                    batch.input_ids.device
                ).gcnArchName
                and not torch.cuda.is_current_stream_capturing()
            )
        token = _post_reuse.set(enabled)
        try:
            return fn(self, batch, *args, **kwargs)
        finally:
            _post_reuse.reset(token)

    return wrapped


def instrument_prefill_mix_reuse(fn):
    if os.getenv(MIX_REUSE_ENV, "0") != "1":
        return fn

    @wraps(fn)
    def wrapped(self, forward_batch, *args, **kwargs):
        batch = forward_batch
        enabled = post_reuse_eligible(self.model_runner, batch)
        if enabled:
            enabled = (
                bool(torch.version.hip)
                and batch.input_ids.device.type == "cuda"
                and "gfx90a" in torch.cuda.get_device_properties(
                    batch.input_ids.device
                ).gcnArchName
                and not torch.cuda.is_current_stream_capturing()
            )
        token = _mix_reuse.set(enabled)
        pair_token = _mix_pair.set(
            enabled and os.getenv(MIX_PAIR_ENV, "0") == "1"
            and mix_pair_eligible(self.model_runner, batch)
        )
        try:
            return fn(self, batch, *args, **kwargs)
        finally:
            _mix_pair.reset(pair_token)
            _mix_reuse.reset(token)

    return wrapped
