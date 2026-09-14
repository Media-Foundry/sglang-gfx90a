"""Opt-in original-V4 TP8 eager-prefill MHC iteration policy.

Keep kernel geometry and weight precision unchanged. All nonempty ordinary
prefill shapes, including tails, use the model's20 iterations. Decode, mixed
prefill/decode, speculation, graph capture and other models remain untouched.
"""

import logging
import os
from contextvars import ContextVar
from functools import wraps

import torch

ENV = "SGLANG_DSV4_PREFILL_MHC_CONFIG_ITERS"
_config_iters = ContextVar("dsv4_prefill_mhc_config_iters", default=None)
_logged = set()
logger = logging.getLogger(__name__)


def resolve_sinkhorn_iters(default: int) -> int:
    selected = _config_iters.get()
    if selected is None:
        return default
    key = (default, selected)
    if key not in _logged:
        logger.info(
            "DSV4 TP8 native prefill MHC Sinkhorn policy selected: legacy=%d config=%d",
            default, selected,
        )
        _logged.add(key)
    return selected


def eligible(runner, batch):
    cfg = runner.model_config.hf_text_config
    ps = runner.ps
    return (
        getattr(cfg, "model_type", None) == "deepseek_v4"
        and getattr(cfg, "num_hidden_layers", None) == 43
        and getattr(cfg, "hidden_size", None) == 4096
        and getattr(cfg, "hc_sinkhorn_iters", None) == 20
        and ps.tp_size == ps.attn_tp_size == 8
        and ps.moe_ep_size == ps.attn_cp_size == 1
        and getattr(ps, "pp_size", 1) == 1
        and getattr(ps, "attn_dcp_size", 1) == 1
        and not getattr(runner, "is_draft_worker", False)
        and (batch.spec_algorithm is None or batch.spec_algorithm.is_none())
        # is_extend_without_speculative() also admits MIXED and DLLM_EXTEND.
        # Compare the enum name without importing ForwardBatch during discovery.
        and getattr(batch.forward_mode, "name", None) == "EXTEND"
        and getattr(batch, "tbo_parent_token_range", None) is None
        and getattr(batch, "_original_forward_mode", None) is None
        and batch.input_ids.shape[0] > 0
    )


def instrument_prefill_mhc_iters(fn):
    if os.getenv(ENV, "0") != "1":
        return fn

    @wraps(fn)
    def wrapped(self, forward_batch, *args, **kwargs):
        runner = self.model_runner
        active = eligible(runner, forward_batch)
        if active:
            active = (
                bool(torch.version.hip)
                and forward_batch.input_ids.device.type == "cuda"
                and torch.cuda.get_device_properties(
                    forward_batch.input_ids.device
                ).gcnArchName.split(":", 1)[0] == "gfx90a"
                and not torch.cuda.is_current_stream_capturing()
            )
        token = _config_iters.set(
            runner.model_config.hf_text_config.hc_sinkhorn_iters if active else None
        )
        try:
            return fn(self, forward_batch, *args, **kwargs)
        finally:
            _config_iters.reset(token)

    return wrapped
