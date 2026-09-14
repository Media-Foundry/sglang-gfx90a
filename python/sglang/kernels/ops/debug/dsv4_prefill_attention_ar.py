"""Default-off original-V4 large-prefill FP32 collective diagnostic.

This isolates BF16 collective rounding. It is deliberately not a model-wide
determinism switch: projection and expert reduction orders remain unchanged.
"""
import logging
import os

import torch
import torch.distributed as dist

_logged = False


def eligible(*, enabled, native, extend, hip, arch, tp, attn_tp, ep, rows):
    return bool(enabled and native and extend and hip and arch == "gfx90a"
                and tp == attn_tp == 8 and ep == 1 and 8192 <= rows <= 36864)


def enabled_for(batch, device, rows, attn_tp):
    if os.getenv("SGLANG_DSV4_DEBUG_PREFILL_ATTN_AR_FP32", "0") != "1":
        return False
    from sglang.srt.runtime_context import get_parallel

    parallel = get_parallel()
    spec = batch.spec_algorithm
    return eligible(
        enabled=True, native=spec is None or spec.is_none(),
        extend=batch.forward_mode.is_extend_without_speculative(),
        hip=bool(torch.version.hip),
        arch=torch.cuda.get_device_properties(device).gcnArchName.split(":")[0]
        if torch.version.hip else "", tp=parallel.tp_size, attn_tp=attn_tp,
        ep=parallel.moe_ep_size, rows=rows)


def reduce(input_):
    global _logged
    from sglang.srt.distributed import get_tp_group

    group = get_tp_group()
    if (group.world_size != 8 or input_.ndim != 2 or input_.shape[1] != 4096
            or not 8192 <= input_.shape[0] <= 36864
            or input_.dtype != torch.bfloat16 or not input_.is_cuda
            or not input_.is_contiguous() or torch.cuda.is_current_stream_capturing()):
        raise RuntimeError("FP32 attention AR diagnostic requires eager TP8 BF16 M8192..36864/H4096")
    work = input_.float()
    dist.all_reduce(work, group=group.device_group)
    output = work.to(input_.dtype)
    if not _logged:
        logging.getLogger(__name__).info(
            "DSV4 diagnostic attention AR selected: FP32 process-group sum, "
            "M=%d H=4096, temporary=%d bytes; decode/draft excluded",
            input_.shape[0], work.numel() * work.element_size())
        _logged = True
    return output
