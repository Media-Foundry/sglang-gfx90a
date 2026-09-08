"""Default-off TP8 M32 wo_a hipBLASLt experiment; no weight conversion."""
import logging

import torch

from sglang.srt.environ import envs

_ready_devices = set()


def eligible(*, enabled, hip, arch, native, decode, batch_size, tp, attn_tp, ep):
    return bool(enabled and hip and arch == "gfx90a" and native and decode
                and batch_size == 32 and tp == 8 and attn_tp == 8 and ep == 1)


def maybe_woa(x, weight, batch, attn_tp):
    if not envs.SGLANG_DSV4_GFX90A_TP8_M32_WOA_HIPBLASLT.get():
        return None
    from sglang.srt.runtime_context import get_parallel

    p = get_parallel()
    spec = batch.spec_algorithm
    if not eligible(enabled=True, hip=bool(torch.version.hip),
        arch=torch.cuda.get_device_properties(x.device).gcnArchName.split(':')[0]
        if torch.version.hip else '', native=spec is None or spec.is_none(),
        decode=batch.forward_mode.is_decode(), batch_size=batch.batch_size,
        tp=p.tp_size, attn_tp=attn_tp, ep=p.moe_ep_size):
        return None
    if (x.shape != (32,1,4096) or weight.shape != (1,1024,4096)
        or x.dtype != torch.bfloat16 or weight.dtype != torch.bfloat16
        or not x.is_cuda or x.device != weight.device
        or not x.is_contiguous() or not weight.is_contiguous()):
        return None

    from aiter import tuned_gemm

    key = x.device.index
    if key not in _ready_devices:
        # Enumerate once during eager graph warmup, never inside capture.
        if torch.cuda.is_current_stream_capturing():
            raise RuntimeError("wo_a solution must be checked before graph capture")
        from aiter.ops.gradlib import hipb_create_extension, hipb_findallsols

        # Share the initializer state used by hipb_gemm. Calling the raw
        # initializer and then hipb_gemm without this guard allocates the
        # 256MiB global workspace twice (the first pointer is overwritten).
        if not tuned_gemm.extensions_created:
            hipb_create_extension()
            tuned_gemm.extensions_created = True
        supported = hipb_findallsols(x[:,0], weight[0].t(), out_dtype=x.dtype)
        if 4429 not in supported:
            raise RuntimeError("wo_a solution4429 unavailable in installed hipBLASLt")
        _ready_devices.add(key)
        logging.getLogger(__name__).info(
            "DSV4 TP8 native M32 wo_a selected: hipBLASLt4429; HIP=%s Torch=%s; "
            "experimental non-bitexact reduction", torch.version.hip, torch.__version__
        )
    return tuned_gemm.hipb_gemm(x[:,0], weight[0], 4429).unsqueeze(1)
