"""Opt-in native TP8 M32 graph-only legacy AR geometry; no new buffers."""
import logging

import torch

from sglang.srt.environ import envs
from .dsv4_ar_experiment import _active


def eligible(*, active, world, shape, bf16, contiguous, quantized, registered, capturing):
    return bool(active and world == 8 and tuple(shape) == (32, 4096)
                and bf16 and contiguous and not quantized and registered and capturing)


def adapt_geometry(parent):
    blocks = envs.SGLANG_DSV4_GFX90A_TP8_M32_AR_BLOCKS.get()
    if blocks == 0:
        return parent
    from sglang.srt.server_args import get_global_server_args

    args = get_global_server_args()
    if (blocks not in (4, 8, 16)
            or not envs.SGLANG_DSV4_GFX90A_TP8_M32_LEGACY_AR.get()
            or args.tp_size != 8 or args.ep_size != 1 or args.dp_size != 1
            or args.pp_size != 1 or args.enable_dp_attention
            or args.speculative_algorithm is not None
            or args.max_total_tokens != 1048576):
        raise RuntimeError('AR geometry requires native TP8/EP1, 1M KV, legacy AR, blocks4/8/16')

    class GeometryAR(parent):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            if (self.disabled or self.world_size != 8 or not torch.version.hip
                    or torch.cuda.get_device_properties(self.device).gcnArchName.split(':')[0] != 'gfx90a'):
                raise RuntimeError('AR geometry requires enabled gfx90a TP8 communicator')
            import aiter
            from sglang.kernels.ops.debug.gfx90a_tp8_ar_geometry_oracle import module

            # Load before the outer capture scope: no JIT or ABI probing inside
            # a captured forward, and no change to the parent's warmup behavior.
            self._geometry_module = module()
            if self._geometry_module.signal_bytes() != aiter.meta_size():
                raise RuntimeError('AR geometry Signal ABI mismatch')
            self._geometry_logged = False

        def all_reduce(self, inp, *, out=None, use_new=True,
                       open_fp8_quant=False, registered=False):
            if eligible(active=_active.get(), world=self.world_size, shape=inp.shape,
                        bf16=inp.dtype == torch.bfloat16, contiguous=inp.is_contiguous(),
                        quantized=open_fp8_quant, registered=registered,
                        capturing=torch.cuda.is_current_stream_capturing()):
                if out is None:
                    out = torch.empty_like(inp)
                if not out.is_contiguous() or out.device != inp.device:
                    raise RuntimeError('AR geometry output must be contiguous on the input device')
                self._geometry_module.run(self._ptr, inp, out, blocks)
                if not self._geometry_logged:
                    logging.getLogger(__name__).info(
                        'DSV4 native TP8 M32 registered graph AR selected blocks=%d', blocks)
                    self._geometry_logged = True
                return out
            # In particular, retain unregistered eager copies and all other
            # modes. Parent custom_all_reduce still owns graph registration.
            return super().all_reduce(inp, out=out, use_new=use_new,
                                      open_fp8_quant=open_fp8_quant, registered=registered)

    return GeometryAR
