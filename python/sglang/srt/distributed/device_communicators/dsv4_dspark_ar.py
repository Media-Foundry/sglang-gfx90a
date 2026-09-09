"""Default-off TP8 DSpark target graph CTA tuning; native AR is untouched."""
import logging

import torch

from sglang.srt.environ import envs
from .dsv4_ar_experiment import _dspark_m128_active


def eligible(*, active, world, shape, bf16, contiguous, quantized, registered, capturing):
    return bool(active and world == 8 and tuple(shape) == (128, 4096)
                and bf16 and contiguous and not quantized and registered and capturing)


def adapt_dspark_ar(parent):
    blocks = envs.SGLANG_DSV4_GFX90A_DSPARK_TP8_M128_AR_BLOCKS.get()
    if blocks == 0:
        return parent
    from sglang.srt.server_args import get_global_server_args
    args = get_global_server_args()
    if (blocks not in (12, 80) or str(args.speculative_algorithm).upper() != 'DSPARK'
            or args.tp_size != 8 or args.ep_size != 1 or args.dp_size != 1
            or args.pp_size != 1 or args.enable_dp_attention):
        raise RuntimeError('DSpark AR grid requires TP8/EP1/DP1/PP1, DSpark, blocks12/80')

    class DsparkAR(parent):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            if (self.disabled or self.world_size != 8 or not torch.version.hip
                    or torch.cuda.get_device_properties(self.device).gcnArchName.split(':')[0] != 'gfx90a'):
                raise RuntimeError('DSpark AR grid requires enabled gfx90a TP8 communicator')
            import aiter
            from sglang.kernels.ops.debug.gfx90a_tp8_dspark_ar_oracle import module
            self._dspark_ar_module = module()
            if self._dspark_ar_module.signal_bytes() != aiter.meta_size():
                raise RuntimeError('DSpark AR Signal ABI mismatch')
            self._dspark_ar_logged = False

        def all_reduce(self, inp, *, out=None, use_new=True,
                       open_fp8_quant=False, registered=False):
            if eligible(active=_dspark_m128_active.get(), world=self.world_size,
                        shape=inp.shape, bf16=inp.dtype == torch.bfloat16,
                        contiguous=inp.is_contiguous(), quantized=open_fp8_quant,
                        registered=registered, capturing=torch.cuda.is_current_stream_capturing()):
                if out is None:
                    out = torch.empty_like(inp)
                if (out.shape != inp.shape or out.dtype != inp.dtype
                        or out.device != inp.device or not out.is_contiguous()):
                    raise RuntimeError('invalid DSpark AR output buffer')
                self._dspark_ar_module.run(self._ptr, inp, out, blocks)
                if not self._dspark_ar_logged:
                    logging.getLogger(__name__).info(
                        'DSV4 TP8 DSpark target M128 AR hit rank=%s blocks=%s', self.rank, blocks)
                    self._dspark_ar_logged = True
                return out
            return super().all_reduce(inp, out=out, use_new=use_new,
                                      open_fp8_quant=open_fp8_quant, registered=registered)

    return DsparkAR
