"""Default-off original-V4 TP8 large-prefill dequant supply, unchanged output bits."""
import logging
import os

import torch

from sglang.kernels.jit.utils import cache_once, load_jit


def eligible(rows, intermediate, shuffled):
    if os.getenv('SGLANG_DSV4_DEBUG_CK_DIRECT_DEQUANT','0') != '1':
        return False
    from sglang.srt.layers.dsv4_prefill_experiments import mix_pair_active
    return (shuffled and intermediate == 256 and 8192 <= rows <= 36864
            and mix_pair_active() and not torch.cuda.is_current_stream_capturing())


@cache_once
def direct_module():
    return load_jit('gfx90a_bf16_direct_rows',
        cuda_files=['deepseek_v4/gfx90a_fp4_bf16_direct_rows.cuh'],
        cuda_wrappers=[('gate','sglang::Gfx90aFp4Bf16DirectRows<512,4096,1664>::run'),
                       ('down','sglang::Gfx90aFp4Bf16DirectRows<4096,256,416>::run')],
        extra_cuda_cflags=['-O3'])


@cache_once
def announce():
    logging.getLogger(__name__).warning('DSV4 direct-row CK dequant selected: gate=1664 down=416')


def run(name, weight, scale, output, reference):
    # Parent helper validated checkpoint shape and converted scales to logical
    # layout. The FFI checks shape/dtype/device/alignment before dispatch.
    assert name in ('gate','down')
    getattr(direct_module(),name)(weight,scale,output)
    if os.getenv('SGLANG_DSV4_DEBUG_CK_DIRECT_DEQUANT_CHECK','0') == '1':
        expected=torch.empty_like(output)
        reference(weight,scale,expected)
        assert torch.equal(output.view(torch.int16),expected.view(torch.int16)),name
        logging.getLogger(__name__).warning('DSV4 direct-row dequant exact: %s',name)
    announce()
