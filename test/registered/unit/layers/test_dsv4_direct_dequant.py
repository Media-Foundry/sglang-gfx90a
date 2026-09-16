import os
from unittest.mock import patch

from sglang.kernels.ops.moe.gfx90a_bf16_direct_rows import eligible
from sglang.srt.layers import dsv4_prefill_experiments as scope


def test_default_off():
    with patch.dict(os.environ,{},clear=True):assert not eligible(32768,256,True)


def test_exact_scope_and_layout():
    with patch.dict(os.environ,{'SGLANG_DSV4_DEBUG_CK_DIRECT_DEQUANT':'1'}), \
         patch('torch.cuda.is_current_stream_capturing',return_value=False):
        assert not eligible(32768,256,True)
        token=scope._mix_pair.set(True)
        try:
            for rows in (8192,32767,32768,36864):assert eligible(rows,256,True)
            for rows in (1,128,8191,36865,65536):assert not eligible(rows,256,True)
            assert not eligible(32768,512,True)
            assert not eligible(32768,256,False)
            with patch('torch.cuda.is_current_stream_capturing',return_value=True):
                assert not eligible(32768,256,True)
        finally:scope._mix_pair.reset(token)
