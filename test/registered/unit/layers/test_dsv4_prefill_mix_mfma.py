"""CPU negative-dispatch checks; GPU arithmetic is covered by the component oracle."""
import os
from types import SimpleNamespace
from unittest.mock import patch

from sglang.kernels.ops.layernorm.gfx90a_mhc_premix_mfma import try_premix_mfma
from sglang.srt.layers import dsv4_prefill_experiments as scope

FLAG = "SGLANG_DSV4_DEBUG_PREFILL_MIX_MFMA"


def test_disabled_does_not_access_inputs():
    with patch.dict(os.environ, {FLAG: "0"}):
        assert try_premix_mfma(None, None, None, None, 1.e-6) is False


def test_inactive_context_does_not_access_storage():
    token = scope._mix_pair.set(False)
    try:
        with patch.dict(os.environ, {FLAG: "1"}):
            x = SimpleNamespace(shape=(32768, 4, 4096))
            assert try_premix_mfma(x, None, None, None, 1.e-6) is False
    finally:
        scope._mix_pair.reset(token)


def test_bounds_and_unaligned_vector_fallback():
    token = scope._mix_pair.set(True)
    try:
        with patch.dict(os.environ, {FLAG: "1"}):
            for m in (0, 1, 128, 8191, 65537):
                assert try_premix_mfma(SimpleNamespace(shape=(m, 4, 4096)), None, None, None, 1.e-6) is False
            x = SimpleNamespace(shape=(32768, 4, 4096), data_ptr=lambda: 2)
            assert try_premix_mfma(x, None, None, None, 1.e-6) is False
            x.data_ptr = lambda: 16
            fn = SimpleNamespace(data_ptr=lambda: 4)
            assert try_premix_mfma(x, fn, None, None, 1.e-6) is False
    finally:
        scope._mix_pair.reset(token)
