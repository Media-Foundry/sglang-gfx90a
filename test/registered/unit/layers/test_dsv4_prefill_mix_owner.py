import os
from types import SimpleNamespace
from unittest.mock import patch
import pytest

from sglang.kernels.ops.layernorm.gfx90a_mhc_premix_owner import partition, try_premix_owner
from sglang.kernels.ops.debug.dsv4_premix_owner_audit import owned_rows
from sglang.srt.layers import dsv4_prefill_experiments as scope

FLAG = 'SGLANG_DSV4_DEBUG_PREFILL_MIX_OWNER'


def test_disabled_no_storage_access():
    with patch.dict(os.environ, {FLAG:'0'}):
        assert try_premix_owner(None,None,None,1.e-6) is None


def test_partition_matches_oracle():
    for m in (8192,8193,32767,32768,65536):
        ranges=[partition(m,r) for r in range(8)]
        assert len({p[2] for p in ranges})==1
        for r,(start,end,capacity) in enumerate(ranges):
            assert (start,end)==owned_rows(m,r)
            assert start%8==0 and end-start<=capacity
        assert sum(b-a for a,b,_ in ranges)==m


def test_scope_bounds_capture():
    for active in (False,True):
        token=scope._mix_pair.set(active)
        try:
            with patch.dict(os.environ,{FLAG:'1'}):
                if not active:
                    assert try_premix_owner(None,None,None,1.e-6) is None
                for m in (1,128,8191,65537):
                    assert try_premix_owner(SimpleNamespace(shape=(m,)),None,None,1.e-6) is None
                with patch('torch.cuda.is_current_stream_capturing',return_value=True):
                    assert try_premix_owner(SimpleNamespace(shape=(32768,)),None,None,1.e-6) is None
        finally:
            scope._mix_pair.reset(token)


def test_conflicting_candidates_fail_before_collective():
    token=scope._mix_pair.set(True)
    try:
        with patch.dict(os.environ,{FLAG:'1','SGLANG_DSV4_DEBUG_PREFILL_MIX_MFMA':'1'}), \
             patch('torch.cuda.is_current_stream_capturing',return_value=False):
            with pytest.raises(ValueError,match='independent candidates'):
                try_premix_owner(SimpleNamespace(shape=(32768,)),None,None,1.e-6)
    finally:
        scope._mix_pair.reset(token)
