"""CPU explanation of the invalid-store wrap; GPU validation lives in screen.py."""
from pathlib import Path


def test_padding_address_wrap_matches_observed_slot():
    row_bytes=4096*4
    for m in (8192,21845):
        extent=m*6*row_bytes
        invalid=(extent+0x80000000)&0xffffffff
        assert invalid>=extent
    for m in (21846,32767,36864):
        extent=m*6*row_bytes
        invalid=(extent+0x80000000)&0xffffffff
        assert invalid<extent
    assert (((32767*6*row_bytes+0x80000000)&0xffffffff)//row_bytes)==65530


def test_overlay_explicitly_predicates_set():
    root=Path(__file__).resolve().parent
    source=(root/'overlay/ck/tensor_operation/gpu/thread/threadwise_tensor_slice_transfer_v7r3_scatter.hpp').read_text()
    needle='if constexpr(DstInMemOp == InMemoryDataOperationEnum::Set)'
    assert source.count(needle)==1
    branch=source.split(needle,1)[1].split('else',1)[0]
    assert 'if(is_dst_valid)' in branch
    assert 'dst_offset, true,' in branch
