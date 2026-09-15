from sglang.kernels.ops.debug.dsv4_h16_peer import eligible

def test_peer_scope_requires_accepted_native_stage_policy():
    assert eligible(True,32767,4,32768)
    assert eligible(True,32768,128,32768)
    assert eligible(True,8192,4,32768)
    assert not eligible(True,8192,128,32768)
    assert not eligible(False,32768,4,32768)
    assert not eligible(True,64,4,32768)
    assert not eligible(True,32769,4,32768)
    assert not eligible(True,32768,1,32768)
