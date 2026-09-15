import numpy as np
from sglang.kernels.ops.debug.dsv4_attention_peer_capture import canonical_slots, canonical_csr


def test_physical_renumbering_preserves_occurrences():
    used,a=canonical_slots([90,-1,5,90,2,5])
    other,b=canonical_slots([12,-1,31,12,7,31])
    assert used.tolist()==[90,5,2] and other.tolist()==[12,31,7]
    assert np.array_equal(a,b) and a.tolist()==[0,-1,1,0,2,1]


def test_empty_slots():
    for x in ([],[-1,-1]):
        used,y=canonical_slots(x)
        assert len(used)==0 and y.tolist()==x


def test_ignore_unused_capacity_and_rebase():
    used,ids,ptr=canonical_csr([999999,9,4,9,-1,99999999],[1,3,5,99999],2)
    assert used.tolist()==[9,4] and ids.tolist()==[0,1,0,-1]
    assert ptr.tolist()==[0,2,4]
