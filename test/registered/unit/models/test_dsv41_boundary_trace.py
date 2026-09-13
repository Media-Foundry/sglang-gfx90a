from types import SimpleNamespace

import torch

from scripts.rocm.dsv41_boundary_trace import snapshot, wrap_method


def test_wrapper_preserves_output_identity_and_bounds():
    saved, outputs = [], []

    def original(self, x, *, scale=1):
        y = x * scale
        outputs.append(y)
        return y

    wrapped = wrap_method(original, "test", lambda *args: saved.append(args), lambda _: True, 1)
    obj, x = SimpleNamespace(), torch.ones(2, 3)
    y = wrapped(obj, x, scale=2)
    assert y is outputs[0]
    assert len(saved) == 1
    x.zero_()
    assert saved[0][3]["x"].eq(1).all()
    y.zero_()
    assert saved[0][4].eq(2).all()
    wrapped(obj, x)
    assert len(outputs) == 2 and len(saved) == 1


def test_unselected_instance_is_not_captured():
    saved = []
    wrapped = wrap_method(lambda self, x: x, "test", lambda *a: saved.append(a), lambda _: False, 1)
    x = torch.ones(1)
    assert wrapped(object(), x) is x
    assert not saved


def test_snapshot_retains_nested_values_without_aliasing():
    x = torch.ones(2)
    copy = snapshot({"pair": (x, None), "value": 3})
    x.zero_()
    assert copy["pair"][0].eq(1).all()
    assert copy["pair"][1] is None and copy["value"] == 3
