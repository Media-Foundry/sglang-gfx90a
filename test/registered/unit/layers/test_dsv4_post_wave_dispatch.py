"""CPU-only dispatch checks; numerical GPU oracles are recorded separately."""
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch

from sglang.kernels.ops.layernorm import gfx90a_mhc_post_fused4 as fused
from sglang.kernels.ops.layernorm import gfx90a_mhc_post_wave as wave
from sglang.srt.layers import dsv4_prefill_experiments as scope


class Kernel:
    def __init__(self):
        self.call = Mock()

    def __getitem__(self, grid):
        return self.call


@pytest.mark.parametrize(
    "enabled,active,rows,selected",
    [(False, True, 8192, False), (True, False, 8192, False),
     (True, True, 128, False), (True, True, 8192, True),
     (True, True, 65536, True), (True, True, 65537, False)],
)
def test_requires_explicit_flag_strict_context_and_large_rows(monkeypatch, enabled, active, rows, selected):
    device = SimpleNamespace(type="cuda")

    def tensor(shape, dtype):
        return SimpleNamespace(shape=shape, dtype=dtype, device=device, is_contiguous=lambda: True)

    x = tensor((rows, 4096), torch.bfloat16)
    residual = tensor((rows, 4, 4096), torch.bfloat16)
    post = tensor((rows, 4), torch.float32)
    comb = tensor((rows, 4, 4), torch.float32)
    output, partials = object(), object()
    kernel, hip = Kernel(), Mock()
    monkeypatch.setattr(torch.version, "hip", "test")
    monkeypatch.setattr(torch.cuda, "get_device_properties", lambda d: SimpleNamespace(gcnArchName="gfx90a"))
    monkeypatch.setattr(torch, "empty_like", lambda t: output)
    monkeypatch.setattr(torch, "empty", lambda *a, **kw: partials)
    monkeypatch.setattr(fused, "_post_combine_fused4", kernel)
    monkeypatch.setattr(wave, "run_post_wave", hip)
    monkeypatch.setenv("SGLANG_DSV4_DEBUG_PREFILL_POST_WAVE", str(int(enabled)))
    token = scope._mix_pair.set(active)
    try:
        assert fused.post_combine_fused4(x, residual, post, comb) == (output, partials)
    finally:
        scope._mix_pair.reset(token)
    assert hip.call_count == int(selected)
    assert kernel.call.call_count == int(not selected)
