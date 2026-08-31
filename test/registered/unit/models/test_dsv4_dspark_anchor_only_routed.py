from types import SimpleNamespace

import torch

import sglang.srt.models.deepseek_v2 as deepseek_v2


class _Mode:
    def __init__(self, target_verify: bool):
        self._target_verify = target_verify

    def is_target_verify(self) -> bool:
        return self._target_verify


def _batch(*, target_verify: bool = True, batch_size: int = 32, width: int = 2):
    return SimpleNamespace(
        forward_mode=_Mode(target_verify),
        batch_size=batch_size,
        spec_info=SimpleNamespace(num_tokens_per_req=width),
    )


def test_dspark_anchor_only_routed_guard_is_speculative_only(monkeypatch):
    monkeypatch.setenv("SGLANG_DSV4_GFX90A_DSPARK_M64_ANCHOR_ONLY_ROUTED", "1")
    monkeypatch.setattr(deepseek_v2, "is_gfx90a_supported", lambda: True)
    hidden = torch.empty((64, 4096))
    guard = deepseek_v2.DeepseekV2MoE._use_dspark_m64_anchor_only_routed

    assert guard(hidden, _batch())
    assert not guard(hidden, _batch(target_verify=False))
    assert not guard(hidden, _batch(batch_size=16))
    assert not guard(hidden, _batch(width=4))
    assert not guard(torch.empty((32, 4096)), _batch())
    assert not guard(hidden, None)
