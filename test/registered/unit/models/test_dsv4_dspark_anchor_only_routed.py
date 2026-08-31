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


def _ragged_batch(
    *, target_verify: bool = True, batch_size: int = 32, width: int = 6
):
    # Nonuniform gamma-five allocation with average width four: 32 anchors plus
    # a 96-row draft budget, hence an exact M128 target graph tier.
    verify_lens = torch.tensor([3, 4, 5, 4] * 8, dtype=torch.int32)
    qo_indptr = torch.nn.functional.pad(torch.cumsum(verify_lens, 0), (1, 0))
    layout = SimpleNamespace(
        graph_num_tokens=128,
        verify_lens=verify_lens,
        qo_indptr_device=qo_indptr,
    )
    return SimpleNamespace(
        forward_mode=_Mode(target_verify),
        batch_size=batch_size,
        spec_info=SimpleNamespace(
            num_tokens_per_req=width, ragged_verify_layout=layout
        ),
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


def test_dspark_m128_anchor_only_routed_guard_is_speculative_only(monkeypatch):
    monkeypatch.setenv("SGLANG_DSV4_GFX90A_DSPARK_M128_ANCHOR_ONLY_ROUTED", "1")
    monkeypatch.setattr(deepseek_v2, "is_gfx90a_supported", lambda: True)
    hidden = torch.empty((128, 4096))
    guard = deepseek_v2.DeepseekV2MoE._use_dspark_m128_anchor_only_routed

    assert guard(hidden, _batch(width=4))
    assert not guard(hidden, _batch(target_verify=False, width=4))
    assert not guard(hidden, _batch(batch_size=16, width=4))
    assert not guard(hidden, _batch(width=3))
    assert not guard(torch.empty((96, 4096)), _batch(width=4))
    assert not guard(hidden, None)


def test_dspark_m128_ragged_anchor_rows_use_device_indptr(monkeypatch):
    monkeypatch.setenv("SGLANG_DSV4_GFX90A_DSPARK_M128_ANCHOR_ONLY_ROUTED", "1")
    monkeypatch.setattr(deepseek_v2, "is_gfx90a_supported", lambda: True)
    hidden = torch.empty((128, 4096))
    rows = deepseek_v2.DeepseekV2MoE._dspark_m128_ragged_anchor_rows

    batch = _ragged_batch()
    expected = batch.spec_info.ragged_verify_layout.qo_indptr_device[:-1].long()
    assert torch.equal(rows(hidden, batch), expected)
    assert rows(hidden, _ragged_batch(target_verify=False)) is None
    assert rows(hidden, _ragged_batch(batch_size=16)) is None
    assert rows(hidden, _ragged_batch(width=4)) is None
    assert rows(torch.empty((96, 4096)), batch) is None
    batch.spec_info.ragged_verify_layout.graph_num_tokens = 96
    assert rows(hidden, batch) is None
