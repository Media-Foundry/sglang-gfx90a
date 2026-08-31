from types import SimpleNamespace

import torch

from sglang.srt.speculative.dspark_components.dspark_worker_v2 import (
    _resolve_observer_target_logits,
)


def test_folded_observer_uses_graph_stable_strided_logits():
    logits = torch.arange(60, dtype=torch.float32).view(12, 5)
    epilogue = SimpleNamespace(strided_logits=logits)

    actual = _resolve_observer_target_logits(
        target_logits=None,
        folded_accept=True,
        epilogue=epilogue,
        bs=2,
        stride=3,
    )

    assert actual.data_ptr() == logits.data_ptr()
    assert torch.equal(actual, logits[:6])


def test_nonfolded_observer_never_reads_epilogue_fallback():
    epilogue = SimpleNamespace(strided_logits=torch.ones((8, 4)))
    assert (
        _resolve_observer_target_logits(
            target_logits=None,
            folded_accept=False,
            epilogue=epilogue,
            bs=2,
            stride=2,
        )
        is None
    )


def test_explicit_target_logits_take_precedence():
    direct = torch.randn((6, 5))
    epilogue = SimpleNamespace(strided_logits=torch.zeros_like(direct))
    assert (
        _resolve_observer_target_logits(
            target_logits=direct,
            folded_accept=True,
            epilogue=epilogue,
            bs=2,
            stride=3,
        )
        is direct
    )
