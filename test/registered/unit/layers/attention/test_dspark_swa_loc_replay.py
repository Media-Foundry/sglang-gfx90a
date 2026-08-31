from types import SimpleNamespace

import torch

from sglang.srt.layers.attention.deepseek_v4_backend_hip_radix import (
    DeepseekV4HipRadixBackend,
)


class _Mode:
    def __init__(self, *, target_verify: bool):
        self._target_verify = target_verify

    def is_target_verify(self):
        return self._target_verify

    def is_decode_or_idle(self):
        return False

    def is_idle(self):
        return False


def _backend(*, is_dspark_draft: bool):
    cached = torch.tensor([77, 88], dtype=torch.int32)
    return SimpleNamespace(
        is_dspark_draft=is_dspark_draft,
        speculative_num_steps=1,
        token_to_kv_pool=SimpleNamespace(unified_swa_ring_size=256),
        forward_metadata=SimpleNamespace(
            core_attn_metadata=SimpleNamespace(
                unified=SimpleNamespace(swa_loc=cached)
            )
        ),
    )


def _batch(*, target_verify: bool):
    return SimpleNamespace(
        forward_mode=_Mode(target_verify=target_verify),
        positions=torch.tensor([1, 130], dtype=torch.int64),
        req_pool_indices=torch.tensor([2, 3], dtype=torch.int64),
    )


def test_dspark_draft_verify_recomputes_swa_loc_from_live_inputs():
    backend = _backend(is_dspark_draft=True)
    result = DeepseekV4HipRadixBackend.get_unified_swa_loc(
        backend, _batch(target_verify=True)
    )

    assert torch.equal(result, torch.tensor([513, 898], dtype=torch.int32))
    cached = backend.forward_metadata.core_attn_metadata.unified.swa_loc
    assert result.data_ptr() != cached.data_ptr()


def test_target_and_native_paths_keep_cached_swa_loc():
    backend = _backend(is_dspark_draft=False)
    cached = backend.forward_metadata.core_attn_metadata.unified.swa_loc
    result = DeepseekV4HipRadixBackend.get_unified_swa_loc(
        backend, _batch(target_verify=True)
    )
    assert result is cached

    draft_backend = _backend(is_dspark_draft=True)
    cached = draft_backend.forward_metadata.core_attn_metadata.unified.swa_loc
    result = DeepseekV4HipRadixBackend.get_unified_swa_loc(
        draft_backend, _batch(target_verify=False)
    )
    assert result is cached
