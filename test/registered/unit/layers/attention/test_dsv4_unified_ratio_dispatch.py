"""CPU constructor regression: unified V4 has no paged C4/C128 dictionary."""

from types import SimpleNamespace

import pytest
import torch

from sglang.srt.layers.attention import deepseek_v4_backend_hip_radix as hip
from sglang.srt.mem_cache.deepseek_v4_memory_pool import DeepSeekV4TokenToKVPool


@pytest.mark.parametrize(
    "unified,sources,paged,expected",
    [
        (True, {4: [2], 128: [3]}, {}, (4, 128)),
        (True, {4: [0]}, {}, (4,)),
        (True, {}, {}, ()),
        (False, {1: [0], 2: [1]}, {1: object(), 2: object()}, (1, 2)),
        (False, {4: [2], 128: [3]}, {4: object(), 128: object()}, (4, 128)),
    ],
)
def test_constructor_uses_active_storage_ratio_contract(monkeypatch, unified, sources, paged, expected):
    pool = object.__new__(DeepSeekV4TokenToKVPool)
    pool._unified_kv = unified
    pool.sources_by_ratio = sources
    pool.kv_pools = paged
    spec = SimpleNamespace(speculative_eagle_topk=1, speculative_num_draft_tokens=4)
    monkeypatch.setattr(hip, "get_spec", lambda: spec)
    runner = SimpleNamespace(
        device="cpu", page_size=256, token_to_kv_pool=pool,
        req_to_token_pool=SimpleNamespace(req_to_token=torch.zeros((4, 1024), dtype=torch.int32)),
        hisparse_coordinator=None, is_draft_worker=False,
        model_config=SimpleNamespace(head_dim=512, v_head_dim=512,
                                     hf_text_config=SimpleNamespace(index_topk=512)),
        server_args=SimpleNamespace(enable_deepseek_v4_fp4_indexer=False),
        spec_algorithm=SimpleNamespace(is_dspark=lambda: True),
    )
    backend = hip.DeepseekV4HipRadixBackend(runner)
    assert backend.present_ratios == expected
    assert backend.has_c4 == (4 in expected)
    assert backend.has_c128 == (128 in expected)
    assert backend.low_ratios == tuple(r for r in expected if r in (1, 2))
    if backend.has_c4:
        monkeypatch.setattr(hip, "_create_flashmla_metadata", lambda: None)
        core = object.__new__(hip.DSV4AttnMetadata)
        core.c4_sparse_topk = 512
        core.present_ratios = expected
        core.low_ratios = ()
        core.c4_topk_lengths_raw = torch.tensor([0, 3, 576], dtype=torch.int32)
        core.c4_topk_lengths_clamp1 = core.c4_topk_lengths_raw.clamp_min(1)
        core.init_flashmla_related(is_prefill=True)
        assert core.c4_sparse_page_indices.shape == (3, 512)
        assert core.c4_sparse_topk_lengths_raw.tolist() == [0, 3, 512]


@pytest.mark.parametrize("ratios", [(), (4,), (128,), (4, 128)])
def test_unified_streams_allow_swa_only_draft_and_single_ratio(monkeypatch, ratios):
    from sglang.kernels.ops.attention.dsv4.unified_kv_kernels import env_gate, runtime

    monkeypatch.setattr(env_gate, "is_unified_kv_triton", lambda: True)
    swa_indices = torch.arange(16, dtype=torch.int32)
    swa_indptr = torch.tensor([0, 8, 16], dtype=torch.int32)
    monkeypatch.setattr(runtime, "build_swa_decode_stream", lambda **kw: (swa_indices, swa_indptr))
    calls = []
    def compressed(**kw):
        calls.append(kw)
        assert kw["hca_len"] is not None and kw["csa_len"] is not None
        assert kw["csa_width"] == (512 if 4 in ratios else 0)
        assert kw["hca_page_indices"].shape == (2, 8 if 128 in ratios else 0)
        return (swa_indices, swa_indptr) * 3
    monkeypatch.setattr(runtime, "build_decode_streams", compressed)
    core = SimpleNamespace(
        positions_casual=torch.tensor([130, 131]),
        swa_topk_lengths=torch.tensor([128, 128], dtype=torch.int32),
        unified=hip.UnifiedKvMetadata(),
        c4_sparse_page_indices=torch.empty((2, 512), dtype=torch.int32) if 4 in ratios else None,
        c128_page_indices=torch.empty((2, 8), dtype=torch.int32) if 128 in ratios else None,
        c4_sparse_topk_lengths_raw=torch.tensor([32, 33]) if 4 in ratios else None,
        c128_topk_lengths_raw=torch.tensor([1, 1]) if 128 in ratios else None,
    )
    backend = SimpleNamespace(token_to_kv_pool=SimpleNamespace(
        unified_swa_window=128, unified_swa_ring_size=131, unified_swa_pages=1024))
    state_slot = torch.tensor([2, 2], dtype=torch.int32)
    hip.DeepseekV4HipRadixBackend._attach_unified_kv_decode_streams(backend, core, state_slot)
    assert core.unified.swa_loc.tolist() == [392, 262]
    assert core.unified.swa_indices is swa_indices
    assert core.unified.verify_store_state_slot is not None
    assert len(calls) == int(bool(ratios))
    if not ratios:
        assert core.unified.csa_indices is None and core.unified.hca_indices is None
