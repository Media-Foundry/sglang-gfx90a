"""CPU semantic and HIP-backend wiring tests for V4.1 two-level selection."""

from dataclasses import fields
from types import SimpleNamespace

import pytest
import torch

from sglang.srt.layers.attention.deepseek_v4_backend_hip_radix import (
    DSV4AttnMetadata,
    DSV4Metadata,
    DeepseekV4HipRadixBackend,
)
from sglang.srt.layers.attention.dsv4.dsv41_sparse import select_candidate_blocks


def _scalar_reference(logits, visible, budget, block_size):
    """Independent loop oracle, no tensor topk/argsort or production helper."""
    width = logits.shape[-1]
    out = torch.zeros(
        (len(logits), (width + block_size - 1) // block_size), dtype=torch.bool
    )
    for row, length in enumerate(visible):
        scores = [max(logits[row, begin : begin + block_size].tolist())
                  for begin in range(0, width, block_size)]
        if length > 0:
            scores[(length - 1) // block_size] = float("inf")
        ordered = sorted(range(len(scores)), key=lambda b: (-scores[b], b))
        for block in ordered[:budget]:
            out[row, block] = scores[block] > -float("inf")
    return out


@pytest.mark.parametrize("width", [0, 1, 7, 8, 9, 17, 16384, 16385, 32769])
@pytest.mark.parametrize("budget", [1, 2, 2048])
def test_candidate_blocks_causal_tail_ties_and_long_boundary(width, budget):
    gen = torch.Generator().manual_seed(83)
    logits = torch.randint(0, 5, (5, width), generator=gen).float()
    visible = torch.tensor([0, min(1, width), width // 2, max(0, width - 1), width])
    logits.masked_fill_(torch.arange(width)[None, :] >= visible[:, None], -torch.inf)
    before = logits.clone()
    actual = select_candidate_blocks(logits, visible[:, None], budget, 8)
    assert torch.equal(actual, _scalar_reference(logits, visible.tolist(), budget, 8))
    assert torch.equal(logits, before), "source scores must remain unchanged"
    assert torch.equal(actual.sum(-1), ((visible + 7) // 8).clamp_max(budget))


def test_newest_low_score_block_is_pinned_and_filler_dropped():
    logits = torch.tensor([[99., 98., 80., 70., -9.], [-torch.inf] * 5])
    actual = select_candidate_blocks(logits, torch.tensor([[5], [0]]), 2, 2)
    assert actual.tolist() == [[True, False, True], [False, False, False]]
    assert torch.equal(select_candidate_blocks(logits[:1], 5, 2, 2), actual[:1])


@pytest.mark.parametrize("budget,block", [(0, 8), (2, 0), (-1, 8)])
def test_invalid_candidate_geometry_is_rejected(budget, block):
    with pytest.raises(ValueError):
        select_candidate_blocks(torch.zeros(1, 8), 8, budget, block)


def _fixture():
    # Unequal request lengths and nontrivial physical page slots. Request 7
    # needs pruning, request 3 fits all blocks; neither may borrow the other.
    req = torch.tensor([7, 7, 3])
    pos = torch.tensor([10, 11, 7])
    page = torch.full((3, 4), -1, dtype=torch.int32)
    raw = page.clone()
    core = SimpleNamespace(
        candidate_blocks={}, candidate_ratio=None,
        sparse_page_indices=lambda ratio: page,
        sparse_raw_indices=lambda ratio: raw,
    )
    backend = SimpleNamespace(
        forward_metadata=SimpleNamespace(core_metadata=core),
        req_to_token=torch.arange(8 * 16).reshape(8, 16) + 100,
        token_to_kv_pool=SimpleNamespace(
            get_low_ratio_index_k_dequant=lambda layer, slots: slots[:, None]
        ),
    )
    indexer = SimpleNamespace(
        is_candidate_source=True, uses_candidates=False,
        candidate_block_size=4, candidate_topk_blocks=2, index_topk=4,
        queries=lambda q, freqs: q,
        head_weights=lambda x: x,
        scores=lambda q, k, w: -torch.arange(len(k)).float().repeat(len(q), 1),
    )
    layer = SimpleNamespace(
        layer_id=20, compress_ratio=1, indexer=indexer, freqs_cis=torch.zeros(16, 1)
    )
    x = torch.arange(3).float()[:, None]
    return backend, core, layer, x, req, pos, raw, page


def _run(backend, layer, x, req, pos):
    DeepseekV4HipRadixBackend._low_ratio_index_topk_torch(
        backend, layer, x, x, req, pos
    )


def test_source_to_consumer_respects_blocks_and_physical_mapping():
    backend, core, layer, x, req, pos, raw, page = _fixture()
    _run(backend, layer, x, req, pos)
    assert core.candidate_blocks[7].tolist() == [[True, False, True]] * 2
    assert core.candidate_blocks[3].tolist() == [[True, True]]
    assert raw.tolist() == [[0, 1, 2, 3]] * 3  # source still uses own full Top-K
    layer.indexer.is_candidate_source = False
    layer.indexer.uses_candidates = True
    layer.indexer.scores = lambda q, k, w: torch.arange(len(k)).float().repeat(len(q), 1)
    _run(backend, layer, x, req, pos)
    expected = torch.tensor([[3, 8, 9, 10], [8, 9, 10, 11], [4, 5, 6, 7]])
    assert torch.equal(raw, expected)
    assert torch.equal(page, backend.req_to_token[req[:, None], expected].int())
    for row in range(3):
        assert core.candidate_blocks[int(req[row])][
            row if row < 2 else 0, raw[row].long() // 4
        ].all()


def test_missing_source_and_wrong_ratio_fail_closed():
    backend, core, layer, x, req, pos, _, _ = _fixture()
    layer.indexer.is_candidate_source = False
    layer.indexer.uses_candidates = True
    with pytest.raises(RuntimeError, match="candidate source"):
        _run(backend, layer, x, req, pos)
    core.candidate_ratio = 2
    with pytest.raises(RuntimeError, match="same cache ratio"):
        _run(backend, layer, x, req, pos)
    core.candidate_ratio = 1
    with pytest.raises(RuntimeError, match="per-request"):
        _run(backend, layer, x, req, pos)


def test_next_forward_and_metadata_copy_drop_candidate_state():
    # Populate every dataclass field so the actual copy_metadata coverage
    # assertion catches newly-added fields missing from the copy contract.
    def empty_core():
        core = object.__new__(DSV4AttnMetadata)
        for item in fields(DSV4AttnMetadata):
            setattr(core, item.name, None)
        core.candidate_blocks = {7: torch.ones(1, 2, dtype=torch.bool)}
        core.candidate_ratio = 1
        return core

    source, dest = empty_core(), empty_core()
    dest.copy_(source)
    assert not dest.candidate_blocks and dest.candidate_ratio is None
    assert 7 in source.candidate_blocks and source.candidate_ratio == 1
    backend = SimpleNamespace(
        forward_metadata=DSV4Metadata(source, indexer_metadata=None), low_ratios=(1, 2)
    )
    DeepseekV4HipRadixBackend.init_forward_metadata_in_graph(
        backend, SimpleNamespace(out_cache_loc=None)
    )
    assert not source.candidate_blocks and source.candidate_ratio is None
