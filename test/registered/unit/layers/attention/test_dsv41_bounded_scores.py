"""Bounded V4.1 score workspace preserves independent query arithmetic."""

import pytest
import torch

from sglang.srt.layers.attention.dsv4.dsv41_sparse import bounded_indexer_scores


@pytest.mark.parametrize("m,n", [(0, 7), (1, 0), (1, 17), (7, 97), (31, 128)])
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float32])
def test_bounded_scores_same_rounding_and_no_input_mutation(m, n, dtype):
    gen = torch.Generator().manual_seed(71)
    q = (torch.randint(-4, 5, (m, 4, 32), generator=gen).float() * .5).to(dtype)
    k = (torch.randint(-4, 5, (n, 32), generator=gen).float() * .5).to(dtype)
    w = (torch.randint(-4, 5, (m, 4), generator=gen).float() * .125).to(dtype)
    inputs = [x.clone() for x in (q, k, w)]
    scores = torch.einsum("bhd,nd->bhn", q, k)
    reference = (scores.relu() * w.unsqueeze(-1)).sum(dim=1).float()
    # Exercise both single-query and ragged multi-query slabs, and one slab.
    for rows in (1, 3, 128):
        budget = max(1, rows * 4 * n * q.element_size())
        actual = bounded_indexer_scores(q, k, w, max_slab_bytes=budget)
        assert actual.dtype == torch.float32 and torch.equal(actual, reference)
    assert all(torch.equal(a, b) for a, b in zip(inputs, (q, k, w)))


def test_bounded_scores_rejects_mixed_weight_dtype_and_invalid_budget():
    q = torch.ones(2, 4, 32, dtype=torch.bfloat16)
    k = torch.ones(17, 32, dtype=torch.bfloat16)
    with pytest.raises(ValueError, match="dtype"):
        bounded_indexer_scores(q, k, torch.ones(2, 4, dtype=torch.float32))
    with pytest.raises(ValueError, match="budget"):
        bounded_indexer_scores(q, k, torch.ones(2, 4, dtype=q.dtype), max_slab_bytes=0)
