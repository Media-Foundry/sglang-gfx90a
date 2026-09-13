"""Do not use reporting Top-K order as a proxy for committed token equality."""

import runpy
from pathlib import Path

import pytest

HELPERS = runpy.run_path(str(Path(__file__).resolve().parents[3] / "scripts/rocm/check_dsv41_cache_prefix.py"))


@pytest.mark.parametrize("cached,actual,ranked_equal,passes", [
    (7, 7, True, True),
    (7, 7, False, True),
    (7, 9, True, False),
    (7, 9, False, False),
])
def test_commits_not_ranked_topk_determine_equality(cached, actual, ranked_equal, passes):
    result = HELPERS["summarize_comparisons"]([
        {"cached_committed_id": cached, "recompute_committed_id": actual,
         "top1_equal": ranked_equal},
    ])
    assert result["all_committed_ids_equal"] is passes
    assert result["all_top1_equal"] is ranked_equal


def test_empty_or_partly_diverged_set_does_not_pass():
    summarize = HELPERS["summarize_comparisons"]
    assert summarize([]) == {"all_committed_ids_equal": False, "all_top1_equal": False}
    rows = [{"cached_committed_id": 1, "recompute_committed_id": 1, "top1_equal": True},
            {"cached_committed_id": 2, "recompute_committed_id": 3, "top1_equal": True}]
    assert summarize(rows)["all_committed_ids_equal"] is False
