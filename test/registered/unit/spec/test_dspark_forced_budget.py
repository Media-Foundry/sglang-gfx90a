from sglang.srt.speculative.dspark_components.dspark_planner import (
    should_use_uniform_verify_all_layout,
)


def test_forced_budget_bypasses_uninitialized_table_verify_all_shortcut():
    assert should_use_uniform_verify_all_layout(
        is_verify_all=True, forced_budget_frac=None
    )
    assert not should_use_uniform_verify_all_layout(
        is_verify_all=True, forced_budget_frac=0.2
    )
    assert not should_use_uniform_verify_all_layout(
        is_verify_all=False, forced_budget_frac=None
    )
