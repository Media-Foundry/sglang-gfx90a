"""Check actual logger formats; do not invent rank tags on legacy prints."""
def check_paths(text, runtime_m):
    lines = text.splitlines()
    # Legacy empty-tile print has no rank tag. Match its eight observations;
    # rank-specific evidence comes from the consuming query path below.
    assert sum('prefill empty tiles selected:' in line for line in lines) >= 8
    for rank in range(8):
        for hit in ('prefill post-fused4 selected:', 'prefill mix-reuse4 selected:'):
            assert any(f'TP{rank}]' in line and hit in line for line in lines), (rank, hit)
        assert any(f'TP{rank}]' in line and 'prefill query-reuse4 selected' in line
            and 'query_group=16' in line and f'runtime_m={int(runtime_m)}' in line
            for line in lines), rank
