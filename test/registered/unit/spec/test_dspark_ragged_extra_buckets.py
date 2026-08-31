import pytest

from sglang.srt.model_executor.runner.decode_cuda_graph_runner import (
    DecodeCudaGraphRunner,
)


def _runner():
    runner = object.__new__(DecodeCudaGraphRunner)
    runner.capture_bs = [1, 32]
    runner.captured_req_width = 6
    return runner


def test_dspark_ragged_extra_token_bucket(monkeypatch):
    monkeypatch.setenv("SGLANG_DSPARK_RAGGED_VERIFY_EXTRA_TOKEN_BUCKETS", "128")
    assert _runner()._build_ragged_verify_token_buckets() == [6, 128, 192]


def test_dspark_ragged_token_bucket_override(monkeypatch):
    monkeypatch.setenv("SGLANG_DSPARK_RAGGED_VERIFY_EXTRA_TOKEN_BUCKETS", "96")
    monkeypatch.setenv("SGLANG_DSPARK_RAGGED_VERIFY_TOKEN_BUCKETS", "128")
    assert _runner()._build_ragged_verify_token_buckets() == [128]


@pytest.mark.parametrize("value", ["0", "193", "not-an-int"])
def test_dspark_ragged_extra_token_bucket_rejects_invalid(monkeypatch, value):
    monkeypatch.setenv("SGLANG_DSPARK_RAGGED_VERIFY_EXTRA_TOKEN_BUCKETS", value)
    with pytest.raises(ValueError):
        _runner()._build_ragged_verify_token_buckets()
