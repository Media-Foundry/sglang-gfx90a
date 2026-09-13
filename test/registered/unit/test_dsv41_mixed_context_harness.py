"""CPU tests for the mixed-context coverage marker; no service required."""

import runpy
from pathlib import Path

import pytest


HELPERS = runpy.run_path(str(Path(__file__).resolve().parents[3] / "scripts/rocm/check_dsv41_mixed_context.py"))


@pytest.mark.parametrize("line,pending,overlap", [
    ("Prefill batch, #running-req: 0, #pending-token: 6196", 6196, False),
    ("Prefill batch, #running-req: 4, #pending-token: 3892", 3892, True),
    ("Prefill batch, #running-req: 1, #pending-token: 0", 0, True),
    ("Decode batch, #running-req: 4, #pending-token: 3000", None, False),
    ("Prefill batch, #pending-token: 3000", 3000, False),
    ("Prefill batch, #running-req: 4", None, False),
    ("", None, False),
])
def test_only_prefill_with_running_decode_counts_as_overlap(line, pending, overlap):
    assert HELPERS["pending_tokens"](line) == pending
    assert HELPERS["prefill_with_running_decode"](line) is overlap
