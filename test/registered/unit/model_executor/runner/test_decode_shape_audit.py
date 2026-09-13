"""Audit reports executed graph rows, not just logical request concurrency."""
import os
from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch

from sglang.srt.model_executor.runner import decode_cuda_graph_runner as mod


@pytest.mark.parametrize("draft,ragged,bs,key,width,actual", [
    (False, False, 1, 1, 4, 4),
    (False, False, 2, 8, 4, 32),
    (False, False, 32, 32, 4, 128),
    (False, False, 64, 64, 4, 256),
    (True, False, 32, 32, 3, 96),
    (False, False, 64, 64, 1, 64),
    (False, True, 32, 128, 4, 128),
])
def test_shape_audit_deduplicates_and_preserves_real_rows(draft, ragged, bs, key, width, actual):
    runner = SimpleNamespace(
        model_runner=SimpleNamespace(is_draft_worker=draft),
        _replay_graph_key=SimpleNamespace(size=key),
        ragged_verify_mode=ragged, captured_req_width=width,
    )
    batch = SimpleNamespace(batch_size=bs, input_ids=torch.empty(bs*width),
                            forward_mode=SimpleNamespace(name="TEST"))
    with patch.dict(os.environ, {"SGLANG_LOG_DECODE_GRAPH_KEY_ONCE": "1"}), patch.object(mod.logger, "info") as log:
        for _ in range(2):
            mod.DecodeCudaGraphRunner._log_decode_graph_key(runner, batch)
        assert log.call_count == 1
        assert log.call_args.args[-2:] == (actual, bs*width)
    with patch.dict(os.environ, {"SGLANG_LOG_DECODE_GRAPH_KEY_ONCE": "0"}), patch.object(mod.logger, "info") as log:
        for _ in range(2):
            mod.DecodeCudaGraphRunner._log_decode_graph_key(runner, batch)
        assert log.call_count == 2
