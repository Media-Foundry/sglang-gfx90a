import importlib.util
from pathlib import Path

import numpy as np
import pytest
import torch


spec = importlib.util.spec_from_file_location('frontier_analysis', Path(__file__).with_name('analyze.py'))
analysis = importlib.util.module_from_spec(spec)
spec.loader.exec_module(analysis)


def test_spans_preserve_request_and_prefix_boundaries():
    metadata = dict(metadata=dict(extend_lens=[3, 4]), positions=torch.tensor([0, 1, 2, 9, 10, 11, 12]))
    result = analysis.spans(np.array([1, 2, 3, 4, 6]), metadata)
    assert [(r['case'], r['first_position'], r['last_position']) for r in result] == [
        (0, 1, 2), (1, 9, 10), (1, 12, 12)]
    assert sum(r['last_row']-r['first_row']+1 for r in result) == 5


def test_payload_corruption_fails_closed(tmp_path):
    path = tmp_path/'sample.pt'
    torch.save(dict(row_hashes=torch.zeros(2, 16, dtype=torch.uint8)), path)
    with pytest.raises(AssertionError):
        analysis.load_values(tmp_path, dict(values_file=path.name, file_sha256='incorrect'))
