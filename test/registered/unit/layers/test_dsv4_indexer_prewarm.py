import importlib.util
from pathlib import Path

import pytest

path=Path(__file__).resolve().parents[4]/'python/sglang/kernels/ops/attention/dsv4/gfx90a_indexer_prewarm.py'
spec=importlib.util.spec_from_file_location('tested_prewarm',path)
module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)


@pytest.mark.parametrize('values',[(512,8,8,0),(4096,64,64,16),(8192,128,128,16),(8192,128,131,8)])
def test_valid(values):
    module.validate_signature(*values)


@pytest.mark.parametrize('values',[(0,8,8,16),(513,9,9,16),(8256,129,129,16),
                                 (8192,127,128,16),(8192,128,127,16),
                                 (4096,64,64,4),(4096.0,64,64,16),(True,64,64,16)])
def test_invalid(values):
    with pytest.raises(ValueError):module.validate_signature(*values)
