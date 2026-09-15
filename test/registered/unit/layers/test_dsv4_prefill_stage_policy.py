"""CPU-only scope gates; no HIP context or full SGLang import."""
import importlib.util
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[4]
PATH = REPO / "python/sglang/kernels/ops/attention/dsv4/unified_kv_kernels/prefill_stage_policy.py"
spec = importlib.util.spec_from_file_location("prefill_stage_policy_tested", PATH)
policy = importlib.util.module_from_spec(spec)
spec.loader.exec_module(policy)


def good():
    return dict(enabled=True, gfx90a=True, model_type="deepseek_v4",
                native_extend=True, speculative=False, draft=False,
                tp=8, cp=1, dcp=1, pp=1, rows=32767, heads=8, dim=512,
                bf16=True, capturing=False)


@pytest.mark.parametrize("rows", [8192, 32766, 32767, 32768, 65536])
def test_admitted(rows):
    args=good();args["rows"]=rows
    assert policy.use_stage1(**args)


@pytest.mark.parametrize("key,value", [
    ("enabled",False),("gfx90a",False),("model_type","deepseek_v41"),
    ("model_type",None),("native_extend",False),("speculative",True),
    ("draft",True),("tp",4),("cp",2),("dcp",2),("pp",2),
    ("rows",0),("rows",1),("rows",128),("rows",8191),("rows",65537),
    ("heads",16),("dim",128),("bf16",False),("capturing",True),
])
def test_excluded(key,value):
    args=good();args[key]=value
    assert not policy.use_stage1(**args)
