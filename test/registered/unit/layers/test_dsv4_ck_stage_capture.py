from contextlib import nullcontext
import importlib.util
from pathlib import Path

import torch

PATH=Path(__file__).resolve().parents[4]/'python/sglang/kernels/ops/debug/dsv4_ck_stage_capture.py'
spec=importlib.util.spec_from_file_location('ck_capture',PATH)
m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)


def test_disabled_scope_does_not_inspect_batch(monkeypatch):
    monkeypatch.delenv('SGLANG_DSV4_DEBUG_CK_STAGE_CAPTURE_DIR',raising=False)
    with m.scope(21, object(), nullcontext()):
        assert m.current() is None


def test_capture_preserves_bf16_bits_and_rejects_overwrite(tmp_path):
    import pytest
    cap=m.Capture(tmp_path/'fixture')
    x=torch.tensor([[1.0,-2.5]],dtype=torch.bfloat16)
    cap.tensor('x',x)
    loaded=torch.load(tmp_path/'fixture/x.pt',weights_only=True)
    assert torch.equal(x.view(torch.uint8),loaded.view(torch.uint8))
    with pytest.raises(AssertionError):cap.tensor('x',x)


def test_cached_stage2_resolves_current_scope_at_invocation():
    import ast
    path=PATH.parents[1]/'moe/gfx90a_bf16_batched_moe.py'
    tree=ast.parse(path.read_text())
    fn=next(n for n in ast.walk(tree) if isinstance(n,ast.FunctionDef) and n.name=='stage2_dsv4_fp32')
    first=fn.body[0]
    assert isinstance(first,ast.Assign) and first.targets[0].id=='probe'
    assert any(isinstance(n,ast.Call) and isinstance(n.func,ast.Name)
               and n.func.id=='_capture_current' for n in ast.walk(first))
