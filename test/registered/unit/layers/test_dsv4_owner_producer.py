import ast
import importlib.util
from pathlib import Path
from types import SimpleNamespace

ROOT=Path(__file__).resolve().parents[4]
PATH=ROOT/'python/sglang/kernels/ops/debug/dsv4_owner_producer.py'
spec=importlib.util.spec_from_file_location('producer_oracle',PATH)
module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)


def test_disabled_and_other_layer_do_not_touch_gpu(monkeypatch):
    values=dict(indexer=SimpleNamespace(layer_id=20),batch=None,x=None,q_lora=None,
        positions=None,q=None,weights=None,cache=None,lengths=None,pages=None,width=2048,
        preshuffle_tile=16,dot_fp16=False,fp8_fnuz=False)
    monkeypatch.delenv('SGLANG_DSV4_DEBUG_OWNER_PRODUCER_DIR',raising=False)
    module.diagnose(**values)
    monkeypatch.setenv('SGLANG_DSV4_DEBUG_OWNER_PRODUCER_DIR','unused')
    values['indexer'].layer_id=21;module.diagnose(**values)
    assert not module._done


def test_oracle_hook_inside_existing_owner_scope():
    tree=ast.parse((ROOT/'python/sglang/srt/layers/attention/dsv4/indexer.py').read_text())
    guard=next(n for n in ast.walk(tree) if isinstance(n,ast.If)
        and 'SGLANG_DSV4_C4_PREFILL_QUERY_OWNER' in ast.unparse(n.test))
    assert any(isinstance(n,ast.Call) and isinstance(n.func,ast.Name)
               and n.func.id=='_owner_producer_oracle' for n in ast.walk(guard))
