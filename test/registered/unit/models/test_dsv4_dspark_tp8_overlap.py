import importlib.util
from pathlib import Path


def test_tp8_target_overlap_guards():
    root = Path(__file__).resolve().parents[4]
    spec = importlib.util.spec_from_file_location(
        'guard', root / 'python/sglang/srt/layers/attention/dsv4_dspark_tp8.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    args = dict(enabled=True, hip=True, tp_size=8, compress_ratio=4,
                rows=128, batch_size=32, target_verify=True, width=4,
                unified_kv=True)
    assert module.m128_overlap_eligible(**args)
    for name, value in [('enabled',False), ('hip',False), ('tp_size',4),
                        ('compress_ratio',128), ('rows',132), ('batch_size',33),
                        ('target_verify',False), ('width',6), ('unified_kv',False)]:
        assert not module.m128_overlap_eligible(**(args | {name:value})), name


def test_tp8_ck_target_only_guard():
    root = Path(__file__).resolve().parents[4]
    spec = importlib.util.spec_from_file_location(
        'guard', root / 'python/sglang/srt/layers/attention/dsv4_dspark_tp8.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    args = dict(enabled=True, gfx90a=True, tp_size=8, compress_ratio=128,
                rows=128, batch_size=32, target_verify=True, width=4,
                inverse_rope=False)
    assert module.m128_ck_eligible(**args)
    for name, value in [('enabled',False), ('gfx90a',False), ('tp_size',4),
                        ('compress_ratio',4), ('rows',132), ('batch_size',33),
                        ('target_verify',False), ('width',6), ('inverse_rope',True)]:
        assert not module.m128_ck_eligible(**(args | {name:value})), name


def test_h8_wrapper_accepts_padded_local_sink_without_copy(monkeypatch):
    # Execute the actual wrapper with a mocked FFI and CPU tensors: this
    # checks shape/ABI handling, not GPU numerics (covered by the oracle).
    import ast
    from types import SimpleNamespace
    import torch

    root = Path(__file__).resolve().parents[4]
    path = root / 'python/sglang/kernels/ops/attention/dsv4/gfx90a_sparse_h8.py'
    tree = ast.parse(path.read_text())
    function = next(n for n in tree.body if isinstance(n, ast.FunctionDef)
                    and n.name == 'run_if_supported')
    calls = []
    scope = {'torch': torch, '_module': lambda: SimpleNamespace(run=lambda *a: calls.append(a))}
    exec(compile(ast.Module(body=[function], type_ignores=[]), str(path), 'exec'), scope)
    monkeypatch.setattr(torch.Tensor, 'is_cuda', property(lambda self: True))
    q = torch.empty((128, 8, 512), dtype=torch.bfloat16)
    kv = torch.empty((16, 512), dtype=torch.bfloat16)
    indices = torch.zeros(128, dtype=torch.int32)
    ptr = torch.arange(129, dtype=torch.int32)
    sink = torch.arange(64, dtype=torch.float32)
    out = scope['run_if_supported'](q, kv, indices, ptr, sink, .1)
    assert out.shape == q.shape and len(calls) == 1
    used_sink = calls[0][4]
    assert used_sink.shape == (8,) and used_sink.data_ptr() == sink.data_ptr()
    assert torch.equal(used_sink, sink[:8])
    assert scope['run_if_supported'](q[:1], kv, indices, ptr, sink, .1) is None
