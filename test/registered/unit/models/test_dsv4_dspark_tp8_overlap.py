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
