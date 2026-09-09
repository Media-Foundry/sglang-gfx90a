import ast
from contextlib import contextmanager
from contextvars import ContextVar
from pathlib import Path
from types import SimpleNamespace


def test_dspark_ar_narrow_graph_guard():
    root = Path(__file__).resolve().parents[4]
    path = root/'python/sglang/srt/distributed/device_communicators/dsv4_dspark_ar.py'
    tree = ast.parse(path.read_text())
    fn = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'eligible')
    scope = {}
    exec(compile(ast.Module(body=[fn], type_ignores=[]), str(path), 'exec'), scope)
    good = dict(active=True, world=8, shape=(128,4096), bf16=True,
                contiguous=True, quantized=False, registered=True, capturing=True)
    assert scope['eligible'](**good)
    for key, value in [('active',False), ('world',4), ('shape',(32,4096)),
                       ('shape',(64,4096)), ('bf16',False), ('contiguous',False),
                       ('quantized',True), ('registered',False), ('capturing',False)]:
        assert not scope['eligible'](**(good | {key:value})), (key,value)


def test_target_scope_excludes_native_and_resets_on_exception():
    root = Path(__file__).resolve().parents[4]
    path = root/'python/sglang/srt/distributed/device_communicators/dsv4_ar_experiment.py'
    tree = ast.parse(path.read_text())
    fn = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'dsv4_ar_scope')
    class Settings:
        def __getattr__(self, name):
            return SimpleNamespace(get=lambda: 12 if name.endswith('DSPARK_TP8_M128_AR_BLOCKS') else False)
    variables = {name: ContextVar(name, default=False) for name in (
        '_active','_native_active','_attention_active','_dspark_m128_active')}
    scope = dict(contextmanager=contextmanager, envs=Settings(),
                 down_uniform_requested=lambda: False, eligible=lambda **kw: False,
                 torch=SimpleNamespace(version=SimpleNamespace(hip=False)), **variables)
    exec(compile(ast.Module(body=[fn], type_ignores=[]), str(path), 'exec'), scope)
    for native, verify, bs, width, expected in [
        (False,True,32,4,True), (True,False,32,4,False),
        (False,False,32,4,False), (False,True,16,4,False),
        (False,True,32,3,False)]:
        batch = SimpleNamespace(
            spec_algorithm=SimpleNamespace(is_none=lambda: native, is_dspark=lambda: not native),
            forward_mode=SimpleNamespace(is_decode=lambda: not verify, is_target_verify=lambda: verify),
            batch_size=bs, spec_info=SimpleNamespace(num_tokens_per_req=width))
        try:
            with scope['dsv4_ar_scope'](batch,None):
                assert variables['_dspark_m128_active'].get() is expected
                raise ValueError('exercise finally')
        except ValueError:
            pass
        assert not variables['_dspark_m128_active'].get()
