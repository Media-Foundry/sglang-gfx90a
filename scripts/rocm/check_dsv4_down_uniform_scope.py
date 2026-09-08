#!/usr/bin/env python3
"""CPU tests of actual scope/shape predicates and down-only integration."""
import ast
from contextlib import contextmanager
from contextvars import ContextVar
from pathlib import Path
from types import SimpleNamespace as NS

root = Path(__file__).resolve().parents[2]
path = root/'python/sglang/srt/distributed/device_communicators/dsv4_ar_experiment.py'
tree = ast.parse(path.read_text())
functions = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name in
             ('eligible', 'down_uniform_eligible', 'dsv4_ar_scope')]
flags = {'SGLANG_DSV4_GFX90A_TP8_M32_DOWN_UNIFORM': True}
class Envs:
    def __getattr__(self, name):
        return NS(get=lambda: flags.get(name, False))
ns = dict(contextmanager=contextmanager, envs=Envs(),
          torch=NS(version=NS(hip=True), cuda=NS(get_device_properties=lambda d: NS(gcnArchName='gfx90a'))),
          _active=ContextVar('ar', default=False),
          _native_active=ContextVar('native', default=False),
          _attention_active=ContextVar('attention', default=False))
exec(compile(ast.Module(body=functions, type_ignores=[]), str(path), 'exec'), ns)
for m in (1,2,4,8,16,24,32,64):
    for decode in (False, True):
        for native in (False, True):
            batch=NS(batch_size=m, forward_mode=NS(is_decode=lambda:decode),
                     spec_algorithm=NS(is_none=lambda:native))
            with ns['dsv4_ar_scope'](batch, 0):
                assert ns['_native_active'].get() == (m==32 and decode and native)
                assert not ns['_active'].get(), 'must not enable legacy AR'
                assert not ns['_attention_active'].get()
            assert not ns['_native_active'].get()
base=dict(native_scope=True,tp_size=8,ep_size=1,gfx90a=True,
          hidden_shape=(32,4096),topk_shape=(32,6),weight_shape=(256,4096,128),
          geometry=(4,2,8,832,True),incompatible=False)
f=ns['down_uniform_eligible']
assert f(**base)
for key, values in dict(native_scope=[False],tp_size=[4,1],ep_size=[2,4],gfx90a=[False],
    hidden_shape=[(1,4096),(64,4096)],topk_shape=[(32,8)],
    weight_shape=[(128,4096,256)],geometry=[(2,2,8,832,True),(4,2,4,832,True),
    (4,2,8,624,True),(4,2,8,832,False)],incompatible=[True]).items():
    for value in values: assert not f(**(base|{key:value})), (key,value)
runner=ast.parse((root/'python/sglang/srt/layers/moe/moe_runner/aiter.py').read_text())
calls=[n for n in ast.walk(runner) if isinstance(n,ast.Call)
       and any(k.arg=='uniform_metadata' for k in n.keywords)]
assert len(calls)==1 and ast.unparse(calls[0].func)=='gfx90a_fp4_expert_down_grouped'
env=(root/'python/sglang/srt/environ.py').read_text()
assert 'SGLANG_DSV4_GFX90A_TP8_M32_DOWN_UNIFORM = EnvBool(False)' in env
print('PASS: scope tiers/native/decode, independent flag, TP/EP/geometry exclusion, down-only/default-off')
