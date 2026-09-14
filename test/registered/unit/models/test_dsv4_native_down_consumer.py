"""No GPU/import of SGLang: native-only scope and actual selector contracts."""
import ast
from contextlib import contextmanager
from contextvars import ContextVar
import itertools
from pathlib import Path
from types import SimpleNamespace as NS

ROOT = Path(__file__).resolve().parents[4]


def source_scope():
    path=ROOT/'python/sglang/srt/distributed/device_communicators/dsv4_ar_experiment.py'
    tree=ast.parse(path.read_text())
    names=('eligible','down_uniform_eligible','dsv4_ar_scope')
    nodes=[n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name in names]
    class Settings:
        enabled=False
        def __getattr__(self,name):
            return NS(get=lambda:self.enabled if name.endswith('TP8_M32_DOWN_CONSUMER') else False)
    env=Settings()
    variables={name:ContextVar(name,default=False) for name in (
        '_active','_native_active','_attention_active','_dspark_m128_active')}
    ns=dict(contextmanager=contextmanager,envs=env,down_uniform_requested=lambda:False,
            torch=NS(version=NS(hip=True),cuda=NS(get_device_properties=lambda _:NS(gcnArchName='gfx90a'))),
            **variables)
    exec(compile(ast.Module(body=nodes,type_ignores=[]),str(path),'exec'),ns)
    return ns,env


def test_consumer_only_scope_excludes_prefill_c1_c64_and_speculation():
    ns,env=source_scope()
    for enabled,native,decode,m in itertools.product((False,True),(False,True),(False,True),(1,32,64,128)):
        env.enabled=enabled
        batch=NS(spec_algorithm=NS(is_none=lambda:native,is_dspark=lambda:not native),
                 spec_info=NS(num_tokens_per_req=4),batch_size=m,
                 forward_mode=NS(is_decode=lambda:decode,is_target_verify=lambda:not decode))
        with ns['dsv4_ar_scope'](batch,None):
            assert ns['_native_active'].get()==(enabled and native and decode and m==32)
            assert not ns['_active'].get(), 'Consumer must not independently enable legacy AR'
            assert not ns['_dspark_m128_active'].get()
        assert not ns['_native_active'].get()


def test_production_consumer_selector():
    ns,_=source_scope()
    path=ROOT/'python/sglang/srt/layers/moe/moe_runner/aiter.py'
    tree=ast.parse(path.read_text())
    node=next(n for n in ast.walk(tree) if isinstance(n,ast.Assign)
              and isinstance(n.value,ast.Call)
              and any(isinstance(t,ast.Name) and t.id=='use_native_m32_down_consumer' for t in n.targets))
    expr=compile(ast.Expression(node.value),str(path),'eval')
    for active,tp,ep,m,k,defer in itertools.product((False,True),(4,8),(1,2),(1,32,64,128),(256,512),(False,True)):
        context=dict(ns,native_m32_active=lambda:active,
                     get_tensor_model_parallel_world_size=lambda:tp,
                     get_moe_expert_parallel_world_size=lambda:ep,_is_runtime_gfx90a=lambda:True,
                     runner_input=NS(hidden_states=NS(shape=(m,4096)),topk_ids=NS(shape=(m,6)),gfx90a_defer_reduction=defer),
                     quant_info=NS(w2_weight=NS(shape=(256,4096,k//2))),
                     grouped_assignments=4,grouped_down_rows=2,use_lds_unpack=True,
                     get_int_env_var=lambda *args:832,use_runtime_m=False,use_mfma32_prefill=False)
        assert eval(expr,context)==(active and tp==8 and ep==1 and m==32 and k==256 and not defer)
