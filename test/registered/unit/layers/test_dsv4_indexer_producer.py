import ast
from pathlib import Path
from types import SimpleNamespace as NS

import pytest
import torch

from sglang.kernels.ops.attention.dsv4.gfx90a_indexer_producer import cpu_plan_inputs

ROOT=Path(__file__).resolve().parents[4]
INDEXER=ROOT/'python/sglang/srt/layers/attention/dsv4/indexer.py'


@pytest.mark.parametrize('ext,pre,rows,width,ok',[
    ([8192]*4,[0]*4,32768,2048,True),
    ([8192,8192,8191,8192],[0]*4,32767,2048,True),
    ([8192],[0],8192,2048,True),
    ([8192],[4],8192,2048,False),
    ([8191],[0],8191,2048,False),
    ([65537],[0],65537,2048,False),
    ([8192],[0],8192,2047,False),
    ([8192],[0],8192,4096,False),
    ([8192],[],8192,2048,False),
    ([8192],[-1],8192,2048,False),
    (None,None,32768,2048,False),
])
def test_cpu_admission(ext,pre,rows,width,ok):
    assert (cpu_plan_inputs(ext,pre,rows,width) is not None)==ok


def test_early_guard_mode_isolation(monkeypatch):
    tree=ast.parse(INDEXER.read_text())
    cls=next(n for n in tree.body if isinstance(n,ast.ClassDef) and n.name=='C4IndexerBackendMixin')
    methods=[n for n in cls.body if isinstance(n,ast.FunctionDef) and n.name in
             ('_use_c4_prefill_query_reuse','_use_c4_empty_prefill_tiles')]
    guard=next(n for n in ast.walk(cls) if isinstance(n,ast.If) and
               'SGLANG_DSV4_C4_PREFILL_QUERY_PRODUCER' in ast.unparse(n.test))
    def flag(v): return NS(get=lambda:v)
    envs=NS(SGLANG_OPT_USE_TILELANG_INDEXER=flag(False),SGLANG_OPT_USE_AITER_INDEXER=flag(False),
        SGLANG_FP8_PAGED_MQA_LOGITS_TORCH=flag(True),SGLANG_OPT_USE_TRITON_INDEXER_FULL=flag(True),
        SGLANG_OPT_USE_TOPK_V2=flag(False),SGLANG_DSV4_GFX90A_INDEXER_BLOCK_S=flag(16))
    parallel=NS(tp_size=8,attn_tp_size=8,moe_ep_size=1,attn_cp_size=1,pp_size=1,attn_dcp_size=1)
    import os
    context=dict(os=os,torch=NS(cuda=NS(is_current_stream_capturing=lambda:False)),
        envs=envs,is_hip=lambda:True,is_gfx90a_supported=lambda:True,
        ForwardMode=NS(EXTEND='extend'),get_parallel=lambda:parallel,
        get_global_indexer_capturer=lambda:None)
    for method in methods: exec(compile(ast.Module(body=[method],type_ignores=[]),'guard','exec'),context)
    backend=NS(token_to_kv_pool=NS(_unified_kv=True),is_draft_worker=False,is_dspark_target=False,
        mtp_enabled=False,dsa_topk_backend=NS(is_sgl_kernel=lambda:True),
        debug_use_external_c4_sparse_indices=False,hisparse_coordinator=None)
    for method in methods:
        setattr(backend,method.name,context[method.name].__get__(backend))
    batch=NS(forward_mode='extend',spec_algorithm=None,tbo_parent_token_range=None,_original_forward_mode=None)
    indexer=NS(compressor=NS(_debug_original_v4=True,ratio=4),index_topk=512)
    meta=NS(c4_page_size=64,use_prefill_cuda_graph=False)
    context.update(self=backend,forward_batch=batch,c4_indexer=indexer,indexer_metadata=meta,
        skip_logits_computation=False,use_fp4_indexer=False,enable_multi_stream=False,q_lora_ready=None,num_queries=32768)
    values={'SGLANG_DSV4_C4_PREFILL_QUERY_PRODUCER':'1','SGLANG_DSV4_C4_PREFILL_QUERY_OWNER':'1',
        'SGLANG_DSV4_C4_PREFILL_QUERY_REUSE4':'1','SGLANG_DSV4_C4_TRIVIAL_LOGITS_SKIP':'1',
        'SGLANG_DSV4_C4_PREFILL_EMPTY_TILE_SKIP':'1','SGLANG_DSV4_C4_PREFILL_QUERY_GROUP_SIZE':'16',
        'SGLANG_DSV4_C4_PREFILL_QUERY_RUNTIME_M':'1','SGLANG_DSV4_GFX90A_CANONICAL_INDEXER_ORDER':'3'}
    for key,value in values.items():monkeypatch.setenv(key,value)
    evaluate=lambda:eval(compile(ast.Expression(guard.test),'guard','eval'),context)
    assert evaluate()
    for obj,attr,value in [(batch,'forward_mode','decode'),(batch,'spec_algorithm',NS(is_none=lambda:False)),
            (batch,'tbo_parent_token_range',(0,1)),(indexer.compressor,'_debug_original_v4',False),
            (backend,'is_draft_worker',True),(backend,'is_dspark_target',True),(backend,'mtp_enabled',True),
            (parallel,'tp_size',4),(parallel,'attn_dcp_size',2),(meta,'use_prefill_cuda_graph',True)]:
        old=getattr(obj,attr);setattr(obj,attr,value);assert not evaluate();setattr(obj,attr,old)
    for key in values:
        monkeypatch.setenv(key,'0');assert not evaluate();monkeypatch.setenv(key,values[key])
    monkeypatch.delenv('SGLANG_DSV4_C4_PREFILL_QUERY_PRODUCER');assert not evaluate()


def test_no_global_blas_switch_in_production():
    for name in ['gfx90a_indexer_producer.py','gfx90a_rocblas_linear.py']:
        source=(ROOT/'python/sglang/kernels/ops/attention/dsv4'/name).read_text()
        assert 'preferred_blas_library' not in source
