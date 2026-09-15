import os
import ast
from pathlib import Path
import subprocess
from types import SimpleNamespace as NS
from types import ModuleType
import sys
from unittest.mock import patch

import pytest

from sglang.srt.layers.dsv4_prefill_experiments import (
    POST_REUSE_ENV, instrument_prefill_post_reuse, post_reuse_active,
    post_reuse_eligible,
    MIX_REUSE_ENV, instrument_prefill_mix_reuse, mix_reuse_active,
)


def inputs():
    cfg = NS(model_type="deepseek_v4", num_hidden_layers=43, hidden_size=4096)
    ps = NS(tp_size=8, attn_tp_size=8, moe_ep_size=1, attn_cp_size=1, pp_size=1)
    runner = NS(model_config=NS(hf_text_config=cfg), ps=ps)
    batch = NS(spec_algorithm=None,
               forward_mode=NS(is_extend_without_speculative=lambda: True),
               input_ids=NS(shape=(32767,), device=NS(type="cuda")))
    return runner, batch


def test_disabled_identity():
    def fn(): pass
    with patch.dict(os.environ, {POST_REUSE_ENV: "0"}):
        assert instrument_prefill_post_reuse(fn) is fn
    assert not post_reuse_active()


def test_eligibility():
    runner, batch = inputs()
    assert post_reuse_eligible(runner, batch)
    cfg, ps = runner.model_config.hf_text_config, runner.ps
    for obj, name, value in [(cfg, "model_type", "deepseek_v41"),
                             (cfg, "num_hidden_layers", 40),
                             (ps, "tp_size", 4), (ps, "attn_tp_size", 4),
                             (ps, "moe_ep_size", 2), (ps, "attn_cp_size", 2),
                             (ps, "pp_size", 2)]:
        old = getattr(obj, name); setattr(obj, name, value)
        assert not post_reuse_eligible(runner, batch)
        setattr(obj, name, old)
    for m, expected in [(128, False), (8191, False), (8192, True),
                         (65536, True), (65537, False)]:
        batch.input_ids.shape = (m,)
        assert post_reuse_eligible(runner, batch) == expected
    batch.input_ids.shape = (32768,)
    batch.spec_algorithm = NS(is_none=lambda: False)
    assert not post_reuse_eligible(runner, batch)
    batch.spec_algorithm = None
    batch.tbo_parent_token_range = (0,32768)
    assert not post_reuse_eligible(runner, batch)
    batch.tbo_parent_token_range = None
    batch._original_forward_mode = "rewritten"
    assert not post_reuse_eligible(runner, batch)
    batch._original_forward_mode = None
    batch.forward_mode.is_extend_without_speculative = lambda: False
    assert not post_reuse_eligible(runner, batch)


def test_context_resets_after_exception_and_graph_is_excluded():
    runner, batch = inputs()
    seen = []
    def fn(self, batch):
        seen.append(post_reuse_active())
        raise RuntimeError("test")
    with patch.dict(os.environ, {POST_REUSE_ENV: "1"}), \
         patch("torch.version.hip", "test"), \
         patch("torch.cuda.get_device_properties", return_value=NS(gcnArchName="gfx90a")), \
         patch("torch.cuda.is_current_stream_capturing", return_value=False) as capturing:
        wrapped = instrument_prefill_post_reuse(fn)
        with pytest.raises(RuntimeError): wrapped(NS(model_runner=runner), batch)
        assert not post_reuse_active()
        capturing.return_value = True
        with pytest.raises(RuntimeError): wrapped(NS(model_runner=runner), batch)
        assert not post_reuse_active()
    assert seen == [True, False]


def test_mix_scope_independent_and_keyword_forward_batch():
    runner, batch = inputs()
    seen = []
    def fn(self, batch):
        seen.append((post_reuse_active(), mix_reuse_active()))
        raise RuntimeError("test")
    with patch.dict(os.environ, {POST_REUSE_ENV: "1", MIX_REUSE_ENV: "1"}), \
         patch("torch.version.hip", "test"), \
         patch("torch.cuda.get_device_properties", return_value=NS(gcnArchName="gfx90a")), \
         patch("torch.cuda.is_current_stream_capturing", return_value=False) as capture:
        wrapped = instrument_prefill_post_reuse(instrument_prefill_mix_reuse(fn))
        with pytest.raises(RuntimeError): wrapped(NS(model_runner=runner), forward_batch=batch)
        assert not post_reuse_active() and not mix_reuse_active()
        capture.return_value = True
        with pytest.raises(RuntimeError): wrapped(NS(model_runner=runner), forward_batch=batch)
        assert not post_reuse_active() and not mix_reuse_active()
    assert seen == [(True,True), (False,False)]
    with patch.dict(os.environ, {MIX_REUSE_ENV:"0"}):
        assert instrument_prefill_mix_reuse(fn) is fn


def test_mix_dispatch_preserves_nondefault_k_contract():
    root = Path(__file__).resolve().parents[4]
    tree = ast.parse((root/'python/sglang/kernels/ops/layernorm/mhc.py').read_text())
    node = next(n for n in tree.body if isinstance(n,ast.FunctionDef)
                and n.name=='gfx90a_mhc_pre_mix_from_partials_triton')
    calls=[]
    class Launch:
        def __getitem__(self, grid):
            return lambda *a,**kw:calls.append(kw['BLOCK_K'])
    kernel_module=ModuleType('sglang.kernels.ops.layernorm.gfx90a_mhc_premix_reuse')
    candidate=object();fallback=object()
    groups=[]
    kernel_module.premix_reuse4=lambda *a,**kw:(groups.append(kw['group_size']) or candidate)
    block_k=1024;enabled=True
    ns=dict(os=os,torch=NS(Tensor=object,version=NS(hip=True),bfloat16='bf16',float32='fp32',
                     empty=lambda *a,**k:fallback),
            envs=NS(SGLANG_DSV4_GFX90A_MHC_BLOCK_K=NS(get=lambda:block_k)),
            _prefill_mix_reuse_active=lambda:enabled,_prefill_mix_reuse_logged=True,
            _prefill_mix_pair_active=None,_prefill_mix_pair_logged=True,
            _gfx90a_mhc_mix_partials_kernel=Launch())
    exec(compile(ast.Module(body=[node],type_ignores=[]),'<current mix dispatch>','exec'),ns)
    x=NS(ndim=3,shape=(8192,4,4096),dtype='bf16',device='cuda',
         is_contiguous=lambda:True,flatten=lambda *a:None)
    fn=NS(shape=(24,16384),dtype='fp32',is_contiguous=lambda:True)
    partials=NS(shape=(8192,64),dtype='fp32',is_contiguous=lambda:True)
    call=ns['gfx90a_mhc_pre_mix_from_partials_triton']
    with patch.dict(sys.modules,{kernel_module.__name__:kernel_module}), \
         patch.dict(os.environ,{'SGLANG_DSV4_PREFILL_MIX_GROUP_SIZE':'4'}):
        assert call(x,fn,partials,1e-6) is candidate
        assert groups==[4]
        with patch.dict(os.environ,{'SGLANG_DSV4_PREFILL_MIX_GROUP_SIZE':'8'}):
            assert call(x,fn,partials,1e-6) is candidate
            assert groups==[4,8]
        block_k=2048
        assert call(x,fn,partials,1e-6) is fallback
        assert calls==[2048]
        block_k=1024;enabled=False
        assert call(x,fn,partials,1e-6) is fallback
        assert calls==[2048,1024]


def test_premix8_wrapper_scope_and_fallback():
    root=Path(__file__).resolve().parents[4]
    path=root/'python/sglang/kernels/ops/layernorm/gfx90a_mhc_premix_reuse.py'
    node=next(n for n in ast.parse(path.read_text()).body if isinstance(n,ast.FunctionDef)
              and n.name=='premix_reuse4')
    calls=[]
    class Launch:
        def __init__(self,group):self.group=group
        def __getitem__(self,grid):
            return lambda *a,**kw:calls.append((self.group,grid,kw))
    device=NS(type='cuda')
    module=ModuleType('sglang.kernels.ops.layernorm.gfx90a_mhc_premix_reuse8')
    module.premix8=Launch(8)
    out=object()
    ns=dict(torch=NS(bfloat16='bf16',float32='fp32',version=NS(hip=True),
        cuda=NS(get_device_properties=lambda _:NS(gcnArchName='gfx90a')),
        empty=lambda *a,**kw:out),triton=NS(cdiv=lambda a,b:(a+b-1)//b),
        _premix_reuse_kernel=Launch(4))
    exec(compile(ast.Module(body=[node],type_ignores=[]),'<current premix wrapper>','exec'),ns)
    def tensor(shape,dtype):return NS(shape=shape,dtype=dtype,device=device,is_contiguous=lambda:True)
    with patch.dict(sys.modules,{module.__name__:module}):
        for m in (128,8191,8192,32765,65536,65537):
            for requested in (4,8):
                x=tensor((m,4,4096),'bf16');fn=tensor((24,16384),'fp32');rms=tensor((m,64),'fp32')
                assert ns['premix_reuse4'](x,fn,rms,1e-6,group_size=requested) is out
                expected=8 if requested==8 and 8192<=m<=65536 else 4
                assert calls[-1]==(expected,(24,(m+expected-1)//expected),{'num_warps':1})
        x=tensor((8192,4,4096),'fp32')
        assert ns['premix_reuse4'](x,fn,tensor((8192,64),'fp32'),1e-6,group_size=8) is None


@pytest.mark.parametrize("flag", [POST_REUSE_ENV, MIX_REUSE_ENV,
                                  'SGLANG_DSV4_PREFILL_MIX_GROUP_SIZE',
                                  'SGLANG_DSV4_PREFILL_MIX_PAIR_COLUMNS'])
def test_profile_default_and_explicit_override(flag):
    root = Path(__file__).resolve().parents[4]
    source = (root / "scripts/rocm_dsv4_flash.sh").read_text()
    start = source.index('GFX90A_TP8_MULTI_REQUEST_PROFILE="${SGLANG_DSV4_GFX90A_TP8_MULTI_REQUEST_PROFILE:-0}"')
    block = source[start:source.index('\nfi\n', start)+4]
    group_flag=flag=='SGLANG_DSV4_PREFILL_MIX_GROUP_SIZE'
    default,off,on=('8','4','8') if group_flag else ('1','0','1')
    for tp_profile, prefill, tp, ep, a2a, override, expected in (
        ('1','1','8','1','none',None,default),
        ('1','1','8','1','none',off,off),
        ('1','1','8','1','none',on,on),
        ('0','1','8','1','none',None,'unset'),
        ('1','0','8','1','none',None,'unset'),
        ('1','1','4','1','none',None,'unset'),
        ('1','1','8','2','none',None,'unset'),
        ('1','1','8','1','mori',None,'unset'),
    ):
        env = dict(PATH=os.environ['PATH'], TP_SIZE=tp, EP_SIZE=ep, MOE_A2A_BACKEND=a2a,
                   SGLANG_DSV4_GFX90A_TP8_MULTI_REQUEST_PROFILE=tp_profile,
                   GFX90A_PREFILL_THROUGHPUT_PROFILE=prefill)
        if override is not None: env[flag] = override
        result = subprocess.run(['bash','--noprofile','--norc','-c',
            'set -eu\n'+block+'\nprintf "%s" "${'+flag+'-unset}"'],
            env=env, text=True, capture_output=True, check=True)
        assert result.stdout == expected
