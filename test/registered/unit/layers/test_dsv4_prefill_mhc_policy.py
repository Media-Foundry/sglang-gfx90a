import ast
import os
from pathlib import Path
from types import SimpleNamespace as NS
from unittest.mock import patch

import pytest

from sglang.srt.layers import dsv4_prefill_mhc_policy as policy


def inputs(m=32767):
    cfg = NS(model_type="deepseek_v4", num_hidden_layers=43, hidden_size=4096,
             hc_sinkhorn_iters=20)
    ps = NS(tp_size=8, attn_tp_size=8, moe_ep_size=1, attn_cp_size=1,
            pp_size=1, attn_dcp_size=1)
    runner = NS(model_config=NS(hf_text_config=cfg), ps=ps, is_draft_worker=False)
    batch = NS(spec_algorithm=None, forward_mode=NS(name="EXTEND"),
               input_ids=NS(shape=(m,), device=NS(type="cuda")))
    return runner, batch


def test_disabled_identity_and_legacy_defaults():
    def fn(): pass
    with patch.dict(os.environ, {policy.ENV: "0"}):
        assert policy.instrument_prefill_mhc_iters(fn) is fn
    assert policy.resolve_sinkhorn_iters(8) == 8
    assert policy.resolve_sinkhorn_iters(20) == 20


@pytest.mark.parametrize("m", [1, 2, 128, 8191, 8192, 32767, 32768, 65537])
def test_all_nonempty_prefill_rows_including_tails(m):
    assert policy.eligible(*inputs(m))


@pytest.mark.parametrize("mode", ["DECODE", "MIXED", "TARGET_VERIFY",
    "DRAFT_EXTEND_V2", "IDLE", "SPLIT_PREFILL", "DLLM_EXTEND", "PREBUILT"])
def test_modes_excluded(mode):
    runner, batch = inputs()
    batch.forward_mode.name = mode
    assert not policy.eligible(runner, batch)


def test_scope_exclusions():
    runner, batch = inputs()
    cfg, ps = runner.model_config.hf_text_config, runner.ps
    for obj, key, value in (
        (cfg,"model_type","deepseek_v41"), (cfg,"num_hidden_layers",40),
        (cfg,"hidden_size",8192), (cfg,"hc_sinkhorn_iters",8),
        (ps,"tp_size",4), (ps,"attn_tp_size",4), (ps,"moe_ep_size",2),
        (ps,"attn_cp_size",2), (ps,"attn_dcp_size",2), (ps,"pp_size",2),
        (runner,"is_draft_worker",True),
    ):
        with patch.object(obj,key,value):
            assert not policy.eligible(runner,batch)
    assert not policy.eligible(*inputs(0))
    batch.spec_algorithm=NS(is_none=lambda:False)
    assert not policy.eligible(runner,batch)
    batch.spec_algorithm=NS(is_none=lambda:True)
    assert policy.eligible(runner,batch)
    for key in ("tbo_parent_token_range","_original_forward_mode"):
        setattr(batch,key,"rewritten")
        assert not policy.eligible(runner,batch)
        setattr(batch,key,None)


def test_context_reset_nested_decode_and_hardware_guards():
    runner, batch = inputs(1)  # Prefill tail, not a decode admission.
    seen=[]
    def fn(self, forward_batch):
        seen.append(policy.resolve_sinkhorn_iters(8))
        raise RuntimeError("fixture")
    with patch.dict(os.environ,{policy.ENV:"1"}), \
         patch("torch.version.hip","test"), \
         patch("torch.cuda.get_device_properties",return_value=NS(gcnArchName="gfx90a:sramecc+")) as props, \
         patch("torch.cuda.is_current_stream_capturing",return_value=False) as capture:
        wrapped=policy.instrument_prefill_mhc_iters(fn)
        with pytest.raises(RuntimeError):wrapped(NS(model_runner=runner),forward_batch=batch)
        assert policy.resolve_sinkhorn_iters(8)==8
        capture.return_value=True
        with pytest.raises(RuntimeError):wrapped(NS(model_runner=runner),batch)
        capture.return_value=False
        props.return_value=NS(gcnArchName="gfx942")
        with pytest.raises(RuntimeError):wrapped(NS(model_runner=runner),batch)
        props.return_value=NS(gcnArchName="gfx90a")
        token=policy._config_iters.set(20)
        try:
            batch.forward_mode.name="DECODE"
            with pytest.raises(RuntimeError):wrapped(NS(model_runner=runner),batch)
            assert policy.resolve_sinkhorn_iters(8)==20
        finally:
            policy._config_iters.reset(token)
    assert seen==[20,8,8,8]
    assert policy.resolve_sinkhorn_iters(8)==8


def test_splitk_launch_uses_scoped_iters_and_preload_helper_matches():
    root=Path(__file__).resolve().parents[4]
    tree=ast.parse((root/'python/sglang/kernels/ops/layernorm/mhc.py').read_text())
    fn=next(n for n in tree.body if isinstance(n,ast.FunctionDef)
            and n.name=='gfx90a_mhc_splitk_fused_tail_triton')
    value=next(k.value for n in ast.walk(fn) if isinstance(n,ast.Call)
               for k in n.keywords if k.arg=='SINKHORN_ITERS')
    other=ast.parse((root/'python/sglang/kernels/ops/layernorm/gfx90a_mhc_post_pre.py').read_text())
    get_iters=next(n for n in other.body if isinstance(n,ast.FunctionDef) and n.name=='_get_sinkhorn_iters')
    for resolver, active, expected in ((None,False,8),(policy.resolve_sinkhorn_iters,False,8),
                                      (policy.resolve_sinkhorn_iters,True,20)):
        ns=dict(envs=NS(SGLANG_DSV4_GFX90A_MHC_SINKHORN_ITERS=NS(get=lambda:8)),
                _prefill_sinkhorn_iters=resolver)
        token=policy._config_iters.set(20 if active else None)
        try:
            assert eval(compile(ast.Expression(value),'splitk-iters','eval'),ns)==expected
            exec(compile(ast.Module(body=[get_iters],type_ignores=[]),'native-iters','exec'),ns)
            assert ns['_get_sinkhorn_iters']()==expected
        finally:
            policy._config_iters.reset(token)


def test_only_extend_entry_is_instrumented():
    root=Path(__file__).resolve().parents[4]
    tree=ast.parse((root/'python/sglang/srt/model_executor/runner/eager_runner.py').read_text())
    names=[n.name for n in ast.walk(tree) if isinstance(n,ast.FunctionDef)
           and any(isinstance(d,ast.Name) and d.id=='instrument_prefill_mhc_iters'
                   for d in n.decorator_list)]
    assert names==['_execute_extend']
