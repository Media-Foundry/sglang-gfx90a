import os
from pathlib import Path
import subprocess
from types import SimpleNamespace as NS
from unittest.mock import patch

import pytest

from sglang.srt.layers.dsv4_prefill_experiments import (
    POST_REUSE_ENV, instrument_prefill_post_reuse, post_reuse_active,
    post_reuse_eligible,
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


def test_profile_default_and_explicit_override():
    root = Path(__file__).resolve().parents[4]
    source = (root / "scripts/rocm_dsv4_flash.sh").read_text()
    start = source.index('GFX90A_TP8_MULTI_REQUEST_PROFILE="${SGLANG_DSV4_GFX90A_TP8_MULTI_REQUEST_PROFILE:-0}"')
    block = source[start:source.index('\nfi\n', start)+4]
    for tp_profile, prefill, tp, ep, a2a, override, expected in (
        ('1','1','8','1','none',None,'1'),
        ('1','1','8','1','none','0','0'),
        ('1','1','8','1','none','1','1'),
        ('0','1','8','1','none',None,'unset'),
        ('1','0','8','1','none',None,'unset'),
        ('1','1','4','1','none',None,'unset'),
        ('1','1','8','2','none',None,'unset'),
        ('1','1','8','1','mori',None,'unset'),
    ):
        env = dict(PATH=os.environ['PATH'], TP_SIZE=tp, EP_SIZE=ep, MOE_A2A_BACKEND=a2a,
                   SGLANG_DSV4_GFX90A_TP8_MULTI_REQUEST_PROFILE=tp_profile,
                   GFX90A_PREFILL_THROUGHPUT_PROFILE=prefill)
        if override is not None: env[POST_REUSE_ENV] = override
        result = subprocess.run(['bash','--noprofile','--norc','-c',
            'set -eu\n'+block+'\nprintf "%s" "${'+POST_REUSE_ENV+'-unset}"'],
            env=env, text=True, capture_output=True, check=True)
        assert result.stdout == expected
