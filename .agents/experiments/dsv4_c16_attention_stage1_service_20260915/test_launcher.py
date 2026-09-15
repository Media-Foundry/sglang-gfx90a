"""Evaluate only the launcher's side-effect-free profile prefix."""
import os
from pathlib import Path
import subprocess

import pytest

repo=Path(__file__).resolve().parents[3]
prefix=(repo/'scripts/rocm_dsv4_flash.sh').read_text().split('# One TP4 replica dedicated')[0]


def resolved(**overrides):
    env={k:v for k,v in os.environ.items() if not k.startswith(('SGLANG_','AITER_'))}
    for key in ('TP_SIZE','EP_SIZE','MOE_A2A_BACKEND'):
        env.pop(key,None)
    env.update(overrides)
    command=prefix+'\nprintf "%s" "${SGLANG_DSV4_PREFILL_ATTN_STAGE1:-unset}"\n'
    return subprocess.check_output(['bash','-s','--','serve'],input=command.encode(),env=env).decode()


def matching(**overrides):
    values=dict(SGLANG_DSV4_GFX90A_TP8_MULTI_REQUEST_PROFILE='1',
                SGLANG_DSV4_GFX90A_PREFILL_THROUGHPUT_PROFILE='1',TP_SIZE='8')
    values.update(overrides)
    return resolved(**values)


def test_matching_defaults():
    assert matching()=='1'
    assert matching(SGLANG_DSV4_PREFILL_ATTN_STAGE1='0')=='0'


@pytest.mark.parametrize('settings',[
    {}, {'SGLANG_DSV4_GFX90A_TP8_MULTI_REQUEST_PROFILE':'1'},
    {'SGLANG_DSV4_GFX90A_PREFILL_THROUGHPUT_PROFILE':'1'},
])
def test_other_profiles(settings):
    assert resolved(**settings)=='unset'


@pytest.mark.parametrize('overrides',[{'TP_SIZE':'4'},{'EP_SIZE':'2'},{'MOE_A2A_BACKEND':'mori'}])
def test_other_parallelism(overrides):
    assert matching(**overrides)=='unset'
