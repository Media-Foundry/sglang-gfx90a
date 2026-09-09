import json
import os
from pathlib import Path
import subprocess
import sys


def invoke(tmp_path, command='serve-dspark', **overrides):
    root = Path(__file__).resolve().parents[4]
    stub = tmp_path/'python-stub'
    stub.write_text(f'#!{sys.executable}\nimport json,os,sys\nprint(json.dumps(dict(argv=sys.argv[1:], env=dict(os.environ))))\n')
    stub.chmod(0o700)
    env = dict(PATH=os.environ['PATH'], PYTHON_BIN=str(stub),
               SGLANG_DIR=str(root), NUMA_INTERLEAVE_ALL='0',
               SGLANG_DSV4_GFX90A_DSPARK_TP8_FULL_TARGET_PROFILE='1')
    env.update(overrides)
    return subprocess.run(['bash',str(root/'scripts/rocm_dsv4_flash.sh'),command],
                          env=env,text=True,capture_output=True,timeout=30)


def test_profile_resolves_full_target_defaults_without_starting_gpu(tmp_path):
    result=invoke(tmp_path,SGLANG_DSV4_GFX90A_PREFILL_THROUGHPUT_PROFILE='1')
    assert result.returncode==0, result.stderr
    payload=json.loads(result.stdout.splitlines()[-1]); argv=payload['argv']; env=payload['env']
    for key,value in [('--tp-size','8'),('--ep-size','1'),('--moe-a2a-backend','none'),
                      ('--speculative-algorithm','DSPARK'),('--speculative-dspark-block-size','3'),
                      ('--max-total-tokens','1048576'),('--chunked-prefill-size','36864')]:
        assert argv[argv.index(key)+1]==value
    assert env['SGLANG_DSV4_GFX90A_DSPARK_M128_ANCHOR_ONLY_ROUTED']=='0'
    assert env['SGLANG_DSV4_GFX90A_DSPARK_M128_PRE_ROUTER_COMPACT']=='0'
    assert env['SGLANG_DSV4_GFX90A_DSPARK_TP8_M128_AR_BLOCKS']=='12'
    assert env['SGLANG_DSV4_GFX90A_DSPARK_TP8_M128_CK_SPARSE_DECODE']=='1'
    assert env['SGLANG_DSV4_GFX90A_DSPARK_TP8_M128_CK_C4']=='0'
    assert env['SGLANG_DSV4_DSA_DENSE_ONLY_GRAPH']=='0'
    assert env['HIP_VISIBLE_DEVICES']=='0,1,2,3,4,5,6,7'
    graph_values=[]
    for value in argv[argv.index('--cuda-graph-bs-decode')+1:]:
        if value.startswith('--'): break
        graph_values.append(value)
    assert graph_values==['1','2','4','8','16','24','32']


def test_profile_rejects_native_and_incompatible_settings(tmp_path):
    assert invoke(tmp_path,'serve').returncode==2
    for settings in [dict(TP_SIZE='4'),dict(EP_SIZE='2'),dict(MOE_A2A_BACKEND='mori'),
                     dict(SGLANG_DSV4_GFX90A_DSPARK_TP8_BS32_PROFILE='1'),
                     dict(SGLANG_DSV4_GFX90A_DSPARK_M128_ANCHOR_ONLY_ROUTED='1')]:
        result=invoke(tmp_path,**settings)
        assert result.returncode==2, result.stdout
