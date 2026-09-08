#!/usr/bin/env python3
"""Scoped independent-service ABBA arms with explicit TP8 NUMA policy.

Run only after the current service has no in-flight benchmark. Leaves the last
service alive. A failed readiness observation never kills/restarts that process.
"""
import argparse
import collections
import json
import subprocess
import sys
import time
from pathlib import Path

import psutil


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--parent', type=int, required=True)
    p.add_argument('--arms', nargs='+', default=['B3:1', 'B4:1', 'A4:0'])
    p.add_argument('--switch', default='SGLANG_DSV4_GFX90A_TP8_M32_ATTN_MULTISTREAM',
                   choices=['SGLANG_DSV4_GFX90A_TP8_M32_ATTN_MULTISTREAM',
                            'SGLANG_DSV4_GFX90A_FUSED_ATTN_PREP_GEMV',
                            'SGLANG_DSV4_GFX90A_TP8_M32_LEGACY_AR',
                            'SGLANG_DSV4_GFX90A_TP8_M32_GATE_PREFETCH'])
    p.add_argument('--prefix', type=Path, required=True)
    p.add_argument('--reference', type=Path, required=True)
    p.add_argument('--inputs', type=Path, required=True)
    p.add_argument('--france-c32', action='store_true',
                   help='Run repeated France sentinel after performance, correctness only')
    p.add_argument('--trace-layer', type=int, choices=range(43),
                   help='Diagnostic only: enables existing graph-only realtime markers')
    p.add_argument('--c32-rounds', type=int, default=6)
    a = p.parse_args()
    parent = psutil.Process(a.parent)
    assert 'sglang.launch_server' in parent.cmdline()
    cmd, env, cwd = parent.cmdline(), parent.environ(), parent.cwd()
    assert not any(arg.startswith('--speculative-') for arg in cmd)
    assert env.get('TP_SIZE') == '8' and env.get('EP_SIZE') == '1'
    assert env.get('MOE_A2A_BACKEND') == 'none'
    report = {'status':'running', 'arms':[], 'parent':parent.pid, 'switch':a.switch}
    def save():
        Path(str(a.prefix)+'_state.json').write_text(json.dumps(report,indent=2)+'\n')
    def bench(argv, logfile):
        with open(logfile,'w') as f:
            subprocess.run([sys.executable,*argv],stdout=f,stderr=subprocess.STDOUT,check=True)
    reference = json.loads(a.reference.read_text())
    for spec in a.arms:
        name, flag = spec.split(':')
        assert flag in ('0','1') and name.isalnum()
        assert parent.is_running() and 'sglang.launch_server' in parent.cmdline()
        children = parent.children(recursive=True)
        parent.terminate()
        _, alive = psutil.wait_procs([parent,*children],timeout=15)
        assert not [p for p in alive if p.status()!=psutil.STATUS_ZOMBIE], 'inspect remaining processes'
        env[a.switch] = flag
        for k in ('SGLANG_DSV4_GFX90A_REALTIME_TRACE_LAYER',
                  'SGLANG_DSV4_GFX90A_REALTIME_TRACE_LOG_EVERY',
                  'SGLANG_DSV4_GFX90A_REALTIME_TRACE_GRAPH_ONLY'):
            env.pop(k,None)
        if a.trace_layer is not None:
            env['SGLANG_DSV4_GFX90A_REALTIME_TRACE_LAYER'] = str(a.trace_layer)
            env['SGLANG_DSV4_GFX90A_REALTIME_TRACE_LOG_EVERY'] = '16'
            env['SGLANG_DSV4_GFX90A_REALTIME_TRACE_GRAPH_ONLY'] = '1'
        stem = str(a.prefix)+'_'+name
        logfile = Path(stem+'_server.log')
        with logfile.open('w') as f:
            child = subprocess.Popen(['numactl','--interleave=all',*cmd],env=env,cwd=cwd,
                                     stdout=f,stderr=subprocess.STDOUT,start_new_session=True)
        parent = psutil.Process(child.pid)
        row = {'arm':name,'flag':flag,'parent':parent.pid,'server_log':str(logfile)}
        row['diagnostic_trace_layer'] = a.trace_layer
        report['arms'].append(row);report['parent']=parent.pid;save()
        print('starting',name,parent.pid,flush=True)
        for _ in range(120):
            assert parent.is_running() and parent.status()!=psutil.STATUS_ZOMBIE
            if 'Application startup complete' in logfile.read_text(errors='replace'):break
            time.sleep(5)
        else:raise RuntimeError('readiness deadline; inspect live process, do not restart blindly')
        policies = {}
        for rank in parent.children(recursive=True):
            if 'scheduler_TP' not in rank.name():continue
            counts = collections.Counter(x.split()[1] for x in Path(f'/proc/{rank.pid}/numa_maps').read_text().splitlines())
            assert any(k.startswith('interleave') for k in counts),counts
            policies[rank.name()] = dict(counts)
        assert len(policies)==8
        row['numa_policies']=policies;save()
        c1=stem+'_c1.json';c32=stem+'_c32.json'
        row['c1_start_line'] = len(logfile.read_text(errors='replace').splitlines())+1
        bench(['scripts/rocm/bench_dsv4_c1_mhc_recovery.py','--arm',name,'--rounds','2',
               '--skip-freeze-gc','--reference',str(a.reference),'--output',c1],stem+'_c1.log')
        result=json.loads(Path(c1).read_text())
        assert result['status']=='complete' and result['france_exact']
        assert all(x[k]==y[k] for x,y in zip(result['teacher_forced'],reference['teacher_forced'])
                   for k in ('output_ids','input_token_logprobs','output_top_logprobs'))
        row['c1']=c1;row['c1_medians']=result['medians'];save()
        row['c1_stop_line'] = len(logfile.read_text(errors='replace').splitlines())
        row['c32_start_line'] = row['c1_stop_line']+1
        bench(['scripts/rocm/bench_dsv4_tp4_diverse_concurrent.py','--base-url','http://127.0.0.1:30011',
               '--inputs',str(a.inputs),'--request-count','32','--tokens','256','--rounds',str(a.c32_rounds),
               '--output',c32],stem+'_c32.log')
        result=json.loads(Path(c32).read_text())
        row['c32_stop_line'] = len(logfile.read_text(errors='replace').splitlines())
        row.update(c32=c32,e2e_tok_s=result['median_tok_s'],decode_tok_s=result['resident_bs32_median_tok_s'])
        transition=stem+'_transitions.json'
        bench(['scripts/rocm/check_dsv4_prefill_probe_transitions.py',
               '--reference',str(a.reference),'--rounds','2','--output',transition],stem+'_transitions.log')
        checks=json.loads(Path(transition).read_text())
        assert checks['status']=='complete' and len(checks['responses'])==12
        assert all(x[k] for x in checks['responses']
                   for k in ('next_id_exact','input_logprobs_exact','output_logprobs_exact'))
        row['transitions']=transition
        if a.france_c32:
            sentinel = stem+'_france_c32.json'
            bench(['scripts/rocm/check_dsv4_france_c32.py', '--output', sentinel],
                  stem+'_france_c32.log')
            checks = json.loads(Path(sentinel).read_text())
            assert checks['exact_count'] == checks['request_count'] == 32
            row['france_c32'] = sentinel
        save();print('completed',name,row['e2e_tok_s'],row['decode_tok_s'],flush=True)
    report['status']='complete';save()


if __name__=='__main__':
    main()
